"""
Local web server for the standalone app.

Serves the single-page UI and a small JSON API over the Workspace and the
Assistant. Standard library only. It binds to 127.0.0.1, checks the Host header
(no DNS rebinding) and requires a per-launch token on every API call, so other
web pages open in the same browser can't drive the simulator or spend the
user's Claude API credit.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

from . import paths, settings
from ..data.elevation import DEFAULT_SPACING_FT
from .assistant import Assistant
from .importers import ElevationConfirmNeeded
from .workspace import InputError, Workspace, inputs_summary, results_series, results_summary

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class App:
    def __init__(self, assistant_client_factory=None):
        self.ws = Workspace()
        self.assistant = Assistant(self.ws, client_factory=assistant_client_factory)
        self.token = secrets.token_urlsafe(24)
        self.httpd: Optional[ThreadingHTTPServer] = None

    # ---- API dispatch -----------------------------------------------------------

    def handle(self, method: str, path: str, query: dict, body: bytes) -> Tuple[int, object]:
        ws, asst = self.ws, self.assistant
        q = {k: v[0] for k, v in query.items()}

        def j() -> dict:
            if not body:
                return {}
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict):
                raise InputError("expected a JSON object")
            return data

        route = (method, path)
        if route == ("GET", "/api/state"):
            return 200, {**ws.state(), "assistant_busy": asst.busy,
                         "assistant_entries": len(asst.transcript)}
        if route == ("GET", "/api/scenarios"):
            return 200, {"scenarios": ws.list_scenarios(),
                         "user_dir": paths.user_scenarios_dir()}
        if route == ("POST", "/api/scenario/load"):
            return 200, ws.load(j().get("id", ""))
        if route == ("GET", "/api/scenario"):
            return 200, inputs_summary(ws.require())
        if route == ("POST", "/api/scenario/update"):
            return 200, {"changed": ws.update_inputs(j().get("changes") or {})}
        if route == ("POST", "/api/scenario/meta"):
            d = j()
            ws.update_meta(d.get("name"), d.get("notes"))
            return 200, {"ok": True}
        if route == ("POST", "/api/scenario/save"):
            d = j()
            return 200, ws.save_as(d.get("name", ""), d.get("notes"), bool(d.get("overwrite")))
        if route == ("POST", "/api/scenario/import"):
            return self._import(q, body)
        if route == ("GET", "/api/intake"):
            return 200, ws.intake_view()
        if route == ("POST", "/api/intake"):
            return 200, ws.apply_intake(j().get("answers") or {})
        if route == ("GET", "/api/precheck"):
            return 200, ws.precheck()
        if route == ("GET", "/api/elevation"):
            return 200, ws.elevation_view(max_points=int(q.get("max_points", 800)))
        if route == ("POST", "/api/elevation/fetch"):
            d = j()
            return 200, ws.fetch_elevation(float(d.get("spacing_ft") or DEFAULT_SPACING_FT),
                                           bool(d.get("reverse")))
        if route == ("POST", "/api/run"):
            ws.start_run()
            return 200, {"ok": True}
        if route == ("GET", "/api/results"):
            res = ws.require_results()
            return 200, {"summary": results_summary(res), "series": results_series(res)}
        if route == ("GET", "/api/results/profile"):
            return 200, ws.profile_at(int(q.get("step", 0)))
        if route == ("POST", "/api/export"):
            out = ws.export(j().get("kind", ""))
            return 200, {"path": out, "url": f"/api/download?path={quote(out)}&token={self.token}"}
        if route == ("GET", "/api/settings"):
            return 200, settings.public_view()
        if route == ("POST", "/api/settings"):
            d = j()
            return 200, settings.update(d.get("api_key"), d.get("model"), bool(d.get("clear_key")),
                                        d.get("elevation_ask_first"))
        if route == ("GET", "/api/chat"):
            return 200, asst.view()
        if route == ("POST", "/api/chat"):
            asst.send(j().get("message", ""))
            return 200, asst.view()
        if route == ("POST", "/api/chat/reset"):
            asst.reset()
            return 200, asst.view()
        if route == ("POST", "/api/shutdown"):
            threading.Thread(target=self.shutdown, daemon=True).start()
            return 200, {"ok": True}
        return 404, {"error": f"no route {method} {path}"}

    def _import(self, q: dict, body: bytes) -> Tuple[int, object]:
        """?fetch_elevation=1 confirms a USGS lookup the user was asked about (409 below)."""
        kind = q.get("kind", "")
        fetch = {"1": True, "0": False}.get(q.get("fetch_elevation", ""))
        filename = os.path.basename(q.get("filename", "upload"))
        if not body:
            raise InputError("no file received")
        suffix = os.path.splitext(filename)[1] or ".dat"
        tmp_dir = tempfile.mkdtemp(prefix="purge_import_")
        tmp = os.path.join(tmp_dir, filename if suffix else filename + suffix)
        with open(tmp, "wb") as f:
            f.write(body)
        try:
            return 200, self.ws.new_from_import(tmp, kind, q.get("name") or os.path.splitext(filename)[0],
                                                fetch_elevation=fetch,
                                                spacing_ft=float(q.get("spacing_ft") or DEFAULT_SPACING_FT))
        except ElevationConfirmNeeded as e:
            return 409, {"error": str(e), "needs_elevation_confirm": True,
                         "points": e.points, "length_mi": round(e.length_mi, 2)}
        finally:
            try:
                os.remove(tmp)
                os.rmdir(tmp_dir)
            except OSError:
                pass

    def download_path(self, raw: str) -> Optional[str]:
        """Only files under the outputs folder can be downloaded."""
        root = os.path.realpath(paths.outputs_dir())
        p = os.path.realpath(raw or "")
        return p if p.startswith(root + os.sep) and os.path.isfile(p) else None

    # ---- serving ---------------------------------------------------------------------

    def serve(self, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "PurgeSimulator"

            def log_message(self, fmt, *args):   # keep the console quiet
                pass

            def _host_ok(self) -> bool:
                h = (self.headers.get("Host") or "").lower()
                port_s = str(self.server.server_address[1])
                return h in (f"127.0.0.1:{port_s}", f"localhost:{port_s}")

            def _send(self, code: int, payload: object, ctype="application/json") -> None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)

            def _dispatch(self, method: str) -> None:
                if not self._host_ok():
                    return self._send(403, {"error": "bad host"})
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if url.path in ("/", "/index.html"):
                    return self._index()
                if not url.path.startswith("/api/"):
                    return self._send(404, {"error": "not found"})
                token = self.headers.get("X-App-Token") or query.get("token", [""])[0]
                if not secrets.compare_digest(token, app.token):
                    return self._send(403, {"error": "missing or bad app token"})
                if url.path == "/api/download" and method == "GET":
                    return self._download(query.get("path", [""])[0])
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_UPLOAD_BYTES:
                    return self._send(413, {"error": "file too large"})
                body = self.rfile.read(length) if length else b""
                try:
                    code, payload = app.handle(method, url.path, query, body)
                except InputError as e:
                    code, payload = 400, {"error": str(e)}
                except (ValueError, KeyError) as e:
                    code, payload = 400, {"error": f"{type(e).__name__}: {e}"}
                except Exception as e:
                    traceback.print_exc()
                    code, payload = 500, {"error": f"{type(e).__name__}: {e}"}
                self._send(code, payload)

            def _index(self) -> None:
                with open(os.path.join(paths.static_dir(), "index.html"), "rb") as f:
                    html = f.read().replace(b"__APP_TOKEN__", app.token.encode())
                self._send(200, html, "text/html; charset=utf-8")

            def _download(self, raw: str) -> None:
                p = app.download_path(raw)
                if not p:
                    return self._send(404, {"error": "file not found"})
                with open(p, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(p)}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._dispatch("GET")

            def do_POST(self):
                self._dispatch("POST")

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        return self.httpd

    @property
    def url(self) -> str:
        assert self.httpd is not None
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/"

    def shutdown(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
