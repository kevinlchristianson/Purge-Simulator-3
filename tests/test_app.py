"""
Tests for the standalone app: the workspace (validation and hard-rule checks), the
local HTTP API, and the assistant's tool loop driven by a scripted fake Claude client
(no network, no API key).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

from purge_sim.app import settings                              # noqa: E402
from purge_sim.app.assistant import Assistant, TOOLS, execute_tool  # noqa: E402
from purge_sim.app.server import App                            # noqa: E402
from purge_sim.app.workspace import InputError, Workspace       # noqa: E402

SHORT = "bundled:CHS_TipvilleSantaRita_East10/tipville_east10_3mph.json"
LOW_FLOOR = "bundled:PMPL_SPtoMT/sp_to_mt_shls_packcoast_48m.json"   # signed-off 45 psi floor


@pytest.fixture(autouse=True)
def _isolated_user_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("PURGE_SIM_SCENARIOS_DIR", str(tmp_path / "scenarios"))
    monkeypatch.setenv("PURGE_SIM_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PURGE_SIM_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PURGE_SIM_MODEL", raising=False)


# ---------------------------------------------------------------- workspace

def test_library_lists_only_real_scenarios():
    items = Workspace().list_scenarios()
    ids = {s["id"] for s in items}
    assert SHORT in ids
    assert not any(i.endswith(("elev_ph30.json", "ph30_route_meta.json")) for i in ids)


def test_edit_validation_and_booster_floor():
    ws = Workspace()
    ws.load(SHORT)
    assert ws.update_inputs({"target_speed_mph": 2.5}) == [{"field": "target_speed_mph", "old": 3.0, "new": 2.5}]
    with pytest.raises(InputError, match="hard rule 4"):
        ws.update_inputs({"spread_suction_min_psig": 50})
    with pytest.raises(InputError, match="can't be edited"):
        ws.update_inputs({"elevation_profile": []})
    with pytest.raises(InputError, match="unknown input field"):
        ws.update_inputs({"not_a_field": 1})
    with pytest.raises(InputError, match="exit_pressure_behavior"):
        ws.update_inputs({"exit_pressure_behavior": "nope"})
    with pytest.raises(InputError, match="purge_end_mp"):
        ws.update_inputs({"purge_end_mp": -1})
    # A rejected edit changes nothing.
    assert ws.scenario.inputs.spread_suction_min_psig == 150.0


def test_signed_off_low_floor_stays_editable():
    ws = Workspace()
    ws.load(LOW_FLOOR)
    assert ws.scenario.inputs.spread_suction_min_psig < 100
    ws.update_inputs({"target_speed_mph": 2.0})            # unrelated edit is fine
    with pytest.raises(InputError):
        ws.update_inputs({"spread_suction_min_psig": 40})   # but no lowering through the app


def test_run_summary_and_profile():
    ws = Workspace()
    ws.load(SHORT)
    res = ws.run()
    assert res.completed
    prof = ws.profile_at(len(res.steps) // 2)
    assert len(prof["mp"]) == len(prof["pressure_psig"]) == len(prof["elevation_ft"])
    assert any(prof["is_gas"]) and not all(prof["is_gas"])


def test_save_never_overwrites_without_flag():
    ws = Workspace()
    ws.load(SHORT)
    out = ws.save_as("my variant", notes="why")
    assert os.path.isfile(out["path"]) and out["id"] == "user:my variant.json"
    with pytest.raises(InputError, match="already exists"):
        ws.save_as("my variant")
    with pytest.raises(InputError):
        ws._resolve("bundled:../../etc/passwd")


def test_client_html_and_all_reports_zip():
    import zipfile
    ws = Workspace()
    ws.load(SHORT)
    ws.update_meta(notes="internal tender note </script><b>x</b>")
    ws.run()
    ws.update_inputs({"target_speed_mph": 2.0})     # edits after the run don't leak into its reports
    page = open(ws.export("html"), encoding="utf-8").read()
    assert "__REPORT_" not in page and "tender note" not in page   # notes are opt-in
    blob = page.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    data = json.loads(blob)
    assert data["summary"]["completed"] and len(data["profile"]["frames"]) > 10
    assert len(data["fill"]["rows"]) == 50 and "<" not in blob
    assert "target 3 mph" in dict(data["basis"])["Pig speed"]
    page = open(ws.export("html", include_notes=True), encoding="utf-8").read()
    assert "</script><b>" not in page and "tender note" in page
    z = ws.export("all", include_gif=False)
    names = zipfile.ZipFile(z).namelist()
    assert z.endswith("_all_reports.zip") and len(names) == 5
    assert {os.path.splitext(n)[1] for n in names} == {".html", ".xlsx", ".txt", ".json"}
    with pytest.raises(InputError, match="unknown export kind"):
        ws.export("pdf")


# ---------------------------------------------------------------- assistant (fake client)

def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(i, name, args):
    return SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=args)


class FakeClient:
    """Scripted stand-in for anthropic.Anthropic: returns queued responses, records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        content, stop = self.responses.pop(0)
        return SimpleNamespace(content=content, stop_reason=stop)


def test_assistant_tool_loop_edits_runs_and_respects_rules():
    ws = Workspace()
    ws.load(SHORT)
    fake = FakeClient([
        ([_text("Lowering speed and trying a low suction floor."),
          _tool(1, "update_inputs", {"changes": {"target_speed_mph": 2.5}, "reason": "test"}),
          _tool(2, "update_inputs", {"changes": {"spread_suction_min_psig": 20}, "reason": "test"})], "tool_use"),
        ([_tool(3, "run_simulation", {})], "tool_use"),
        ([_text("Done: completed.")], "end_turn"),
    ])
    a = Assistant(ws, client_factory=lambda: fake)
    a.send("slow it down and run", background=False)

    assert ws.scenario.inputs.target_speed_mph == 2.5
    assert ws.scenario.inputs.spread_suction_min_psig == 150.0       # rejected
    assert ws.results is not None and ws.results.completed

    # Both tool results went back in ONE user message, the rejected one flagged as an error.
    second = fake.requests[1]["messages"][-1]
    assert second["role"] == "user" and len(second["content"]) == 2
    assert second["content"][1]["is_error"] and "hard rule 4" in second["content"][1]["content"]

    req = fake.requests[0]
    assert req["model"] == settings.DEFAULT_MODEL
    assert req["thinking"] == {"type": "adaptive"}
    assert req["fallbacks"] == "default"
    assert {t["name"] for t in req["tools"]} == {t["name"] for t in TOOLS}

    roles = [m["role"] for m in a.transcript]
    assert roles[0] == "user" and roles[-1] == "assistant"
    assert any("rejected" in m["text"] for m in a.transcript if m["role"] == "tool")
    assert not a.busy


def test_assistant_error_rewinds_turn():
    ws = Workspace()

    def boom():
        raise RuntimeError("No Claude API key is set.")

    a = Assistant(ws, client_factory=boom)
    a.send("hello", background=False)
    assert a.messages == []
    assert a.transcript[-1]["role"] == "error" and "API key" in a.transcript[-1]["text"]


def test_sweep_leaves_open_scenario_unchanged():
    ws = Workspace()
    ws.load(SHORT)
    out = execute_tool(ws, "run_sweep", {"field": "target_speed_mph", "values": [2.5, 3.0]})
    assert [r["target_speed_mph"] for r in out["rows"]] == [2.5, 3.0]
    assert all("n2_total_fresh_scf" in r for r in out["rows"])
    assert ws.scenario.inputs.target_speed_mph == 3.0 and ws.results is None


# ---------------------------------------------------------------- settings

def test_settings_never_expose_key(monkeypatch):
    assert settings.public_view()["api_key_set"] is False
    v = settings.update("sk-ant-test-1234", "claude-opus-5-5")
    assert v == {"api_key_set": True, "api_key_source": "settings", "api_key_hint": "…1234",
                 "model": "claude-opus-5-5", "elevation_ask_first": False}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-9999")
    assert settings.api_key() == "sk-env-9999" and settings.public_view()["api_key_source"] == "environment"


# ---------------------------------------------------------------- HTTP API

@pytest.fixture
def server():
    app = App(assistant_client_factory=lambda: FakeClient([([_text("hi")], "end_turn")]))
    httpd = app.serve(port=0)
    import threading
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app
    app.shutdown()
    httpd.server_close()


def _call(app, path, body=None, token=True, host=None):
    req = urllib.request.Request(app.url.rstrip("/") + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None)
    if token:
        req.add_header("X-App-Token", app.token)
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_requires_token_and_local_host(server):
    assert _call(server, "/api/state", token=False)[0] == 403
    assert _call(server, "/api/state", host="evil.example:80")[0] == 403
    with urllib.request.urlopen(server.url, timeout=10) as r:
        html = r.read().decode()
    assert server.token in html and "__APP_TOKEN__" not in html


def test_http_guide_tab_serves_user_guide(server):
    code, g = _call(server, "/api/guide")
    assert code == 200 and g["markdown"].startswith("# Purge Simulator")


def test_http_load_edit_run_export_chat(server):
    assert _call(server, "/api/scenario/load", {"id": SHORT})[0] == 200
    code, err = _call(server, "/api/scenario/update", {"changes": {"spread_suction_min_psig": 10}})
    assert code == 400 and "hard rule 4" in err["error"]
    assert _call(server, "/api/run", {})[0] == 200
    for _ in range(300):
        st = _call(server, "/api/state")[1]
        if st["job"]["state"] != "running":
            break
        time.sleep(0.2)
    assert st["job"]["state"] == "done" and st["has_results"]
    code, res = _call(server, "/api/results")
    assert code == 200 and res["summary"]["completed"] and len(res["series"]["pig_mp"]) > 10
    code, exp = _call(server, "/api/export", {"kind": "log"})
    assert code == 200 and os.path.isfile(exp["path"])
    assert _call(server, "/api/chat", {"message": "hello"})[0] == 200
    for _ in range(50):
        chat = _call(server, "/api/chat")[1]
        if not chat["busy"]:
            break
        time.sleep(0.1)
    assert chat["transcript"][-1] == {**chat["transcript"][-1], "role": "assistant", "text": "hi"}


def test_http_scenario_download(server):
    req = urllib.request.Request(server.url.rstrip("/") + "/api/scenario/download?id=" + SHORT
                                 + "&token=" + server.token)
    with urllib.request.urlopen(req, timeout=10) as r:
        assert 'filename="tipville_east10_3mph.json"' in r.headers["Content-Disposition"]
        assert "inputs" in json.loads(r.read())
    for bad in ("bundled:../app.py", "bundled:CHS_TipvilleSantaRita_East10/nope.json", "nope"):
        assert _call(server, "/api/scenario/download?id=" + bad)[0] == 404
    assert _call(server, "/api/scenario/download?id=" + SHORT, token=False)[0] == 403


def test_assistant_load_refuses_to_discard_unsaved_edits():
    ws = Workspace()
    ws.load(SHORT)
    ws.update_inputs({"target_speed_mph": 2.5})
    with pytest.raises(InputError, match="unsaved"):
        execute_tool(ws, "load_scenario", {"scenario_id": LOW_FLOOR})
    execute_tool(ws, "load_scenario", {"scenario_id": LOW_FLOOR, "discard_unsaved": True})
    assert ws.scenario_id == LOW_FLOOR
