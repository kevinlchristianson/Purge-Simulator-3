"""
Tests for the built-in elevation lookup (purge_sim/data/elevation.py) and how the app
uses it: 2D-KMZ detection, route sampling, peak refinement, gap fill and flags, the
cache, the USGS response handling, and the import / re-fetch / assistant paths.

All offline: a fake DEM stands in for USGS 3DEP. One live smoke test runs only when
PURGE_SIM_LIVE_ELEVATION=1.
"""
import io
import json
import math
import os
import sys
import threading
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

from purge_sim.app import settings                               # noqa: E402
from purge_sim.app.assistant import execute_tool                 # noqa: E402
from purge_sim.app.importers import ElevationConfirmNeeded       # noqa: E402
from purge_sim.app.server import App                             # noqa: E402
from purge_sim.app.workspace import InputError, Workspace        # noqa: E402
from purge_sim.data import elevation as E                        # noqa: E402
from purge_sim.data.profile_parser import parse_kml_files, parse_profile   # noqa: E402

# A ~7 mi route in Montana with one bend.
ROUTE = [(45.000, -108.500), (45.050, -108.500), (45.080, -108.450), (45.100, -108.450)]
PEAK = (45.065, -108.475)   # on the second leg


def terrain(lat, lon):
    """Rolling ground plus a sharp ~150 ft knob about 1,500 ft wide at PEAK."""
    d_ft = math.hypot((lat - PEAK[0]) * 364000, (lon - PEAK[1]) * 258000)
    return 3000.0 + 80.0 * math.sin(lat * 900) + 150.0 * math.exp(-(d_ft / 700.0) ** 2)


class FakeDEM:
    name = "fake_dem"
    label = "Fake DEM"

    def __init__(self, f=terrain, nodata=lambda lat, lon: False):
        self.f, self.nodata = f, nodata
        self.points = 0

    def lookup(self, pts, progress_cb=None):
        self.points += len(pts)
        if progress_cb:
            progress_cb(1.0, f"{len(pts)}/{len(pts)} points")
        return [None if self.nodata(la, lo) else self.f(la, lo) for la, lo in pts]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PURGE_SIM_SCENARIOS_DIR", str(tmp_path / "scenarios"))
    monkeypatch.setenv("PURGE_SIM_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PURGE_SIM_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("PURGE_SIM_ELEV_CACHE", str(tmp_path / "elev.sqlite"))


@pytest.fixture
def dem(monkeypatch):
    d = FakeDEM()
    monkeypatch.setattr(E, "default_provider", lambda: d)
    return d


def write_kmz(path, coords, alt=None):
    trip = " ".join(f"{lon},{lat}" + ("" if alt is None else f",{alt(lat, lon)}") for lat, lon in coords)
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f'<Placemark><name>Line</name><LineString><coordinates>{trip}</coordinates></LineString></Placemark>'
           '</Document></kml>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("doc.kml", kml)
    return str(path)


# ---------------------------------------------------------------- parser

def test_2d_and_zero_altitude_kmz_need_elevation(tmp_path):
    for name, alt in (("flat2d.kmz", None), ("zeros.kmz", lambda la, lo: 0)):
        p = parse_profile(write_kmz(tmp_path / name, ROUTE, alt))
        assert p.elevation_status == "none" and p.needs_elevation
        assert np.isnan(p.elevations_ft).all()      # never a silent 0 ft profile
    p = parse_profile(write_kmz(tmp_path / "real.kmz", ROUTE, lambda la, lo: 900))
    assert p.elevation_status == "full" and np.allclose(p.elevations_ft, 900 * 3.28084)


# ---------------------------------------------------------------- sampling

def test_sampling_spacing_length_and_vertices():
    lat, lon = zip(*ROUTE)
    res = E.fetch_route_elevation(lat, lon, spacing_ft=250, provider=FakeDEM(), use_cache=False)
    length = E.route_mileposts(np.array(lat), np.array(lon))[-1]
    assert res.length_mi == pytest.approx(length, abs=1e-9)
    gaps = np.diff(res.mileposts) * E.FT_PER_MI
    assert gaps.max() <= 250 + 1e-6 and gaps.min() > 0
    for vla, vlo in ROUTE:   # every bend is sampled exactly
        assert np.min(np.hypot(res.lat - vla, res.lon - vlo)) < 1e-9
    assert res.source["provider"] == "fake_dem" and res.source["spacing_ft"] == 250
    assert len(res.profile()) == res.source["points"]


def test_dense_route_is_not_collapsed():
    # A KMZ with a vertex every ~35 ft (Cheyenne-Kaneb 762) used to come back as two points,
    # the route's ends, because each vertex replaced the one before it.
    n = 600
    lat = 41.10 + np.arange(n) * 1e-4          # ~36 ft apart, ~4.1 mi
    lon = np.full(n, -104.83)
    vmp = E.route_mileposts(lat, lon)
    mp = E.sample_mileposts(vmp, 250.0)
    gaps = np.diff(mp) * E.FT_PER_MI
    assert len(mp) >= vmp[-1] * E.FT_PER_MI / 250.0
    assert gaps.max() <= 250 + 1e-6 and gaps[:-1].min() >= 125 - 1e-6   # the last may be short


def test_peak_refinement_catches_a_narrow_crest(monkeypatch):
    lat, lon = zip(*ROUTE)
    true_peak = terrain(*PEAK)
    refined = E.fetch_route_elevation(lat, lon, spacing_ft=1000, provider=FakeDEM(), use_cache=False)
    monkeypatch.setattr(E, "MAX_REFINE_PASSES", 0)
    coarse = E.fetch_route_elevation(lat, lon, spacing_ft=1000, provider=FakeDEM(), use_cache=False)
    assert len(refined.mileposts) > len(coarse.mileposts)
    assert true_peak - refined.elevations_ft.max() < true_peak - coarse.elevations_ft.max()


def test_gaps_are_filled_and_flagged():
    lat, lon = zip(*ROUTE)
    dem = FakeDEM(nodata=lambda la, lo: 45.010 < la < 45.020)
    res = E.fetch_route_elevation(lat, lon, provider=dem, use_cache=False)
    assert res.filled_points > 0 and np.isfinite(res.elevations_ft).all()
    assert any("interpolated" in f for f in res.flags)


def test_no_coverage_raises():
    lat, lon = zip(*ROUTE)
    with pytest.raises(E.ElevationError, match="US only"):
        E.fetch_route_elevation(lat, lon, provider=FakeDEM(nodata=lambda la, lo: True), use_cache=False)


def test_flat_water_like_run_is_flagged():
    lat, lon = zip(*ROUTE)
    lake = FakeDEM(f=lambda la, lo: 2500.0 if 45.02 < la < 45.04 else terrain(la, lo))
    res = E.fetch_route_elevation(lat, lon, provider=lake, use_cache=False)
    assert any("Dead-flat" in f for f in res.flags)


def test_cache_reuses_earlier_lookups(tmp_path):
    lat, lon = zip(*ROUTE)
    cache = E.ElevationCache(str(tmp_path / "c.sqlite"))
    dem = FakeDEM()
    first = E.fetch_route_elevation(lat, lon, provider=dem, cache=cache)
    n = dem.points
    again = E.fetch_route_elevation(lat, lon, provider=dem, cache=cache)
    assert dem.points == n   # nothing new asked of the service
    assert np.allclose(first.elevations_ft, again.elevations_ft)


# ---------------------------------------------------------------- USGS provider

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_usgs_parsing_endpoint_fallback_and_unreachable():
    seen = []

    def urlopen(req, timeout):
        seen.append(req.full_url)
        if "/v1/json" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "gone", {}, None)
        lat = float(req.full_url.split("y=")[1].split("&")[0])
        return _Resp(json.dumps({"value": "-1000000" if lat > 50 else "1234.5"}).encode())

    p = E.USGS3DEP(workers=2, retries=1, urlopen=urlopen)
    assert p.lookup([(45.0, -108.5), (60.0, -108.5)]) == [1234.5, None]
    assert any("/v1/points" in u for u in seen)

    def down(req, timeout):
        raise urllib.error.URLError("no route to host")

    with pytest.raises(E.ElevationError, match="Couldn't reach"):
        E.USGS3DEP(workers=2, retries=0, urlopen=down).lookup([(45.0, -108.5)] * 10)


def test_keepalive_reuses_one_connection_per_worker():
    # Each worker keeps its connection open instead of a new TCP/TLS handshake per point.
    import http.server
    conns = set()

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            conns.add(self.client_address)
            if "/v1/json" in self.path:
                body, code = b"gone", 404
            else:
                body, code = json.dumps({"value": "1234.5"}).encode(), 200
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        p = E.USGS3DEP(workers=2, retries=1)
        p._keepalive = True
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        p.URLS = tuple(u.replace("https://epqs.nationalmap.gov", base) for u in E.USGS3DEP.URLS)
        assert p.lookup([(45.0 + i * 1e-3, -108.5) for i in range(40)]) == [1234.5] * 40
        assert len(conns) <= 4   # 2 workers (plus a reconnect after the 404 at most)
    finally:
        srv.shutdown()


def _samples_urlopen(calls, meters_of=lambda lat: 1000.0, point_ft=None, fail=False):
    """Fake urlopen for getSamples (POST, many points) and the point service (GET, one)."""
    import urllib.parse as up

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if "getSamples" in req.full_url:
            if fail:
                return _Resp(json.dumps({"error": {"code": 500, "message": "busy"}}).encode())
            form = up.parse_qs(req.data.decode())
            pts = json.loads(form["geometry"][0])["points"]
            return _Resp(json.dumps({"samples": [
                {"locationId": i, "value": "NoData" if y > 50 else str(meters_of(y))}
                for i, (x, y) in enumerate(pts)]}).encode())
        lat = float(req.full_url.split("y=")[1].split("&")[0])
        ft = point_ft(lat) if point_ft else meters_of(lat) * E.FT_PER_M
        return _Resp(json.dumps({"value": str(ft)}).encode())
    return urlopen


def test_samples_service_answers_many_points_per_request():
    calls = []
    p = E.USGS3DEP(workers=2, retries=0, urlopen=_samples_urlopen(calls))
    p.use_samples = True
    pts = [(45.0 + i * 1e-4, -108.5) for i in range(450)] + [(60.0, -108.5)]
    vals = p.lookup(pts)
    assert vals[:450] == [pytest.approx(1000.0 * E.FT_PER_M)] * 450 and vals[450] is None
    assert sum("getSamples" in u for u in calls) == 3          # 200 + 200 + 51
    assert sum("getSamples" not in u for u in calls) == 1      # the one units check


def test_samples_service_falls_back_to_point_service():
    calls = []   # the batch service errors: every point goes to the point service
    p = E.USGS3DEP(workers=2, retries=0, urlopen=_samples_urlopen(calls, fail=True))
    p.use_samples = True
    assert p.lookup([(45.0, -108.5), (45.1, -108.5)]) == [pytest.approx(3280.84)] * 2
    calls = []   # units disagree with the point service: trust the point service
    p = E.USGS3DEP(workers=2, retries=0, urlopen=_samples_urlopen(calls, point_ft=lambda lat: 1234.0))
    p.use_samples = True
    assert p.lookup([(45.0, -108.5), (45.1, -108.5)]) == [1234.0, 1234.0]


# ---------------------------------------------------------------- app paths

def test_import_2d_kmz_looks_up_elevation(tmp_path, dem):
    ws = Workspace()
    out = ws.new_from_import(write_kmz(tmp_path / "line.kmz", ROUTE), "profile", "Line")
    inp = ws.scenario.inputs
    assert inp.data_source == "KMZ+USGS"
    assert inp.elevation_source["provider"] == "fake_dem" and len(inp.route_latlon) >= len(ROUTE)
    assert out["elevation_source"]["points"] == len(inp.elevation_profile) and out["route_points"] > 1
    assert min(e for _, e in inp.elevation_profile) > 2800          # real ground, not 0 ft
    assert "Fake DEM" in ws.scenario.meta.notes
    assert ws.job["state"] == "idle"


def test_kmz_altitudes_in_meters_are_kept(tmp_path, dem):
    ws = Workspace()
    out = ws.new_from_import(write_kmz(tmp_path / "m.kmz", ROUTE, lambda la, lo: terrain(la, lo) / 3.28084),
                             "profile", "M")
    inp = ws.scenario.inputs
    assert inp.elevation_source["provider"] == "file" and inp.data_source == "KMZ"
    assert inp.elevation_source["altitude_check"]["verdict"] == "ok"
    assert dem.points <= 8 and out["warnings"] == []   # only the spot check was looked up


def test_kmz_altitudes_in_feet_are_read_as_feet(tmp_path, dem):
    # CHS Santa Rita 16": the KMZ's altitude field carried feet, which read 3.28x too high.
    ws = Workspace()
    out = ws.new_from_import(write_kmz(tmp_path / "f.kmz", ROUTE, terrain), "profile", "F")
    inp = ws.scenario.inputs
    assert inp.elevation_source["altitude_check"]["verdict"] == "feet"
    assert max(abs(e - terrain(la, lo)) for (_, e), (la, lo) in zip(inp.elevation_profile, ROUTE)) < 0.5
    assert any("in feet, not meters" in w for w in out["warnings"])
    assert "in feet, not meters" in ws.scenario.meta.notes


def test_kmz_altitudes_that_disagree_are_replaced(tmp_path, dem):
    ws = Workspace()
    out = ws.new_from_import(write_kmz(tmp_path / "g.kmz", ROUTE, lambda la, lo: 100), "profile", "G")
    inp = ws.scenario.inputs
    assert inp.data_source == "KMZ+USGS" and inp.elevation_source["provider"] == "fake_dem"
    assert inp.elevation_source["altitude_check"]["verdict"] == "mismatch"
    assert min(e for _, e in inp.elevation_profile) > 2800
    assert any("disagree with USGS 3DEP" in w for w in out["warnings"])


def test_kmz_altitudes_unchecked_when_offline(tmp_path, monkeypatch):
    class Down(FakeDEM):
        def lookup(self, pts, progress_cb=None):
            raise E.ElevationError("Couldn't reach the USGS elevation service")
    monkeypatch.setattr(E, "default_provider", lambda: Down())
    ws = Workspace()
    out = ws.new_from_import(write_kmz(tmp_path / "o.kmz", ROUTE, lambda la, lo: 900), "profile", "O")
    inp = ws.scenario.inputs
    assert inp.elevation_source["altitude_check"]["verdict"] == "unverified"
    assert inp.elevation_profile[0][1] == pytest.approx(900 * 3.28084, abs=0.1)
    assert any("Couldn't check" in w for w in out["warnings"])


def test_ask_first_setting_and_http_confirm(tmp_path, dem):
    settings.update(elevation_ask_first_value=True)
    path = write_kmz(tmp_path / "line.kmz", ROUTE)
    with pytest.raises(ElevationConfirmNeeded):
        Workspace().new_from_import(path, "profile", "Line")
    assert dem.points == 0

    app = App()
    httpd = app.serve(port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        data = open(path, "rb").read()

        def post(extra=""):
            req = urllib.request.Request(f"{app.url}api/scenario/import?kind=profile&filename=line.kmz{extra}",
                                         data=data, method="POST", headers={"X-App-Token": app.token})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        code, body = post()
        assert code == 409 and body["needs_elevation_confirm"] and "USGS" in body["error"]
        code, body = post("&fetch_elevation=1")
        assert code == 200 and body["elevation_source"]["provider"] == "fake_dem"
    finally:
        app.shutdown()
        httpd.server_close()


def test_refetch_reverse_and_assistant_confirmation(tmp_path, dem):
    ws = Workspace()
    ws.new_from_import(write_kmz(tmp_path / "line.kmz", ROUTE), "profile", "Line")
    first = ws.scenario.inputs.elevation_profile
    out = ws.fetch_elevation(spacing_ft=500, reverse=True)
    prof = ws.scenario.inputs.elevation_profile
    assert out["source"]["spacing_ft"] == 500 and len(prof) < len(first)
    assert prof[0][1] == pytest.approx(first[-1][1], abs=1.0)   # now starts at the old far end
    assert ws.dirty

    settings.update(elevation_ask_first_value=True)
    with pytest.raises(InputError, match="user_confirmed"):
        execute_tool(ws, "fetch_elevation", {"reason": "check"})
    res = execute_tool(ws, "fetch_elevation", {"reason": "check", "user_confirmed": True})
    assert res["points"] > 0


def test_refetch_needs_a_route():
    ws = Workspace()
    ws.load("bundled:CHS_TipvilleSantaRita_East10/tipville_east10_3mph.json")
    with pytest.raises(InputError, match="no stored route"):
        ws.fetch_elevation()


def test_elevation_fields_are_import_only(tmp_path, dem):
    ws = Workspace()
    ws.new_from_import(write_kmz(tmp_path / "line.kmz", ROUTE), "profile", "Line")
    for f in ("elevation_source", "route_latlon"):
        with pytest.raises(InputError, match="can't be edited"):
            ws.update_inputs({f: []})


@pytest.mark.skipif(os.environ.get("PURGE_SIM_LIVE_ELEVATION") != "1",
                    reason="live USGS lookup; set PURGE_SIM_LIVE_ELEVATION=1 to run")
def test_live_usgs_smoke():
    vals = E.USGS3DEP().lookup([(45.78, -108.50), (45.80, -108.52)])
    assert all(v is not None and 2500 < v < 5000 for v in vals)   # Billings, MT area


# ---------------------------------------------------------------- map (gis.py)

def _along(route, n):
    """n evenly spaced (milepost, lat, lon) points along the route."""
    lat, lon = [p[0] for p in route], [p[1] for p in route]
    vmp = E.route_mileposts(lat, lon)
    mp = np.linspace(0, vmp[-1], n)
    plat, plon = E.interpolate_route(vmp, lat, lon, mp)
    return list(zip(mp, plat, plon))


def test_map_view_places_route_and_stations(tmp_path, dem):
    ws = Workspace()
    ws.new_from_import(write_kmz(tmp_path / "line.kmz", ROUTE), "profile", "Line")
    ws.update_inputs({"pump_stations": [{"mp": 3.0, "name": "Mid PS", "suction_psig": 30}],
                      "check_valves": [{"mp": 3.0, "name": "Mid PS CV"}, {"mp": 99.0, "name": "Far CV"}]})
    v = ws.map_view()
    assert v["has_route"] and len(v["mp"]) == len(v["lat"]) == len(v["elevation_ft"])
    assert v["lat"][0] == pytest.approx(ROUTE[0][0]) and v["lat"][-1] == pytest.approx(ROUTE[-1][0])
    assert v["mp"][-1] == pytest.approx(ws.scenario.inputs.elevation_profile[-1][0], abs=1e-3)
    assert abs(v["mp_scale"] - 1) < 0.01
    kinds = {f["kind"]: f for f in v["features"]}
    assert kinds["pump"]["lat"] is not None
    assert [f["name"] for f in v["features"] if f["kind"] == "check_valve"] == ["Far CV"]   # CV at the pump folded in
    assert any("Far CV" in w for w in v["warnings"])                                         # MP 99 is off the route


def test_attach_route_keeps_profile_and_picks_direction(tmp_path):
    ws = Workspace()
    ws.load("bundled:CHS_TipvilleSantaRita_East10/tipville_east10_3mph.json")
    assert not ws.map_view()["has_route"]
    inp = ws.scenario.inputs
    inp.elevation_profile = [[float(m), terrain(la, lo)] for m, la, lo in _along(ROUTE, 200)]
    before = [list(p) for p in inp.elevation_profile]
    # the file is drawn from the far end; its own elevations (metres in KML) show that
    drawn = [(la, lo) for _, la, lo in _along(ROUTE, 40)][::-1]
    path = write_kmz(tmp_path / "rev.kmz", drawn, lambda la, lo: terrain(la, lo) / 3.28084)
    out = ws.attach_route(path)
    assert out["notes"][0].startswith("Direction picked from the file's elevations: reversed") and ws.dirty
    assert inp.elevation_profile == before
    assert inp.route_latlon[0] == pytest.approx(list(ROUTE[0]), abs=1e-6)
    assert "rev.kmz" in ws.scenario.meta.source_files
    ws.attach_route(path, "as_is")
    assert inp.route_latlon[0] == pytest.approx(list(ROUTE[-1]), abs=1e-6)
    with pytest.raises(InputError, match="direction"):
        ws.attach_route(path, "sideways")


def test_map_http_route_upload_and_static_assets(tmp_path):
    app = App()
    httpd = app.serve(port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        app.ws.load("bundled:CHS_TipvilleSantaRita_East10/tipville_east10_3mph.json")
        with open(write_kmz(tmp_path / "line.kmz", ROUTE), "rb") as f:
            data = f.read()
        req = urllib.request.Request(f"{app.url}api/route/import?filename=line.kmz&direction=as_is", data=data,
                                     method="POST", headers={"X-App-Token": app.token})
        with urllib.request.urlopen(req, timeout=30) as r:
            assert json.loads(r.read())["route_points"] >= len(ROUTE)
        req = urllib.request.Request(f"{app.url}api/map", headers={"X-App-Token": app.token})
        with urllib.request.urlopen(req, timeout=30) as r:
            assert json.loads(r.read())["has_route"]
        with urllib.request.urlopen(f"{app.url}static/vendor/leaflet/leaflet.js", timeout=10) as r:
            assert r.headers["Content-Type"].startswith("text/javascript") and b"Leaflet" in r.read(2000)
        for bad in ("static/../server.py", "static/%2e%2e/server.py", "static/nope.js"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(app.url + bad, timeout=10)
            assert e.value.code in (403, 404)
    finally:
        app.shutdown()
        httpd.server_close()


def test_zero_and_out_of_range_coordinates_are_dropped(tmp_path, dem):
    """GL-09's PxP sheet has a few rows with x = y = 0; they drew the route to Africa."""
    lat = [45.0, 45.05, 0.0, 45.08, 95.0, 45.1]
    lon = [-108.5, -108.5, 0.0, -108.45, -108.45, -108.45]
    assert list(E.valid_coords(lat, lon)) == [True, True, False, True, False, True]
    assert [p[0] for p in E.thin_route(lat, lon)] == [45.0, 45.05, 45.08, 45.1]
    ws = Workspace()
    ws.new_from_import(write_kmz(tmp_path / "line.kmz", ROUTE), "profile", "Line")
    inp = ws.scenario.inputs
    inp.route_latlon = inp.route_latlon[:2] + [[0.0, 0.0]] + inp.route_latlon[2:]   # saved before the fix
    inp.purge_start_mp = round(inp.elevation_profile[0][0] - 0.0004, 3)               # rounded a hair outside
    v = ws.map_view()
    assert v["route_length_mi"] < 10 and abs(v["mp_scale"] - 1) < 0.01 and not v["warnings"]
    assert all(f["lat"] is not None for f in v["features"])


# ---------------------------------------------------------------- KMZs with several lines

def write_multi_kmz(path, lines, pins=()):
    """lines: [(name, [(lat, lon), ...])]; pins: [(name, lat, lon)]."""
    pms = "".join(f'<Placemark><name>{n}</name><LineString><coordinates>'
                  + " ".join(f"{lo},{la},0" for la, lo in c) + '</coordinates></LineString></Placemark>'
                  for n, c in lines)
    pms += "".join(f'<Placemark><name>{n}</name><Point><coordinates>{lo},{la},0</coordinates></Point></Placemark>'
                   for n, la, lo in pins)
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f'<Folder>{pms}</Folder></Document></kml>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("doc.kml", kml)
    return str(path)


# Thunderbird Purges.kmz in miniature: two lines drawn outward from a shared junction,
# one long (south to "Reno"), one shorter (north to "Belle Creek", via "Elk Creek").
JCT = (45.0, -108.5)
SOUTH = [JCT, (44.95, -108.52), (44.90, -108.55), (44.85, -108.56)]
NORTH = [JCT, (45.03, -108.49), (45.06, -108.47)]
PINS = [("Reno", 44.8501, -108.5601), ("Elk Creek", 45.03, -108.4901), ("Belle Creek", 45.0601, -108.47),
        ("Far away", 46.0, -108.0)]


def test_kmz_lines_are_joined_end_to_end(tmp_path):
    p = parse_profile(write_multi_kmz(tmp_path / "tb.kmz", [("S", SOUTH), ("N", NORTH)], PINS))
    # longest line keeps its drawn direction; the other is flipped onto its start
    assert (p.lat[0], p.lon[0]) == pytest.approx(NORTH[-1]) and (p.lat[-1], p.lon[-1]) == pytest.approx(SOUTH[-1])
    assert len(p.lat) == len(SOUTH) + len(NORTH) - 1           # the shared junction point once
    assert "2 were joined" in p.notes[0]
    names = [x["name"] for x in p.pins]
    assert names == ["Belle Creek", "Elk Creek", "Reno"]       # in milepost order; "Far away" dropped
    assert p.pins[0]["mp"] < 0.05 and abs(p.pins[-1]["mp"] - p.total_length_mi) < 0.05
    assert any("Far away" in n for n in p.notes)
    assert [q["name"] for q in p.parts] == ["S", "N"]
    assert sum(q["length_mi"] for q in p.parts) == pytest.approx(p.total_length_mi)


def test_kmz_gap_bridged_and_lateral_left_out(tmp_path):
    a = [(45.0, -108.5), (45.05, -108.5)]
    b = [(45.0505, -108.5), (45.10, -108.5)]                  # ~180 ft past a's end
    lateral = [(45.025, -108.45), (45.025, -108.40)]           # off the middle, miles from either end
    p = parse_profile(write_multi_kmz(tmp_path / "g.kmz", [("A", a), ("B", b), ("Lat", lateral)]))
    assert len(p.lat) == 4 and p.lat[0] == pytest.approx(45.0) and p.lat[-1] == pytest.approx(45.10)
    assert any("ft gap bridged" in n for n in p.notes)
    assert any("Left out" in n and '"Lat"' in n for n in p.notes)


def test_import_multi_line_kmz_adds_pins_as_map_landmarks(tmp_path, dem):
    ws = Workspace()
    ws.new_from_import(write_multi_kmz(tmp_path / "tb.kmz", [("S", SOUTH), ("N", NORTH)], PINS), "profile", "TB")
    inp = ws.scenario.inputs
    end = inp.elevation_profile[-1][0]
    assert inp.purge_end_mp == pytest.approx(end, abs=1e-3)
    assert [lm["kind"] for lm in inp.landmarks] == ["map_pin"] * 3
    assert inp.landmarks[-1]["name"] == "Reno" and abs(inp.landmarks[-1]["mp"] - end) < 0.05
    assert "2 were joined" in ws.scenario.meta.notes
    pins = [f for f in ws.map_view()["features"] if f["kind"] == "map_pin"]
    assert len(pins) == 3 and all(f["lat"] is not None for f in pins)


def test_add_route_picks_the_line_that_fits_the_scenario(tmp_path, dem):
    ws = Workspace()
    ws.new_from_import(write_kmz(tmp_path / "s.kmz", SOUTH), "profile", "South only")
    ws.scenario.inputs.route_latlon = []
    out = ws.attach_route(write_multi_kmz(tmp_path / "tb.kmz", [("S", SOUTH), ("N", NORTH)]), "as_is")
    assert out["notes"][0].startswith('The file has 2 lines. Used "S"')
    assert ws.scenario.inputs.route_latlon[-1] == pytest.approx(list(SOUTH[-1]), abs=1e-6)


# ------------------------------------- one MultiGeometry line plus separate Start / End files

def write_multigeom_kmz(path, lines):
    """All lines in one Placemark's MultiGeometry, as a GIS export writes them."""
    geoms = "".join('<LineString><coordinates>' + " ".join(f"{lo},{la},0" for la, lo in c)
                    + '</coordinates></LineString>' for c in lines)
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f'<Placemark><name>WE-6</name><MultiGeometry>{geoms}</MultiGeometry></Placemark></Document></kml>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("doc.kml", kml)
    return str(path)


# University to Fullerton in miniature: pieces drawn west to east in file order, except the
# last, which starts where the first does and runs west. Start is at the east end, End partway.
WE6 = [[(32.427, -102.807), (32.420, -102.700), (32.419, -102.651)],
       [(32.419, -102.651), (32.412, -102.529)],
       [(32.412, -102.529), (32.413, -102.450), (32.415, -102.373)],
       [(32.427, -102.807), (32.409, -102.887)]]


def test_multigeometry_pieces_chain_and_start_end_files_orient(tmp_path):
    line = write_multigeom_kmz(tmp_path / "we6.kmz", WE6)
    p = parse_profile(line)
    assert (p.lon[0], p.lon[-1]) == pytest.approx((-102.887, -102.373))
    assert np.all(np.diff(p.lon) > 0)                          # one line, no doubling back
    start = write_multi_kmz(tmp_path / "Start.kmz", [], [("Start", 32.415, -102.373)])
    end = write_multi_kmz(tmp_path / "End.kmz", [], [("End", 32.413, -102.450)])
    p = parse_kml_files([line, start, end])
    assert p.lon[0] == pytest.approx(-102.373) and np.all(np.diff(p.lon) < 0)   # reversed: Start first
    assert p.launch_mp == pytest.approx(0.0, abs=0.01)
    assert 4 < p.stop_mp < 5 and p.total_length_mi > 25
    assert any("reversed" in n for n in p.notes) and any("past End" in n for n in p.notes)


def test_import_with_start_end_files_sets_launch_and_stop(tmp_path, dem):
    line = write_multigeom_kmz(tmp_path / "we6.kmz", WE6)
    start = write_multi_kmz(tmp_path / "Start.kmz", [], [("Start", 32.415, -102.373)])
    end = write_multi_kmz(tmp_path / "End.kmz", [], [("End", 32.413, -102.450)])
    ws = Workspace()
    ws.new_from_import(line, "auto", "WE-6", extra_paths=[start, end])
    sc = ws.scenario
    inp = sc.inputs
    assert inp.purge_start_mp == pytest.approx(inp.elevation_profile[0][0], abs=0.01)
    assert 4 < inp.purge_end_mp < 5 < inp.elevation_profile[-1][0]   # route kept past the pig stop
    assert inp.route_latlon[0][1] == pytest.approx(-102.373, abs=1e-3)
    assert [lm["name"] for lm in inp.landmarks] == ["Start", "End"]
    assert sc.meta.source_files == ["we6.kmz", "Start.kmz", "End.kmz"]
    with pytest.raises(Exception, match="Only KMZ / KML"):
        ws.new_from_import(line, "auto", "x", extra_paths=[str(tmp_path / "x.csv")])


def test_server_imports_several_files_in_one_body(tmp_path, dem):
    files = [write_multigeom_kmz(tmp_path / "we6.kmz", WE6),
             write_multi_kmz(tmp_path / "Start.kmz", [], [("Start", 32.415, -102.373)]),
             write_multi_kmz(tmp_path / "End.kmz", [], [("End", 32.413, -102.450)])]
    blobs = [open(f, "rb").read() for f in files]
    q = {"kind": "auto", "filenames": "we6.kmz|Start.kmz|End.kmz", "sizes": ",".join(str(len(b)) for b in blobs),
         "fetch_elevation": "1"}
    app = App()
    code, body = app.handle("POST", "/api/scenario/import", {k: [v] for k, v in q.items()}, b"".join(blobs))
    assert code == 200 and 4 < app.ws.scenario.inputs.purge_end_mp < 5
    with pytest.raises(InputError, match="match their sizes"):
        app.handle("POST", "/api/scenario/import", {k: [v] for k, v in {**q, "sizes": "1,2,3"}.items()},
                   b"".join(blobs))


def test_several_files_still_ask_before_elevation_lookup(tmp_path, dem):
    settings.update(elevation_ask_first_value=True)
    files = [write_multigeom_kmz(tmp_path / "we6.kmz", WE6),
             write_multi_kmz(tmp_path / "Start.kmz", [], [("Start", 32.415, -102.373)])]
    blobs = [open(f, "rb").read() for f in files]
    q = {"kind": ["auto"], "filenames": ["we6.kmz|Start.kmz"], "sizes": [",".join(str(len(b)) for b in blobs)]}
    app = App()
    code, body = app.handle("POST", "/api/scenario/import", q, b"".join(blobs))
    assert code == 409 and body["needs_elevation_confirm"] and dem.points == 0
    code, body = app.handle("POST", "/api/scenario/import", {**q, "fetch_elevation": ["1"]}, b"".join(blobs))
    assert code == 200
