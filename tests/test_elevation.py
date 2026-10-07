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
from purge_sim.data.profile_parser import parse_profile          # noqa: E402

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


def test_import_with_elevations_skips_lookup(tmp_path, dem):
    ws = Workspace()
    ws.new_from_import(write_kmz(tmp_path / "z.kmz", ROUTE, lambda la, lo: 900), "profile", "Z")
    assert dem.points == 0 and ws.scenario.inputs.elevation_source["provider"] == "file"
    assert ws.scenario.inputs.data_source == "KMZ"


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
