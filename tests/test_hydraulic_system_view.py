"""
A purge that stops short of the hydraulic endpoint (fluid_exit_mp) is drawn twice: the purged
segment, and the whole hydraulic system out to the endpoint, so high points past the pig stop
show whether they slack. Tests the HGL span, the app's profile view, the client page data and
the GIF frames.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

from purge_sim.app.client_report import build_data                 # noqa: E402
from purge_sim.app.workspace import Workspace                     # noqa: E402
from purge_sim.engine.hgl import compute_hgl, hydraulic_end_mp    # noqa: E402

TB = "bundled:Bridger_THUNDERBIRD/thunderbird_packcoast.json"     # 60.07 mi, pack and coast
STOP, ENDPOINT = 35.0, 60.069


@pytest.fixture(scope="module")
def midpoint():
    """Thunderbird purged to MP 35 with the liquid leaving at the far end of the line."""
    ws = Workspace()
    ws.load(TB)
    ws.update_inputs({"purge_end_mp": STOP, "fluid_exit_mp": ENDPOINT})
    res = ws.run()
    return ws, res


def test_hgl_full_system_reaches_the_endpoint(midpoint):
    _, res = midpoint
    cfg = res.config
    assert hydraulic_end_mp(cfg) == pytest.approx(ENDPOINT)
    step = res.steps[len(res.steps) // 2]
    seg = compute_hgl(step, cfg)
    full = compute_hgl(step, cfg, full_system=True)
    assert seg.mp[-1] == pytest.approx(STOP)
    assert full.mp[-1] == pytest.approx(ENDPOINT)
    assert not full.is_gas[full.mp > STOP + 1e-6].any()        # liquid-full past the pig stop
    # the purged segment's pressures are the same in both views
    common = full.mp <= STOP + 1e-9
    assert np.allclose(np.interp(seg.mp, full.mp[common], full.pressure_psig[common]),
                       seg.pressure_psig, atol=0.5)
    # the liquid leaves at the endpoint at the exit pressure
    assert full.pressure_psig[-1] == pytest.approx(cfg.exit_pressure_run_psig, abs=5.0)


def test_profile_view_carries_the_system(midpoint):
    ws, res = midpoint
    p = ws.profile_at(len(res.steps) // 2)
    assert p["pig_stop_mp"] == pytest.approx(STOP)
    assert p["hydraulic_end_mp"] == pytest.approx(ENDPOINT, abs=1e-3)
    v = p["system"]
    assert v and v["mp"][-1] == pytest.approx(ENDPOINT, abs=1e-3)
    assert len(v["mp"]) == len(v["pressure_psig"]) == len(v["is_gas"]) == len(v["elevation_ft"])
    low = v["downstream_low"]
    assert STOP < low["mp"] <= ENDPOINT + 1e-3
    assert low["below_floor"] == (low["psig"] < p["slack_floor_psig"])
    assert low["mp"] in v["mp"]                     # the low point survives the downsampling
    # rate floaters: N2 injection and liquid flow at this moment, with their maxima
    assert p["injection_scfm"] >= 0 and p["injection_max_scfm"] > 0
    assert 0 < p["flow_bph"] <= p["flow_max_bph"] * 1.5


def test_client_page_has_second_profile(midpoint):
    _, res = midpoint
    prof = build_data(res, "TB midpoint")["profile"]
    sysv = prof["system"]
    assert sysv["end_mp"] == pytest.approx(ENDPOINT, abs=0.01) and sysv["stop_mp"] == pytest.approx(STOP)
    assert len(sysv["frames"]) == len(prof["frames"])
    assert all(len(f["p"]) == len(sysv["mp"]) for f in sysv["frames"])
    assert all("low" in f for f in sysv["frames"])
    assert prof["inj_max"] > 0 and prof["flow_max"] > 0
    assert all(f["inj"] >= 0 and f["bph"] >= 0 for f in prof["frames"])


def test_no_system_view_without_a_downstream_endpoint():
    ws = Workspace()
    ws.load(TB)
    res = ws.run()
    assert hydraulic_end_mp(res.config) is None
    assert ws.profile_at(0)["system"] is None
    assert "system" not in build_data(res, "TB")["profile"]


def test_gif_frames_are_stacked(midpoint, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from PIL import Image
    from purge_sim.engine.animation import generate_animation
    _, res = midpoint
    out = tmp_path / "tb.gif"
    generate_animation(res, res.config, out_gif=str(out), n_frames=2)
    im = Image.open(out)
    assert im.height > im.width * 0.8           # two charts one above the other
    assert not (tmp_path / "tb_full.gif").exists()
