"""
Friction factor (Churchill 1977): one smooth curve through laminar, transitional and
turbulent flow, so the pig solver always finds a true drive = resistance balance instead of
parking at a jump. The old engine switched from 64/Re to Swamee-Jain at Re 2300 and held the
Timber Creek pig at exactly 2.567 mph (the Re 2300 speed) from MP 4.8 to 11.1.
"""
import math
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from purge_sim.engine.physics import (                                         # noqa: E402
    darcy_friction_factor, darcy_friction_factor_array, flow_regime, liquid_friction_loss_psi,
    liquid_reynolds, mph_to_fts)
from purge_sim.engine.pig_solver import PigSolverConfig, solve_pig_speed      # noqa: E402

TIMBER_CREEK = os.path.join(REPO_ROOT, "scenarios", "Bridger_TimberCreek", "timber_creek_packcoast_80cst.json")
RR = 0.00015 / (6.313 / 12.0)          # Timber Creek relative roughness


def _swamee_jain(Re, rr):
    return 0.25 / math.log10(rr / 3.7 + 5.74 / Re ** 0.9) ** 2


def test_laminar_matches_hagen_poiseuille():
    for Re in (10.0, 100.0, 500.0, 1000.0, 1500.0, 1900.0):
        assert darcy_friction_factor(Re, RR) == pytest.approx(64.0 / Re, rel=0.005)


@pytest.mark.parametrize("rr", [1e-6, 1e-5, 1e-4, 3.4e-4, 1e-3, 1e-2])
def test_turbulent_matches_swamee_jain(rr):
    for Re in np.logspace(math.log10(5000.0), 8, 60):
        assert darcy_friction_factor(Re, rr) == pytest.approx(_swamee_jain(Re, rr), rel=0.002)


def test_no_jump_and_pressure_drop_rises_with_speed():
    Re = np.logspace(2.0, 5.0, 200000)
    f = darcy_friction_factor_array(Re, RR)
    # continuous: the log-log slope stays bounded (Churchill peaks at about +1.75 near
    # Re 2,500); the old switch was a vertical step
    slope = np.diff(np.log(f)) / np.diff(np.log(Re))
    assert slope.min() > -1.01 and slope.max() < 2.0
    # pressure drop ~ f * Re^2 rises everywhere, so drive -> speed is one-to-one
    assert np.all(np.diff(f * Re ** 2) > 0)


def test_scalar_and_array_agree():
    Re = np.array([1.0, 50.0, 2000.0, 2300.0, 3000.0, 1e5, 1e7])
    arr = darcy_friction_factor_array(Re, RR)
    for r, a in zip(Re, arr):
        assert darcy_friction_factor(r, RR) == pytest.approx(a, rel=1e-12)


def test_flow_regime_labels():
    assert flow_regime(1500) == "laminar"
    assert flow_regime(2300) == "transitional"
    assert flow_regime(5000) == "turbulent"
    # 80 cSt in 6.313 in ID at 2.567 mph is Re 2300
    assert liquid_reynolds(6.313 / 12.0, mph_to_fts(2.567), 80.0) == pytest.approx(2300, rel=0.002)


def test_solver_balances_through_the_transition():
    """Across drives that put the pig in Re 1,300-4,500, the solved speed balances drive and
    resistance within 0.5 psi, and speed rises steadily with drive (no parked speed)."""
    cfg = PigSolverConfig(od_in=6.625, wt_in=0.156, roughness_ft=0.00015, sg=0.82, viscosity_cst=80.0,
                          min_speed_mph=1.0, max_speed_mph=6.0, target_speed_mph=3.0)
    D_ft = (6.625 - 2 * 0.156) / 12.0
    exit_mp, exit_psig = 5.0, 50.0
    speeds = []
    for drive in np.linspace(130.0, 330.0, 60):
        r = solve_pig_speed(cfg, 0.0, 1000.0, exit_psig, exit_mp, "tank", float(drive), lambda mp: 1000.0)
        assert not r.slack_line_risk and not r.overspeed
        resist = exit_psig + liquid_friction_loss_psi(exit_mp * 5280.0, D_ft, r.pig_speed_fts,
                                                       cfg.sg, cfg.viscosity_cst, cfg.roughness_ft)
        if r.pig_speed_mph < cfg.target_speed_mph * 0.999 or r.pig_speed_mph > cfg.target_speed_mph * 1.001:
            assert resist == pytest.approx(drive, abs=0.5)
        speeds.append(r.pig_speed_mph)
    speeds = np.array(speeds)
    assert speeds[0] < 2.3 and speeds[-1] > 3.0           # the sweep spans the transition
    assert np.all(np.diff(speeds) > 0)                     # strictly rising, never parked
    # nothing piles up at the old Re 2300 speed
    assert np.sum(np.abs(speeds - 2.567) < 0.005) <= 1


def test_timber_creek_no_plateau_at_re_2300():
    from purge_sim.data.scenario import load_scenario
    from purge_sim.engine.config_builder import build_sim_config
    from purge_sim.engine.simulator import simulate

    res = simulate(build_sim_config(load_scenario(TIMBER_CREEK).inputs))
    assert res.completed
    steps = [s for s in res.steps if 5.0 <= s.pig_mp <= 11.0]
    speeds = np.array([s.pig_speed_mph for s in steps])
    # old engine: every step here at exactly 2.567 mph
    assert np.mean(np.abs(speeds - 2.567) < 0.002) < 0.1
    assert speeds.max() - speeds.min() > 0.2
    assert sum(s.slack_line_risk for s in res.steps) == 0
    assert res.total_scf_vented == 0
    assert {"transitional"} <= {s.flow_regime for s in steps}
