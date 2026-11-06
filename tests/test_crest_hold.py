"""
When a high point ahead, not the exit, sets the drive, the liquid past the crest would run
down to the exit at its minimum, so the exit has to hold the column back to keep the crest
at the slack floor. The pig solver already sizes the drive that way; these tests check the
exit pressure it reports is the one that hydraulics implies (and that a fixed exit, which
can't hold anything back, flags the crest as going slack).
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from purge_sim.engine.physics import liquid_friction_loss_psi, mph_to_fts, static_head_psi  # noqa: E402
from purge_sim.engine.pig_solver import PigSolverConfig, solve_pig_speed      # noqa: E402

CFG = PigSolverConfig(od_in=6.625, wt_in=0.156, roughness_ft=0.00015, sg=0.9, viscosity_cst=15.0,
                      min_speed_mph=0.5, max_speed_mph=5.0, target_speed_mph=3.0, temperature_f=45.0)
EXIT_MP, EXIT_PSIG, FLOOR = 20.0, 50.0, 25.0
# flat at 1000 ft, a 500 ft crest at MP 16, the exit back down at 1000 ft
T_MP = np.array([0.0, 15.0, 16.0, 17.0, 20.0])
T_EL = np.array([1000.0, 1000.0, 1500.0, 1000.0, 1000.0])


def _elev(mp):
    return float(np.interp(mp, T_MP, T_EL))


def _solve(drive, exit_max=None):
    return solve_pig_speed(CFG, 0.0, 1000.0, EXIT_PSIG, EXIT_MP, "Tankage inlet", drive, _elev,
                           terrain_mp=T_MP, terrain_elev=T_EL, slack_margin_psi=FLOOR,
                           exit_max_psig=exit_max)


def _fric(v_mph, miles):
    D = (CFG.od_in - 2 * CFG.wt_in) / 12.0
    return liquid_friction_loss_psi(miles * 5280.0, D, mph_to_fts(v_mph), CFG.sg, CFG.viscosity_cst,
                                    CFG.roughness_ft)


def test_exit_holds_back_column_behind_a_crest():
    r = _solve(600.0, exit_max=750.0)
    assert 0.5 < r.pig_speed_mph < 5.0
    # the crest sits at the slack floor
    crest = 600.0 - static_head_psi(500.0, CFG.sg) - _fric(r.pig_speed_mph, 16.0)
    assert crest == pytest.approx(FLOOR, abs=1.0)
    # and the exit pressure is what is left at the exit, well above the 50 psig minimum
    at_exit = 600.0 - _fric(r.pig_speed_mph, EXIT_MP)
    assert r.exit_psig == pytest.approx(at_exit, abs=1.0)
    assert r.exit_psig > EXIT_PSIG + 50.0
    assert r.peak_hold_psi == pytest.approx(r.exit_psig - EXIT_PSIG, abs=0.01)
    assert r.endpoint_added_psi == 0.0 and r.peak_hold_short_psi == 0.0


def test_fixed_exit_cannot_hold_the_crest():
    r = _solve(600.0, exit_max=EXIT_PSIG)
    assert r.exit_psig == pytest.approx(EXIT_PSIG)
    assert r.peak_hold_psi == 0.0 and r.peak_hold_short_psi > 50.0


def test_no_hold_when_the_exit_sets_the_drive():
    flat = np.array([1000.0] * len(T_MP))
    r = solve_pig_speed(CFG, 0.0, 1000.0, EXIT_PSIG, EXIT_MP, "Tankage inlet", 400.0, lambda m: 1000.0,
                        terrain_mp=T_MP, terrain_elev=flat, slack_margin_psi=FLOOR)
    assert r.peak_hold_psi == 0.0 and r.exit_psig == pytest.approx(EXIT_PSIG)
