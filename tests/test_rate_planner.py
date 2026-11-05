"""The steady-injection planner: a fixed rate schedule the engine pumps exactly, and the
planner that finds the one with the fewest rate changes that still holds every rule."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purge_sim.data.scenario import load_scenario                          # noqa: E402
from purge_sim.engine.config_builder import build_sim_config              # noqa: E402
from purge_sim.engine.simulator import simulate, schedule_average_scfm    # noqa: E402
from purge_sim.engine import rate_planner as rp                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THUNDERBIRD = os.path.join(ROOT, "scenarios", "Bridger_THUNDERBIRD", "thunderbird_packcoast.json")


def test_schedule_average_over_a_step():
    sched = [(2.0, 6000.0), (3.0, 5000.0)]
    assert schedule_average_scfm(sched, 0.0, 1.0) == 6000.0
    assert schedule_average_scfm(sched, 1.5, 2.5) == 5500.0     # half in each piece
    assert schedule_average_scfm(sched, 4.5, 5.5) == 2500.0     # runs out half way
    assert schedule_average_scfm(sched, 6.0, 7.0) == 0.0


def test_fit_reproduces_a_two_rate_curve():
    t = np.linspace(0.0, 8.0, 801)
    cum = np.where(t <= 3.0, 6000.0 * 60 * t, 6000.0 * 60 * 3.0 + 4000.0 * 60 * (t - 3.0))
    one = rp.fit_schedule(t, cum, 8.0, 1)
    assert len(one) == 1 and abs(one[0][1] - cum[-1] / 8.0 / 60.0) < 1.0
    two = rp.fit_schedule(t, cum, 8.0, 2)
    assert [round(h, 2) for h, _ in two] == [3.0, 5.0]
    assert [round(r) for _, r in two] == [6000, 4000]


def test_round_schedule_keeps_the_total():
    sched = rp.round_schedule([(3.04, 5983.0), (5.01, 4012.0)], 20000.0, total_scf=2.28e6)
    assert sched[0] == (3.0, 6000.0)
    assert sched[1][1] == 4000.0
    assert abs(rp.schedule_scf(sched) - 2.28e6) < 4000.0 * 60 * 0.1 + 1
    assert rp.schedule_text(sched) == "6,000 SCFM for 3.0 h, then 4,000 SCFM for 5.0 h, then stop"


def test_engine_pumps_the_schedule_exactly():
    sc = load_scenario(THUNDERBIRD)
    cfg = build_sim_config(sc.inputs)
    cfg.injection_schedule = [(4.0, 6000.0), (5.0, 5000.0)]
    res = simulate(cfg)
    assert res.completed
    assert abs(res.total_scf_injected - (4.0 * 6000 + 5.0 * 5000) * 60) < 2000
    on = [s for s in res.steps if s.t_hr < 3.9]
    assert all(abs(s.injection_scfm - 6000.0) < 1 for s in on)
    assert all(abs(s.scheduled_scfm - 6000.0) < 1e-6 for s in on)
    assert all(s.injection_scfm == 0.0 for s in res.steps if s.t_hr > 9.1)


def test_planner_finds_one_flat_rate_for_thunderbird():
    sc = load_scenario(THUNDERBIRD)
    cfg = build_sim_config(sc.inputs)
    plan = rp.plan_steady_injection(cfg)
    assert plan.best is not None and len(plan.best.schedule) == 1
    assert plan.best.passed
    assert abs(plan.best.n2_scf - plan.reference["n2_scf"]) < 0.02 * plan.reference["n2_scf"]
    assert plan.best.summary["duration_hr"] <= plan.reference["duration_hr"] * 1.1 + 0.25
    assert "then stop" in plan.text and "One rate the whole time" in plan.text
    view = rp.plan_to_dict(plan)
    assert view["schedule"] == [[h, r] for h, r in plan.best.schedule]
    assert view["best"]["passed"] and all(r["ok"] for r in view["best"]["rules"])
