"""Lean (smooth-injection) launch: the N2 pump is at rate when the pig is released, so the pig
starts at target speed instead of stalling while the pump ramps up from zero."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purge_sim.data.scenario import load_scenario                      # noqa: E402
from purge_sim.engine.config_builder import build_sim_config          # noqa: E402
from purge_sim.engine.simulator import simulate                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_lean_launch_starts_at_target_speed_without_slack_flags():
    sc = load_scenario(os.path.join(ROOT, "scenarios", "CHS_LaurelAllendale", "laurel_allendale_3mph.json"))
    cfg = build_sim_config(sc.inputs)
    assert cfg.smooth_injection and cfg.n2_budget_scf is None
    steps = simulate(cfg).steps
    launch = [s for s in steps if s.pig_mp - cfg.purge_start_mp < 0.25]
    assert min(s.pig_speed_mph for s in launch) >= cfg.min_speed_mph
    assert not any(s.slack_line_risk for s in launch)
