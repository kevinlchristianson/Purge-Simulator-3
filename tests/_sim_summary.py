"""
Shared helpers for the regression baseline: what counts as a "real" scenario
file under scenarios/, how to run one and summarize it, and where the frozen
baseline lives. Used by both capture_baseline.py (writes the baseline) and
test_regression.py (checks against it) so the two can never silently drift
out of sync with each other.
"""
from __future__ import annotations

import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SCENARIOS_DIR = os.path.join(REPO_ROOT, "scenarios")
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")


def discover_scenarios() -> list[str]:
    """Repo-relative paths ('scenarios/...') of every real scenario JSON under
    scenarios/, skipping loose non-scenario data files (raw elevation arrays,
    route metadata) that happen to live alongside them."""
    paths = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "**", "*.json"), recursive=True))
    out = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "inputs" in raw:
            out.append(os.path.relpath(path, REPO_ROOT).replace("\\", "/"))
    return out


def run_scenario(rel_path: str) -> dict:
    """Load + simulate one scenario (repo-relative path) and summarize the result."""
    from purge_sim.data.scenario import load_scenario
    from purge_sim.engine.simulator import simulate
    from run_headless import build_sim_config

    scenario = load_scenario(os.path.join(REPO_ROOT, rel_path))
    cfg = build_sim_config(scenario.inputs)
    return summarize(simulate(cfg))


def summarize(results) -> dict:
    """Reduce a full SimResults down to the scalars worth regression-checking."""
    steps = results.steps
    faces = [s.pig_face_psig for s in steps] if steps else [0.0]
    speeds = [s.pig_speed_mph for s in steps] if steps else [0.0]
    return {
        "completed": results.completed,
        "abort_reason": results.abort_reason,
        "n_steps": len(steps),
        "total_scf_injected": round(results.total_scf_injected, 1),
        "total_scf_booster_fresh": round(results.total_scf_booster_fresh, 1),
        "total_scf_vented": round(results.total_scf_vented, 1),
        "max_face_psig": round(max(faces), 2),
        "avg_speed_mph": round(sum(speeds) / len(speeds), 4),
        "min_speed_mph": round(min(speeds), 4),
        "final_mp": round(steps[-1].pig_mp, 4) if steps else None,
        "mop_violations": sum(s.mop_violations for s in steps),
    }


def load_baseline() -> dict:
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
