"""
Capture (or re-capture) the regression baseline: run every tracked scenario and
record its key summary scalars to tests/baseline.json.

This is NOT the regression test itself (see test_regression.py) — it is the tool
that creates/updates the frozen expectations that test checks against. Re-run it
deliberately when a change is meant to alter sim behavior; never run it just to
make a failing test pass without understanding why the numbers moved.

Usage:
    python tests/capture_baseline.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purge_sim.data.scenario import load_scenario
from purge_sim.engine.simulator import simulate
from run_headless import build_sim_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS_DIR = os.path.join(REPO_ROOT, "scenarios")
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")


def summarize(results) -> dict:
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


def main():
    scenario_paths = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "**", "*.json"), recursive=True))
    baseline = {}
    skipped = []
    for path in scenario_paths:
        rel = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict) or "inputs" not in raw:
            skipped.append(rel)
            continue
        print(f"Running {rel} ...", flush=True)
        scenario = load_scenario(path)
        cfg = build_sim_config(scenario.inputs)
        results = simulate(cfg)
        summary = summarize(results)
        baseline[rel] = summary
        print(f"  -> completed={summary['completed']} "
              f"n2={summary['total_scf_injected']:,.0f} "
              f"vented={summary['total_scf_vented']:,.0f} "
              f"steps={summary['n_steps']}")

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
    print(f"\nWrote baseline for {len(baseline)} scenarios -> {BASELINE_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)} non-scenario json files under scenarios/:")
        for rel in skipped:
            print(f"  {rel}")


if __name__ == "__main__":
    main()
