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
import json

from _sim_summary import discover_scenarios, run_scenario, BASELINE_PATH


def main():
    baseline = {}
    for rel in discover_scenarios():
        print(f"Running {rel} ...", flush=True)
        summary = run_scenario(rel)
        baseline[rel] = summary
        print(f"  -> completed={summary['completed']} "
              f"n2={summary['total_scf_injected']:,.0f} "
              f"vented={summary['total_scf_vented']:,.0f} "
              f"steps={summary['n_steps']}")

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
    print(f"\nWrote baseline for {len(baseline)} scenarios -> {BASELINE_PATH}")


if __name__ == "__main__":
    main()
