"""
Regression suite: every tracked scenario must still produce the same summary
behavior (N2 totals, completion, speeds, MOP violations) recorded in
tests/baseline.json by tests/capture_baseline.py.

A failure here means a change altered simulated behavior for a real,
previously-run client job. That's not automatically wrong — a genuine
engine fix is expected to move some numbers — but it must be a DELIBERATE,
understood change: re-run capture_baseline.py, review exactly what moved and
why, then commit the updated baseline alongside the change. Never re-capture
just to silence a failure without reading the diff first.

Floats get a small tolerance to absorb harmless adaptive-timestep noise;
completion status, abort reason, and MOP violation count must match exactly.
"""
import pytest

from _sim_summary import discover_scenarios, run_scenario, load_baseline

SCENARIOS = discover_scenarios()
BASELINE = load_baseline()

# Scalars allowed a small relative tolerance (float/timestep noise).
_REL_TOL = 0.01  # 1%
_FLOAT_KEYS = (
    "total_scf_injected", "total_scf_booster_fresh", "total_scf_vented",
    "max_face_psig", "avg_speed_mph", "min_speed_mph", "final_mp",
)
# Exact-match keys: any change here is a real behavior change, not noise.
_EXACT_KEYS = ("completed", "abort_reason", "mop_violations")


def test_no_stale_baseline_entries():
    """Catches scenario files that were deleted/renamed without updating the baseline."""
    stale = set(BASELINE) - set(SCENARIOS)
    assert not stale, f"baseline.json has entries for scenarios that no longer exist: {stale}"


@pytest.mark.parametrize("rel_path", SCENARIOS, ids=SCENARIOS)
def test_scenario_matches_baseline(rel_path):
    assert rel_path in BASELINE, (
        f"{rel_path} has no baseline entry — run `python tests/capture_baseline.py` "
        f"and commit the updated tests/baseline.json"
    )
    expected = BASELINE[rel_path]
    actual = run_scenario(rel_path)

    for key in _EXACT_KEYS:
        assert actual[key] == expected[key], (
            f"{rel_path}: {key} changed ({expected[key]!r} -> {actual[key]!r})"
        )

    for key in _FLOAT_KEYS:
        exp_v, act_v = expected[key], actual[key]
        if exp_v is None or act_v is None:
            assert exp_v == act_v, f"{rel_path}: {key} changed ({exp_v!r} -> {act_v!r})"
            continue
        assert act_v == pytest.approx(exp_v, rel=_REL_TOL, abs=1.0), (
            f"{rel_path}: {key} changed ({exp_v} -> {act_v})"
        )
