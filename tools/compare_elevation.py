"""
compare_elevation.py — Check the built-in USGS 3DEP lookup against a scenario's stored profile.

Looks the KMZ's route up the way the app does (purge_sim/data/elevation.py), lines it up
with the scenario's elevation profile, reports the elevation differences, then runs the
scenario twice (stored profile, new profile) and compares the results.

Usage
-----
    python tools/compare_elevation.py route.kmz scenarios/Job/job.json
    python tools/compare_elevation.py route.kmz scenarios/Job/job.json --reverse --offset-mi 13.489

Without --reverse/--offset-mi, both route directions and every offset are tried and the
best-matching alignment is used (the scenario may cover only part of the route, and may
run the opposite way to the KMZ). Needs internet for the 3DEP lookup.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purge_sim.data.elevation import DEFAULT_SPACING_FT, fetch_route_elevation, valid_coords  # noqa: E402
from purge_sim.data.profile_parser import parse_profile  # noqa: E402
from purge_sim.data.scenario import load_scenario  # noqa: E402


def _align(new_mp, new_el, old_mp, old_el, offsets):
    best = None
    for off in offsets:
        d = np.interp(old_mp + off, new_mp, new_el) - old_el
        rms = float(np.sqrt(np.mean(d ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(off))
    return best


def _run(inputs):
    from purge_sim.engine.config_builder import build_sim_config
    from purge_sim.engine.simulator import simulate
    res = simulate(build_sim_config(inputs))
    steps = res.steps
    return {
        "completed": res.completed,
        "n2_total_scf": float(res.total_scf_n2),
        "max_face_psig": float(max(s.pig_face_psig for s in steps)),
        "duration_hr": float(steps[-1].t_hr),
        "slack_steps": sum(bool(s.slack_line_risk) for s in steps),
        "mop_violation_steps": sum(1 for s in steps if s.mop_violations),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route")
    ap.add_argument("scenario")
    ap.add_argument("--spacing", type=float, default=DEFAULT_SPACING_FT)
    ap.add_argument("--reverse", action="store_true", help="scenario MPs run opposite to the KMZ")
    ap.add_argument("--offset-mi", type=float, default=None, help="route MP of the scenario's MP 0")
    ap.add_argument("--no-sim", action="store_true", help="skip the two simulator runs")
    args = ap.parse_args()

    prof = parse_profile(args.route)
    ok = valid_coords(prof.lat, prof.lon)
    sc = load_scenario(args.scenario)
    old = np.asarray(sc.inputs.elevation_profile, dtype=float)
    old_mp, old_el = old[:, 0] - old[0, 0], old[:, 1]
    span = old_mp[-1]

    results = {}
    for rev in ([args.reverse] if (args.reverse or args.offset_mi is not None) else [False, True]):
        lat, lon = prof.lat[ok], prof.lon[ok]
        if rev:
            lat, lon = lat[::-1], lon[::-1]
        print(f"Looking up {'reversed ' if rev else ''}route in USGS 3DEP...", flush=True)
        r = fetch_route_elevation(lat, lon, spacing_ft=args.spacing,
                                  progress_cb=lambda f, m: print(f"\r  {m} ({f*100:.0f}%)   ", end="", flush=True))
        print()
        offs = [args.offset_mi] if args.offset_mi is not None else np.arange(0.0, max(r.length_mi - span, 0) + 1e-9, 0.005)
        results[rev] = (r, _align(r.mileposts, r.elevations_ft, old_mp, old_el, offs))
    rev = min(results, key=lambda k: results[k][1][0])
    r, (rms, off) = results[rev]
    new_el_at_old = np.interp(old_mp + off, r.mileposts, r.elevations_ft)
    d = new_el_at_old - old_el

    print(f"\nRoute {r.length_mi:.3f} mi, {len(r.mileposts)} points at {args.spacing:g} ft "
          f"({r.filled_points} interpolated). Scenario span {span:.3f} mi, {len(old_mp)} stored points.")
    print(f"Alignment: {'reversed' if rev else 'as drawn'}, scenario MP 0 = route MP {off:.3f}")
    print(f"Elevation difference new - stored at stored points: mean {d.mean():+.1f} ft, "
          f"RMS {rms:.1f} ft, max |{np.abs(d).max():.1f}| ft at scenario MP {old_mp[np.abs(d).argmax()]:.3f}")
    seg = (r.mileposts >= off) & (r.mileposts <= off + span)
    print(f"Stored high/low {old_el.max():.0f}/{old_el.min():.0f} ft; new high/low over the span "
          f"{r.elevations_ft[seg].max():.0f}/{r.elevations_ft[seg].min():.0f} ft")
    for f in r.flags:
        print("  FLAG:", f)

    if not args.no_sim:
        new_inputs = copy.deepcopy(sc.inputs)
        mp = r.mileposts[seg] - off + old[0, 0]
        new_inputs.elevation_profile = [[float(m), float(e)] for m, e in zip(mp, r.elevations_ft[seg])]
        print("\nRunning the scenario with the stored and the new profile...", flush=True)
        a, b = _run(sc.inputs), _run(new_inputs)
        print(f"{'':22}{'stored':>14}{'new 3DEP':>14}{'change':>10}")
        def fmt(v):
            return f"{v:,.1f}" if isinstance(v, float) else str(v)
        for k in a:
            va, vb = a[k], b[k]
            ch = f"{(vb - va) / va * 100:+.2f}%" if isinstance(va, float) and va else ""
            print(f"{k:22}{fmt(va):>14}{fmt(vb):>14}{ch:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
