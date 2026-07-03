"""
Animated pipeline profile GIF / MP4 generator — CLI wrapper.

Usage:
    python make_profile_gif.py [--scenario <path.json>]
                               [--out-dir <directory>]
                               [--mp-step 3.0]
                               [--no-mp4]
                               [--no-cache]

Without --scenario, defaults to the SP->MT pack-and-coast 58M scenario.
Delete the .pkl cache file to force a fresh simulation.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from purge_sim.data.scenario import load_scenario
from purge_sim.engine import simulator as S
from purge_sim.engine.animation import generate_animation
from run_headless import build_sim_config


_DEFAULT_SCENARIO = "C:/Users/kevin/PurgeSimScenarios/SP to MT_pack-and-coast 58M.json"


def _load_or_run(scenario_path: str, cache_path):
    """Load a cached (cfg, results) pair or run the simulation fresh."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            print(f"Loaded cached sim from {cache_path}")
            return pickle.load(fh)

    scenario = load_scenario(scenario_path)
    cfg = build_sim_config(scenario.inputs)
    print(f"Simulating: {scenario.meta.name} ...")
    results = S.simulate(cfg)
    print(f"  -> {len(results.steps)} steps  "
          f"N2 {results.total_scf_n2 / 1e6:.2f} MMscf  "
          f"wall {results.wall_time_s:.1f}s")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump((cfg, results), fh)
        print(f"Cached sim -> {cache_path}")

    return cfg, results


def main():
    parser = argparse.ArgumentParser(
        description="Generate animated pipeline profile GIF / MP4"
    )
    parser.add_argument(
        "--scenario", default=_DEFAULT_SCENARIO,
        help="Path to scenario JSON  (default: SP->MT pack-and-coast 58M)"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory  (default: same folder as the scenario JSON)"
    )
    parser.add_argument(
        "--mp-step", type=float, default=3.0,
        help="Miles between animation frames  (default: 3.0)"
    )
    parser.add_argument(
        "--no-mp4", action="store_true",
        help="Skip MP4 -- produce GIF only"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force a fresh simulation even if a cache file exists"
    )
    args = parser.parse_args()

    if not os.path.exists(args.scenario):
        print(f"ERROR: scenario not found: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    base    = os.path.splitext(os.path.basename(args.scenario))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.scenario))
    os.makedirs(out_dir, exist_ok=True)

    cache_path = (
        None if args.no_cache
        else os.path.join(out_dir, f".{base}_anim_cache.pkl")
    )
    out_gif = os.path.join(out_dir, f"{base}_profile.gif")
    out_mp4 = None if args.no_mp4 else os.path.join(out_dir, f"{base}_profile.mp4")

    cfg, results = _load_or_run(args.scenario, cache_path)

    span = cfg.purge_end_mp - cfg.purge_start_mp
    approx_frames = max(1, int(span / args.mp_step))
    print(f"Building animation: ~{approx_frames} frames "
          f"({cfg.purge_start_mp:.1f}->{cfg.purge_end_mp:.1f} mi, "
          f"step {args.mp_step} mi) ...")

    def _progress(done, total):
        if done % max(1, total // 10) == 0 or done == total:
            print(f"  frame {done}/{total}", flush=True)

    generate_animation(
        results, cfg,
        out_gif=out_gif,
        out_mp4=out_mp4,
        mp_step=args.mp_step,
        progress_cb=_progress,
    )

    print(f"Saved GIF -> {out_gif}")
    if out_mp4 and os.path.exists(out_mp4):
        print(f"Saved MP4 -> {out_mp4}")
    print("Done.")


if __name__ == "__main__":
    main()
