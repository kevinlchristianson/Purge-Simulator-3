"""
Headless purge simulation runner — no GUI required.

Usage:
    python run_headless.py <scenario.json> [--log <output.txt>]

The scenario JSON is created by File > Save Scenario in the GUI. It contains
everything: elevation profile, pipe geometry, infrastructure, fluid properties.

The log is written to <scenario_basename>_log.txt by default.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from purge_sim.data.scenario import load_scenario
from purge_sim.engine.simulator import simulate
from purge_sim.engine.config_builder import build_sim_config
from purge_sim.engine.log_export import export_run_log
from purge_sim.engine.purge_report import export_purge_report


def main():
    parser = argparse.ArgumentParser(description='Headless purge simulation runner')
    parser.add_argument('scenario', help='Path to scenario JSON file')
    parser.add_argument('--log', default=None, help='Output log file (default: <scenario>_log.txt)')
    args = parser.parse_args()

    print(f"Loading: {args.scenario}")
    scenario = load_scenario(args.scenario)
    print(f"Scenario: {scenario.meta.name}")

    cfg = build_sim_config(scenario.inputs)
    span = cfg.purge_end_mp - cfg.purge_start_mp

    log_path = args.log or (os.path.splitext(args.scenario)[0] + '_log.txt')

    _last_pct = [-1]
    def _progress(f):
        pct = int(f * 100)
        if pct != _last_pct[0] and pct % 5 == 0:
            mp = cfg.purge_start_mp + f * span
            print(f"  {pct:3d}%  MP {mp:.1f}", flush=True)
            _last_pct[0] = pct

    print("Simulating...")
    results = simulate(cfg, progress_cb=_progress)

    if results.roadmap is not None:
        print("\n" + results.roadmap.report())

    if results.booster_plan is not None:
        bp = results.booster_plan
        print("\nBooster siting plan:")
        for note in bp.notes:
            print(f"  {note}")
        for mp in bp.sites_mp:
            print(f"    MP {mp:.1f}: {bp.reasons.get(mp, '')}")

    status = 'COMPLETED' if results.completed else f'ABORTED: {results.abort_reason}'
    print(f"\nStatus:    {status}")
    print(f"Steps:     {len(results.steps):,}")
    print(f"N2 total:  {results.total_scf_injected:,.0f} SCF")
    print(f"N2 vented: {results.total_scf_vented:,.0f} SCF  (MOP relief)")
    print(f"Wall time: {results.wall_time_s:.1f}s")

    if results.steps:
        faces  = [s.pig_face_psig  for s in results.steps]
        speeds = [s.pig_speed_mph  for s in results.steps]
        viols  = sum(s.mop_violations for s in results.steps)
        last   = results.steps[-1]
        print(f"Avg face:  {sum(faces)/len(faces):.1f} psig")
        print(f"Max face:  {max(faces):.1f} psig")
        print(f"Avg speed: {sum(speeds)/len(speeds):.3f} mph")
        print(f"Min speed: {min(speeds):.3f} mph")
        print(f"MOP viol:  {viols:,}")
        print(f"Final MP:  {last.pig_mp:.3f}  ({100*(last.pig_mp - cfg.purge_start_mp)/span:.1f}% of route)")

    print(f"\nExporting log -> {log_path}")
    export_run_log(results, log_path, scenario_name=scenario.meta.name)

    # xlsx deliverable — generated alongside the log file
    _base = os.path.splitext(log_path)[0]
    xlsx_path = (_base[:-4] if _base.endswith('_log') else _base) + '_report.xlsx'
    try:
        pi = {
            "project": scenario.meta.name,
            "notes":   scenario.meta.notes,
            "date":    scenario.meta.modified_at[:10],
        }
        export_purge_report(results, xlsx_path, scenario_name=scenario.meta.name,
                            project_info=pi)
        print(f"Exported xlsx  -> {xlsx_path}")
    except ImportError as e:
        print(f"xlsx export skipped (missing library): {e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"xlsx export failed: {e}")

    # Formatted (client-facing) report — Purge Report + embedded pressure/run charts
    fmt_path = (_base[:-4] if _base.endswith('_log') else _base) + '_formatted.xlsx'
    try:
        from purge_sim.engine.formatted_report import export_formatted_report
        pi2 = {
            "project": scenario.meta.name,
            "notes":   scenario.meta.notes,
            "date":    scenario.meta.modified_at[:10],
        }
        export_formatted_report(results, fmt_path,
                                scenario_name=scenario.meta.name, project_info=pi2)
        print(f"Exported fmt    -> {fmt_path}")
    except ImportError as e:
        print(f"Formatted report skipped (missing library): {e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Formatted report failed: {e}")

    # Animation GIF + MP4 — always a deliverable
    _anim_base = _base[:-4] if _base.endswith('_log') else _base
    gif_path = _anim_base + '_profile.gif'
    mp4_path = _anim_base + '_profile.mp4'
    try:
        from purge_sim.engine.animation import generate_animation
        span = cfg.purge_end_mp - cfg.purge_start_mp
        n_frames = max(1, int(span / 3.0))
        print(f"Generating animation (~{n_frames} frames) ...")
        _last_anim = [-1]
        def _anim_cb(done, total):
            pct = int(done / total * 100)
            if pct != _last_anim[0] and pct % 25 == 0:
                print(f"  anim {pct}%", flush=True)
                _last_anim[0] = pct
        generate_animation(results, cfg,
                           out_gif=gif_path, out_mp4=mp4_path,
                           progress_cb=_anim_cb,
                           scenario_name=scenario.meta.name)
        print(f"Exported GIF    -> {gif_path}")
        if os.path.exists(mp4_path):
            print(f"Exported MP4    -> {mp4_path}")
    except ImportError as e:
        print(f"Animation skipped (missing library): {e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Animation failed: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
