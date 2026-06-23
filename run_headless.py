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
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from purge_sim.data.scenario import load_scenario, ScenarioInputs
from purge_sim.engine.simulator import SimConfig, simulate
from purge_sim.engine.segment_model import PipeGeometry
from purge_sim.engine.check_valve import CheckValve
from purge_sim.engine.booster import BoosterConfig
from purge_sim.engine.pump_stations import PumpStationConfig
from purge_sim.engine.bpcv import BPCVConfig, BCPVDownstreamJoint
from purge_sim.engine.physics import fts_to_bph, mph_to_fts, pipe_area_ft2
from purge_sim.engine.mop_check import MOPJoint
from purge_sim.engine.log_export import export_run_log


def build_sim_config(inp: ScenarioInputs) -> SimConfig:
    geom = PipeGeometry()
    for s in inp.pipe_segments:
        geom.segments.append((s['start_mp'], s['end_mp'], s['od_in'], s['wt_in']))

    elev = np.array(inp.elevation_profile, dtype=float)

    cvs = [CheckValve(mp=cv['mp'], name=cv['name']) for cv in inp.check_valves]

    pss = [
        PumpStationConfig(mp=ps['mp'], name=ps['name'], suction_psig=ps['suction_psig'])
        for ps in inp.pump_stations
    ]

    bss = [
        BoosterConfig(
            mp=b['mp'],
            name=b['name'],
            discharge_psig=inp.spread_discharge_psig,
            suction_min_psig=inp.spread_suction_min_psig,
            max_flow_scfm=inp.spread_max_flow_scfm,
        )
        for b in inp.booster_stations
    ]

    bpcv = None
    if inp.bpcv:
        b = inp.bpcv
        bpcv_mp = b['mp']
        ds_joints = [
            BCPVDownstreamJoint(mp=j['mp'], mop_psig=j['mop_psig'], elevation_ft=j['elevation_ft'])
            for j in inp.mop_joints if j['mp'] > bpcv_mp
        ]
        ds_od, ds_wt = geom.od_wt_at(bpcv_mp + 0.1)
        ds_area = pipe_area_ft2(ds_od, ds_wt)
        bpcv = BPCVConfig(
            mp=bpcv_mp,
            elevation_ft=b['elevation_ft'],
            name=b.get('name', 'BPCV'),
            downstream_joints=ds_joints,
            downstream_flow_bph=fts_to_bph(mph_to_fts(inp.target_speed_mph), ds_area),
            downstream_od_in=ds_od,
            downstream_wt_in=ds_wt,
            downstream_sg=inp.fluid_sg,
            downstream_viscosity_cst=inp.fluid_viscosity_cst,
            downstream_roughness_ft=inp.fluid_roughness_ft,
        )

    return SimConfig(
        pipe_geometry=geom,
        elevation_profile=elev,
        purge_start_mp=inp.purge_start_mp,
        purge_end_mp=inp.purge_end_mp,
        fluid_sg=inp.fluid_sg,
        fluid_viscosity_cst=inp.fluid_viscosity_cst,
        fluid_roughness_ft=inp.fluid_roughness_ft,
        n2_temperature_f=inp.n2_temperature_f,
        max_injection_psig=inp.max_injection_psig or math.inf,
        max_injection_scfm=inp.max_injection_scfm,
        exit_pressure_run_psig=inp.exit_pressure_run_psig,
        exit_pressure_end_psig=inp.exit_pressure_end_psig,
        exit_pressure_behavior=inp.exit_pressure_behavior,
        throttle_down_miles=inp.throttle_down_miles,
        min_speed_mph=inp.min_speed_mph,
        max_speed_mph=inp.max_speed_mph,
        target_speed_mph=inp.target_speed_mph,
        maop_psig=inp.maop_psig or math.inf,
        max_drive_psig=inp.max_drive_psig or math.inf,
        mop_joints=[
            MOPJoint(mp=j['mp'], mop_psig=j['mop_psig'],
                     elevation_ft=j['elevation_ft'],
                     od_in=j.get('od_in', 24.0), wt_in=j.get('wt_in', 0.313))
            for j in inp.mop_joints
        ],
        mop_warning_fraction=getattr(inp, 'mop_warning_fraction', 0.95),
        check_valves=cvs,
        booster_configs=bss,
        pump_stations=pss,
        bpcv=bpcv,
        n_spreads=inp.n_spreads,
        mob_time_hr=inp.mob_time_hr,
        dt_hr=inp.dt_hr,
        adaptive_dt=inp.adaptive_dt,
    )


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

    status = 'COMPLETED' if results.completed else f'ABORTED: {results.abort_reason}'
    print(f"\nStatus:    {status}")
    print(f"Steps:     {len(results.steps):,}")
    print(f"N2 total:  {results.total_scf_injected:,.0f} SCF")
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
    print("Done.")


if __name__ == '__main__':
    main()
