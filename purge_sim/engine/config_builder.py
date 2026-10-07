"""
ScenarioInputs -> SimConfig.

The one place a saved scenario turns into an engine configuration. Shared by the
headless runner, the regression tests and the desktop app, so every entry point
simulates a scenario exactly the same way.
"""

import math

import numpy as np

from ..data.scenario import ScenarioInputs
from .simulator import SimConfig
from .segment_model import PipeGeometry
from .check_valve import CheckValve
from .booster import BoosterConfig
from .pump_stations import PumpStationConfig
from .bpcv import BPCVConfig, BCPVDownstreamJoint
from .physics import fts_to_bph, mph_to_fts, pipe_area_ft2
from .mop_check import MOPJoint


def build_sim_config(inp: ScenarioInputs) -> SimConfig:
    geom = PipeGeometry()
    for s in inp.pipe_segments:
        geom.segments.append((s['start_mp'], s['end_mp'], s['od_in'], s['wt_in']))

    elev = np.array(inp.elevation_profile, dtype=float)

    cvs = [CheckValve(mp=cv['mp'], name=cv['name']) for cv in inp.check_valves]

    # A pump station past the pig stop is not part of this purge: the pig never reaches it
    # and the liquid beyond the stop is carried by the exit pressure (see
    # precheck.exit_pressure_at_stop). Left in, the engine would push the pig toward that
    # station's suction instead of the exit at the stop (GL-09: Lavina at MP 239.5 vs the
    # stop at MP 220, 30 psig instead of 583).
    pss = [
        PumpStationConfig(mp=ps['mp'], name=ps['name'], suction_psig=ps['suction_psig'])
        for ps in inp.pump_stations
        if ps['mp'] < inp.purge_end_mp
    ]

    # Per-booster overrides (max_flow_scfm / discharge_psig / suction_min_psig) fall back
    # to the global spread_* params. Lets one booster (e.g. LS feeding the Sutton climb) run
    # more aggressively than the rest without changing the whole spread.
    bss = [
        BoosterConfig(
            mp=b['mp'],
            name=b['name'],
            discharge_psig=b.get('discharge_psig', inp.spread_discharge_psig),
            suction_min_psig=b.get('suction_min_psig', inp.spread_suction_min_psig),
            max_flow_scfm=b.get('max_flow_scfm', inp.spread_max_flow_scfm),
            run_at_max=b.get('run_at_max', False),
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
        deployed_booster_mps=inp.deployed_booster_mps,
        n2_budget_scf=inp.n2_budget_scf,
        drive_mop_fraction=inp.drive_mop_fraction,
        smooth_injection=getattr(inp, 'smooth_injection', True),
        injection_ramp_scfm_per_hr=getattr(inp, 'injection_ramp_scfm_per_hr', None),
        drive_setpoint_slew_psi_per_hr=getattr(inp, 'drive_setpoint_slew_psi_per_hr', None),
        drive_ceiling_fraction=inp.drive_ceiling_fraction,
        bpcv=bpcv,
        n_spreads=inp.n_spreads,
        mob_time_hr=inp.mob_time_hr,
        dt_hr=inp.dt_hr,
        adaptive_dt=inp.adaptive_dt,
    )
