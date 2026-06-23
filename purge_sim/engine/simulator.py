"""
Main simulation loop: simulate(config) -> SimResults

The simulator advances pig position in time steps, updating:
  1. Pig position (from pig_solver)
  2. N2 segment inventories (injection + booster transfers + pig advance)
  3. Check valve state machine (cascade merges)
  4. Booster activations and SCF transfers
  5. Pump station shutdown evaluations
  6. BPCV state
  7. Per-joint MOP check
  8. Injection SCFM computation (minimum to maintain booster suction floors)

All time in hours, mileposts in miles, pressures in psig, SCF in standard cubic feet.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

import numpy as np
from scipy.interpolate import interp1d

from .constants import ATM_PSI, N2_TEMP_F_DEFAULT
from .physics import (
    pipe_area_ft2, pipe_volume_ft3,
    scfm_from_pig_velocity, psig_to_psia, psia_to_psig,
    static_head_psi, liquid_friction_loss_psi, gas_friction_loss_psi,
    target_exit_pressure, fts_to_bph, bph_to_fts, mph_to_fts,
)
from .segment_model import N2Segment, SegmentList, PipeGeometry, scf_from_pressure_volume
from .check_valve import CheckValve, evaluate_check_valves, insert_check_valve_boundary
from .booster import (
    BoosterConfig, BoosterState, BoosterSpread,
    step_booster, active_booster_mps, activate_booster_if_pig_passed,
)
from .bpcv import BPCVConfig, BPCVState, step_bpcv
from .pump_stations import (
    PumpStationConfig, PumpStationState, StationStatus,
    evaluate_shutdowns, effective_exit_condition, next_active_station,
)
from .pig_solver import PigSolverConfig, PigSolverResult, solve_pig_speed, minimum_n2_floor_to_sustain_flow
from .mop_check import (
    MOPJoint, MOPCheckResult, MOPStatus,
    check_mop_liquid_side, check_mop_gas_side,
    build_gas_pressure_profile, mop_summary, thin_mop_joints,
)


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Complete simulation configuration — all inputs in one place."""

    # --- Pipeline geometry ---
    pipe_geometry: PipeGeometry           # variable OD/WT along route
    elevation_profile: np.ndarray         # shape (N, 2): [[mp, elev_ft], ...]
    purge_start_mp: float = 0.0
    purge_end_mp: float = 236.0

    # --- Fluid ---
    fluid_sg: float = 0.85
    fluid_viscosity_cst: float = 2.7
    fluid_roughness_ft: float = 0.00015   # default pipe roughness

    # --- N2 gas ---
    n2_temperature_f: float = N2_TEMP_F_DEFAULT
    n2_initial_pressure_psig: float = 50.0   # pressure at purge start

    # --- Exit / endpoint ---
    exit_pressure_run_psig: float = 50.0
    exit_pressure_end_psig: float = 10.0
    exit_pressure_behavior: str = 'taper_last_n_miles'
    throttle_down_miles: float = 5.0

    # --- Pig speed limits ---
    min_speed_mph: float = 0.5
    max_speed_mph: float = 6.0
    target_speed_mph: float = 3.0

    # --- MOP check ---
    mop_joints: List[MOPJoint] = field(default_factory=list)
    mop_warning_fraction: float = 0.95

    # --- Infrastructure ---
    check_valves: List[CheckValve] = field(default_factory=list)
    booster_configs: List[BoosterConfig] = field(default_factory=list)
    pump_stations: List[PumpStationConfig] = field(default_factory=list)
    bpcv: Optional[BPCVConfig] = None

    # --- Injection limits ---
    max_injection_psig: float = 200.0
    max_injection_scfm: float = 5000.0

    # --- MAOP / drive envelope ---
    maop_psig: float = 1200.0
    max_drive_psig: float = 1000.0        # operator-chosen max N2 drive behind pig

    # --- Spread logistics ---
    # n_spreads = 0 means unlimited (every booster activates when pig passes, v29 behavior)
    n_spreads: int = 0
    mob_time_hr: float = 18.0             # travel + rig-up time per spread move (hours)

    # --- Simulation control ---
    dt_hr: float = 1.0 / 60.0            # default 1-minute timesteps
    max_steps: int = 100_000
    adaptive_dt: bool = True
    dt_min_hr: float = 1.0 / 3600.0      # 1-second minimum step
    dt_max_hr: float = 5.0 / 60.0        # 5-minute maximum step
    max_stall_hr: float = 48.0           # abort if pig stalled this long with no progress


# ---------------------------------------------------------------------------
# Per-timestep record
# ---------------------------------------------------------------------------

@dataclass
class SimStep:
    t_hr: float
    pig_mp: float
    pig_elevation_ft: float
    pig_speed_mph: float

    # N2 gas column
    total_scf: float
    injection_scfm: float
    injection_psig: float
    pig_face_psig: float
    segments: List[dict]   # segment_model.summary() list

    # Pressure profile (mp, psig) pairs for N2 gas side
    gas_pressure_profile: List[tuple]

    # Exit condition
    exit_psig: float
    exit_mp: float
    exit_description: str

    # MOP checks
    mop_violations: int
    mop_warnings: int
    worst_mop_mp: Optional[float]
    worst_mop_margin: Optional[float]
    mop_results: List[MOPCheckResult] = field(default_factory=list)

    # Booster states
    booster_states: List[dict] = field(default_factory=list)

    # Pump station pressures (one entry per station per timestep)
    # Each dict: {mp, name, discharge_psig, suction_psig, status}
    #   status: 'bypassed' | 'running' | 'shutdown'
    #   bypassed  — pig has passed; discharge_psig is N2 gas pressure at that location
    #   running   — pump active ahead of pig; discharge_psig is liquid HGL discharge
    #   shutdown  — pump offline, pig hasn't passed; discharge_psig is passive liquid HGL
    station_pressures: List[dict] = field(default_factory=list)

    # Pump station events
    station_shutdown_events: List[dict] = field(default_factory=list)

    # Flags
    slack_line_risk: bool = False
    meter_valve_active: bool = False


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class SimResults:
    config: SimConfig
    steps: List[SimStep] = field(default_factory=list)
    completed: bool = False
    abort_reason: str = ""
    wall_time_s: float = 0.0
    total_scf_injected: float = 0.0
    spread_events: List[dict] = field(default_factory=list)
    # Each entry: {spread_id, from_mp, to_mp, reason, t_hr, arrive_at_hr}

    @property
    def times_hr(self) -> List[float]:
        return [s.t_hr for s in self.steps]

    @property
    def pig_mileposts(self) -> List[float]:
        return [s.pig_mp for s in self.steps]

    @property
    def pig_speeds_mph(self) -> List[float]:
        return [s.pig_speed_mph for s in self.steps]

    @property
    def total_scf_history(self) -> List[float]:
        return [s.total_scf for s in self.steps]

    @property
    def injection_scfm_history(self) -> List[float]:
        return [s.injection_scfm for s in self.steps]

    @property
    def pig_face_pressure_history(self) -> List[float]:
        return [s.pig_face_psig for s in self.steps]


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def compute_initial_n2_pressure(
    cfg: SimConfig,
    elevation_at: Callable[[float], float],
) -> float:
    """
    Auto-compute the N2 startup pressure needed to push the pig at target speed
    from the purge start to the first active exit point.

    First exit = first pump station (suction psig), then BPCV, then end-of-line.
    Returns a value clamped between (exit_psig + 5) and (MAOP × 0.9).
    """
    # First active exit point
    if cfg.pump_stations:
        first = min(cfg.pump_stations, key=lambda s: s.mp)
        exit_mp   = first.mp
        exit_psig = first.suction_psig
    elif cfg.bpcv:
        exit_mp   = cfg.bpcv.mp
        exit_psig = cfg.bpcv.set_point_psig
    else:
        exit_mp   = cfg.purge_end_mp
        exit_psig = cfg.exit_pressure_run_psig

    L_ft    = max(0.0, exit_mp - cfg.purge_start_mp) * 5280.0
    od, wt  = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    id_ft   = (od - 2.0 * wt) / 12.0
    v_fts   = mph_to_fts(cfg.target_speed_mph)

    friction_psi = liquid_friction_loss_psi(
        L_ft, id_ft, v_fts,
        cfg.fluid_sg, cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft,
    )
    head_psi = static_head_psi(elevation_at(exit_mp) - elevation_at(cfg.purge_start_mp),
                                cfg.fluid_sg)

    p = exit_psig + friction_psi + head_psi
    return float(max(exit_psig + 5.0, min(p, cfg.maop_psig * 0.9)))


def _compute_station_pressures(
    station_states: List[PumpStationState],
    pig_mp: float,
    pig_face_psig: float,
    exit_psig: float,
    exit_mp: float,
    gas_profile: List[tuple],
    od_in: float,
    wt_in: float,
    sg: float,
    viscosity_cst: float,
    roughness_ft: float,
    flow_bph: float,
    elevation_at,
) -> List[dict]:
    """
    Estimate pressure at each pump station for the current timestep.

    Bypassed stations (pig has passed): N2 gas pressure at their milepost, read from
    the step-function gas_profile.

    Running stations ahead of pig: liquid HGL discharge pressure. Computed by working
    from the exit condition backwards through the chain of running pumps. Each pump's
    discharge = pressure needed to push liquid to its downstream target (next pump
    suction or final exit), accounting for friction and static head.

    Shutdown stations ahead of pig (pump offline, pig hasn't reached them): passive
    liquid HGL pressure. Computed from pig face forward using friction and static head —
    the pig is pushing liquid through these points without pump assist.
    """
    D_ft  = (od_in - 2.0 * wt_in) / 12.0
    area  = pipe_area_ft2(od_in, wt_in)
    v_fts = bph_to_fts(flow_bph, area) if area > 0 else 0.0

    def _fric_and_head(from_mp: float, to_mp: float) -> tuple[float, float]:
        L_ft = max(0.0, to_mp - from_mp) * 5280.0
        fric = liquid_friction_loss_psi(L_ft, D_ft, v_fts, sg, viscosity_cst, roughness_ft)
        head = static_head_psi(elevation_at(to_mp) - elevation_at(from_mp), sg)
        return fric, head

    def _gas_pressure_at(mp: float) -> float:
        # Step-function lookup: largest mp in gas_profile that is <= target mp.
        # gas_profile is sorted ascending by mp.
        psig = pig_face_psig  # fallback if mp is beyond all gas profile points
        for gmp, gp in gas_profile:
            if gmp <= mp:
                psig = gp
            else:
                break
        return psig

    # --- Split stations by status ---
    running_ahead = sorted(
        [s for s in station_states if s.is_active and s.mp > pig_mp],
        key=lambda s: s.mp,
    )
    shutdown_ahead = [
        s for s in station_states
        if s.status == StationStatus.SHUT_DOWN and s.mp > pig_mp
    ]

    # --- Running stations: build HGL discharge from exit backwards ---
    discharge_by_mp: dict[float, float] = {}
    next_mp   = exit_mp
    next_psig = exit_psig
    for station in reversed(running_ahead):
        fric, head = _fric_and_head(station.mp, next_mp)
        discharge_by_mp[station.mp] = next_psig + fric + head
        next_mp   = station.mp
        next_psig = station.config.suction_psig

    # --- Shutdown stations ahead of pig: passive liquid HGL from pig face forward ---
    passive_by_mp: dict[float, float] = {}
    for station in shutdown_ahead:
        fric, head = _fric_and_head(pig_mp, station.mp)
        passive_by_mp[station.mp] = pig_face_psig - fric - head

    # --- Assemble results ---
    out = []
    for s in sorted(station_states, key=lambda st: st.mp):
        behind_pig = s.mp <= pig_mp  # pig has passed; station now in N2 gas column
        if behind_pig:
            p = _gas_pressure_at(s.mp)
            out.append({
                'mp': s.mp, 'name': s.config.name,
                'discharge_psig': p, 'suction_psig': p,
                'status': 'bypassed',
            })
        elif s.is_active:
            # Running pump ahead of pig
            disc = discharge_by_mp.get(s.mp, pig_face_psig)
            out.append({
                'mp': s.mp, 'name': s.config.name,
                'discharge_psig': disc,
                'suction_psig': s.config.suction_psig,
                'status': 'running',
            })
        else:
            # Pump is offline, pig hasn't passed it yet — passive liquid HGL
            p = passive_by_mp.get(s.mp, pig_face_psig)
            out.append({
                'mp': s.mp, 'name': s.config.name,
                'discharge_psig': p, 'suction_psig': p,
                'status': 'shutdown',
            })
    return out


def simulate(cfg: SimConfig, progress_cb: Optional[Callable[[float], None]] = None) -> SimResults:
    """
    Run the purge simulation.

    Args:
        cfg:         full simulation configuration
        progress_cb: optional callback(fraction_complete) called each step

    Returns:
        SimResults with full timestep history
    """
    t_wall_start = time.monotonic()
    results = SimResults(config=cfg)

    # --- Thin MOP joints (one-time preprocessing) ---
    # Reduces 33,000+ ILI joints to ~300–700 hydraulically critical points:
    # elevation peaks/valleys (RDP ε=3 ft), MOP step-change boundaries (≥25 psig),
    # and per-interval worst-case MOP. The full joint list stays in cfg for the
    # scenario file; only the thinned list is used during the loop.
    mop_joints = thin_mop_joints(cfg.mop_joints) if cfg.mop_joints else []

    # Rebuild BPCV downstream joints and upstream elevation array from the thinned set.
    # Both the MOP cap (downstream) and slack-line floor (upstream) benefit from reduction.
    thinned_elev_profile: Optional[np.ndarray] = None
    if cfg.bpcv is not None and mop_joints:
        from .bpcv import BCPVDownstreamJoint
        cfg.bpcv.downstream_joints = [
            BCPVDownstreamJoint(mp=j.mp, mop_psig=j.mop_psig, elevation_ft=j.elevation_ft)
            for j in mop_joints if j.mp > cfg.bpcv.mp
        ]
        # Thinned elevation profile for upstream slack-line check (pig → BPCV section)
        thinned_elev_profile = np.array(
            [[j.mp, j.elevation_ft] for j in mop_joints if j.mp <= cfg.bpcv.mp],
            dtype=float,
        ) if mop_joints else None

    # --- Pre-sort MOP joint arrays for fast per-step cap computation ---
    # Sorted numpy arrays allow O(log N) slicing each step instead of list comprehension.
    if mop_joints:
        _mj_mp   = np.array([j.mp          for j in mop_joints], dtype=float)
        _mj_mop  = np.array([j.mop_psig    for j in mop_joints], dtype=float)
        _mj_elev = np.array([j.elevation_ft for j in mop_joints], dtype=float)
        _mj_sort = np.argsort(_mj_mp)
        _mj_mp, _mj_mop, _mj_elev = _mj_mp[_mj_sort], _mj_mop[_mj_sort], _mj_elev[_mj_sort]
    else:
        _mj_mp = _mj_mop = _mj_elev = None

    # --- Elevation interpolator ---
    elev_arr = np.asarray(cfg.elevation_profile, dtype=float)
    elev_interp = interp1d(elev_arr[:, 0], elev_arr[:, 1], kind='linear',
                           bounds_error=False, fill_value=(elev_arr[0, 1], elev_arr[-1, 1]))
    elevation_at: Callable[[float], float] = lambda mp: float(elev_interp(mp))

    # --- Auto-compute initial N2 pressure (override user-supplied value) ---
    cfg.n2_initial_pressure_psig = compute_initial_n2_pressure(cfg, elevation_at)

    # --- Initialize segment list ---
    segs = SegmentList(cfg.pipe_geometry)
    segs.initialize_from_pig_at(
        pig_mp=cfg.purge_start_mp + 0.001,
        purge_start_mp=cfg.purge_start_mp,
        initial_pressure_psig=cfg.n2_initial_pressure_psig,
    )

    # --- Initialize booster states ---
    booster_states: List[BoosterState] = [
        BoosterState(config=bc) for bc in sorted(cfg.booster_configs, key=lambda b: b.mp)
    ]
    booster_boundary_inserted: set = set()   # track which MPs have had boundaries inserted

    # --- Initialize spread logistics ---
    # Each spread starts at one of the first N booster stations (pre-positioned before purge).
    # spread_assignment: booster_mp -> BoosterSpread currently assigned there
    spread_assignment: Dict[float, BoosterSpread] = {}
    if cfg.n_spreads > 0 and booster_states:
        for i in range(min(cfg.n_spreads, len(booster_states))):
            sp = BoosterSpread(spread_id=i,
                               current_mp=booster_states[i].mp,
                               available_at_hr=0.0)
            spread_assignment[booster_states[i].mp] = sp

    # --- Initialize pump station states ---
    station_states: List[PumpStationState] = [
        PumpStationState(config=pc) for pc in sorted(cfg.pump_stations, key=lambda s: s.mp)
    ]

    # --- Initialize BPCV ---
    bpcv_state: Optional[BPCVState] = BPCVState(config=cfg.bpcv) if cfg.bpcv else None

    # --- Check valves: sorted by mp ---
    check_valves = sorted(cfg.check_valves, key=lambda v: v.mp)

    # --- Pig solver config (updated each step for variable pipe) ---
    pig_mp = cfg.purge_start_mp + 0.001
    t_hr   = 0.0
    dt_hr  = cfg.dt_hr
    total_scf_injected = 0.0
    stall_start_t_hr: Optional[float] = None   # time when pig first stalled (None = not stalled)

    for step_num in range(cfg.max_steps):
        pig_elevation_ft = elevation_at(pig_mp)

        # --- Ensure check valve boundaries exist as pig approaches them ---
        for cv in check_valves:
            if pig_mp > cv.mp and cv.mp > cfg.purge_start_mp:
                insert_check_valve_boundary(segs, cv)

        # --- Spread arrivals: complete mobilizations that have finished ---
        if cfg.n_spreads > 0:
            for sp in spread_assignment.values():
                if sp.in_transit and t_hr >= sp.available_at_hr:
                    sp.arrive(t_hr)

        # Booster activation is demand-driven — see block after pig speed solve.

        # --- Get pipe props at pig location (needed for flow-rate conversions below) ---
        od_in, wt_in = cfg.pipe_geometry.od_wt_at(pig_mp)
        area_ft2 = pipe_area_ft2(od_in, wt_in)

        # --- BPCV update ---
        bpcv_set_point_psig: Optional[float] = None
        bpcv_gas_constraint: Optional[float] = None
        if bpcv_state is not None:
            bpcv_result = step_bpcv(
                bpcv_state, pig_mp,
                target_flow_bph=fts_to_bph(mph_to_fts(cfg.target_speed_mph), area_ft2),
                fluid_sg=cfg.fluid_sg,
                fluid_viscosity_cst=cfg.fluid_viscosity_cst,
                elevation_profile=thinned_elev_profile if thinned_elev_profile is not None else cfg.elevation_profile,
                upstream_od_in=od_in,
                upstream_wt_in=wt_in,
                upstream_roughness_ft=cfg.fluid_roughness_ft,
            )
            bpcv_set_point_psig = bpcv_result['set_point_psig']
            if bpcv_result['pig_has_passed']:
                # N2 must push through BPCV set point — add gas friction from pig to BPCV
                # This is an approximation; pig_solver handles the detailed balance
                bpcv_gas_constraint = bpcv_set_point_psig

        # --- Determine exit condition for pig's liquid push ---
        bpcv_mp   = cfg.bpcv.mp if cfg.bpcv else None
        tankage_mp = cfg.purge_end_mp
        exit_psig, exit_mp, exit_desc = effective_exit_condition(
            station_states, pig_mp,
            bpcv_set_point_psig, bpcv_mp,
            tankage_psig=target_exit_pressure(pig_mp, {
                'exit_pressure_run': cfg.exit_pressure_run_psig,
                'exit_pressure_end': cfg.exit_pressure_end_psig,
                'exit_pressure_behavior': cfg.exit_pressure_behavior,
            }, cfg.purge_start_mp, cfg.purge_end_mp, cfg.throttle_down_miles),
            tankage_mp=tankage_mp,
        )

        # --- Pig speed solve ---
        pig_solver_cfg = PigSolverConfig(
            od_in=od_in,
            wt_in=wt_in,
            roughness_ft=cfg.fluid_roughness_ft,
            sg=cfg.fluid_sg,
            viscosity_cst=cfg.fluid_viscosity_cst,
            min_speed_mph=cfg.min_speed_mph,
            max_speed_mph=cfg.max_speed_mph,
            target_speed_mph=cfg.target_speed_mph,
            temperature_f=cfg.n2_temperature_f,
        )
        pig_face_psig = segs.pig_face_pressure_psig()
        pig_result: PigSolverResult = solve_pig_speed(
            pig_solver_cfg, pig_mp, pig_elevation_ft,
            exit_psig, exit_mp, exit_desc,
            pig_face_psig,
            elevation_at,
            bpcv_gas_constraint_psig=bpcv_gas_constraint,
        )

        # --- Activate boosters (unconditional) ---
        # Each booster needs a segment boundary so it can maintain a pressure
        # differential. Physically this is the booster's discharge check valve.
        # If no check valve already exists at the booster's MP, auto-insert one.
        #
        # With spread logistics (n_spreads > 0): a booster activates as soon as
        # the pig passes the station and a spread is present and rigged up there.
        # The booster always provides value — even when the pig is at target speed,
        # the booster compresses upstream N2 downstream so injection only needs to
        # maintain the short upstream stub at suction floor pressure (not pig-face).
        # Without spread constraints: legacy unlimited-spread mode (every station fires).
        for bs in booster_states:
            if cfg.n_spreads > 0:
                spread = spread_assignment.get(bs.mp)
                if (spread is not None and spread.is_available_at(t_hr)
                        and not bs.requested and pig_mp > bs.mp):
                    bs.requested = True
                    newly = True
                else:
                    newly = False
            else:
                # Unlimited spreads (legacy): activate immediately when pig passes
                newly = activate_booster_if_pig_passed(bs, pig_mp)

            if newly and bs.mp not in booster_boundary_inserted:
                if not any(abs(existing_cv.mp - bs.mp) < 0.001 for existing_cv in check_valves):
                    bcv = CheckValve(mp=bs.mp, name=f"{bs.config.name} boundary")
                    insert_check_valve_boundary(segs, bcv)
                    check_valves.append(bcv)
                    check_valves.sort(key=lambda v: v.mp)
                booster_boundary_inserted.add(bs.mp)

        # --- Adaptive timestep ---
        if cfg.adaptive_dt and pig_result.pig_speed_mph > 0:
            dt_distance_hr = 0.1 / max(0.001, pig_result.pig_speed_mph)  # 0.1-mile steps
            dt_hr = max(cfg.dt_min_hr, min(dt_distance_hr, cfg.dt_max_hr))
        else:
            dt_hr = cfg.dt_hr

        # --- Advance pig ---
        d_miles     = pig_result.pig_speed_mph * dt_hr
        new_pig_mp  = pig_mp + d_miles
        delta_vol   = segs.advance_pig(new_pig_mp)   # ft³ vacated by pig

        # --- Compute required injection ---
        # Injection must maintain the most-upstream segment above all booster suction floors
        min_suction_floor = 0.0
        for bs in booster_states:
            if bs.requested:
                min_suction_floor = max(min_suction_floor, bs.config.suction_min_psig)

        inj_seg = segs.injection_segment()
        inj_vol_ft3  = inj_seg.volume_ft3
        # Deficit: how much SCF we need to keep injection segment at floor pressure
        scf_at_floor = scf_from_pressure_volume(psig_to_psia(min_suction_floor), inj_vol_ft3, cfg.n2_temperature_f)
        scf_deficit  = max(0.0, scf_at_floor - inj_seg.scf)

        # Fill pressure determines how hard injection pushes.
        #
        # Meter valve active (pig at target speed): throttle to the minimum face pressure
        # needed to sustain current speed + a small margin.
        #
        # Drive-limited with boosters running: injection only needs to supply the booster
        # suction floor. The booster chain handles downstream pressure — injecting to
        # pig_face_psig would wastefully over-pressurize the upstream segment.
        #
        # Drive-limited with no boosters: injection is the sole pressure source; target
        # is pig face pressure (maintain what we have, booster-chain can't help here).
        #
        # Stalled (speed=0), no boosters: inject at max rate — the only way to build
        # pig face pressure is to keep pushing gas in from origin.
        is_stalled = (pig_result.pig_speed_mph == 0.0)

        # Minimum pig face needed to sustain pig motion at target speed.
        # Used to:
        #   (a) compute the correct mass injection when boosters are running
        #   (b) cap the last active booster's discharge so pig face stays at minimum needed
        #       rather than always running to rated discharge (which wastes stored N2 and
        #       raises consumption proportional to pig face pressure).
        _min_face_psig = (pig_result.exit_psig
                          + pig_result.liquid_friction_psi
                          + pig_result.static_head_psi)

        if pig_result.meter_valve_active and not is_stalled:
            if min_suction_floor > 0:
                # Boosters running at target speed: injection stub only needs suction floor.
                fill_pressure_psig = min_suction_floor
                # Pig face target = minimum needed + margin. The booster is capped here
                # so SCF consumed per unit advance ∝ pig face P (mass conservation).
                pig_face_target_psig = _min_face_psig + 20.0
            else:
                # No boosters: injection IS the pig face.
                fill_pressure_psig = _min_face_psig + 20.0
                pig_face_target_psig = fill_pressure_psig
        elif min_suction_floor > 0:
            # Boosters running but drive-limited: want max booster output to recover speed.
            # Injection stub still only needs suction floor; booster discharge is uncapped.
            fill_pressure_psig = min_suction_floor
            pig_face_target_psig = max(
                (bs.config.discharge_psig for bs in booster_states if bs.requested),
                default=pig_face_psig,
            )
        else:
            # No boosters, drive-limited: maintain current pig face pressure.
            # n2_initial_pressure_psig must NOT be used as a floor here — it was only for
            # initialization and would force 600–900 psig injection in steady state.
            fill_pressure_psig = pig_face_psig
            pig_face_target_psig = fill_pressure_psig

        # --- Transition lookahead: pre-build face to survive the next station shutdown ---
        #
        # Without this, injection targets only the *current* exit (e.g., NW suction 1 mile
        # ahead at 55 psig). When NW's 1-mile limit fires, pig suddenly must push 52 miles
        # to LS — far beyond current face pressure — and stalls.
        #
        # Fix: at all times, ensure face >= minimum needed to push from pig's current position
        # to the station AFTER the current exit target, at minimum speed. This means injection
        # continuously pre-builds the pressure cushion required for each upcoming handoff.
        if min_suction_floor == 0.0:
            _bpcv_mp_tl   = cfg.bpcv.mp if cfg.bpcv else None
            if _bpcv_mp_tl is not None and pig_mp < _bpcv_mp_tl:
                _ult_mp_tl   = _bpcv_mp_tl
                _ult_psig_tl = bpcv_set_point_psig if bpcv_set_point_psig is not None else cfg.exit_pressure_run_psig
            else:
                _ult_mp_tl   = cfg.purge_end_mp
                _ult_psig_tl = cfg.exit_pressure_run_psig

            _cur_exit_sta = next_active_station(station_states, pig_mp)
            if _cur_exit_sta is not None:
                _after = sorted(
                    [s for s in station_states if s.is_active and s.mp > _cur_exit_sta.mp],
                    key=lambda s: s.mp,
                )
                _post_mp   = _after[0].mp                    if _after else _ult_mp_tl
                _post_psig = _after[0].config.suction_psig   if _after else _ult_psig_tl
                _D_ft_tl   = (od_in - 2.0 * wt_in) / 12.0
                _v_min_tl  = mph_to_fts(cfg.min_speed_mph)
                _L_tl      = max(0.0, _post_mp - pig_mp) * 5280.0
                _fric_tl   = liquid_friction_loss_psi(
                    _L_tl, _D_ft_tl, _v_min_tl,
                    cfg.fluid_sg, cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft,
                )
                _head_tl   = static_head_psi(
                    elevation_at(_post_mp) - pig_elevation_ft, cfg.fluid_sg,
                )
                _transition_floor = max(0.0, _post_psig + _fric_tl + _head_tl)
                fill_pressure_psig     = max(fill_pressure_psig,     _transition_floor)
                pig_face_target_psig   = max(pig_face_target_psig,   _transition_floor)

        # --- MOP hard ceiling ---
        # MOP violations are zero-tolerance. The pig slows below configured minimum
        # before we allow any joint to be overpressured. Cap ALL injection and booster
        # targets at the maximum pressure the weakest joint can tolerate.
        #
        # Gas side:    min MOP of joints in the N2 column (purge_start → pig_mp).
        #              Injection and booster discharge cannot push any segment above this.
        #
        # Liquid side: min_j [ MOP_j + friction(pig→j) + head(pig→j) ] for j downstream.
        #              If pig face exceeds this, pressure at joint j would exceed MOP_j.
        mop_cap = math.inf
        if _mj_mp is not None:
            # Gas side cap: include a 20-mile lookahead so face ramps DOWN before a
            # low-MOP joint enters the gas column (not after). Uses a separate index
            # from the liquid side so the two computations don't interfere.
            _GAS_LOOKAHEAD_MI = 20.0
            g_hi_gas = int(np.searchsorted(_mj_mp, pig_mp + _GAS_LOOKAHEAD_MI, side='right'))
            if g_hi_gas > 0:
                mop_cap = float(np.min(_mj_mop[:g_hi_gas]))

            # Liquid side cap: joints between pig_mp and exit_mp.
            # Uses pig_mp (not pig+lookahead) so nearby downhill joints are captured.
            pig_idx = int(np.searchsorted(_mj_mp, pig_mp, side='right'))
            l_hi = int(np.searchsorted(_mj_mp, exit_mp, side='right'))
            if pig_idx < l_hi:
                lj_mps   = _mj_mp  [pig_idx:l_hi]
                lj_mops  = _mj_mop [pig_idx:l_hi]
                lj_elevs = _mj_elev[pig_idx:l_hi]
                _D_ft_c  = (od_in - 2.0 * wt_in) / 12.0
                _v_c     = mph_to_fts(cfg.target_speed_mph)
                _m       = 0.3048
                _D_m, _v_m = _D_ft_c * _m, _v_c * _m
                _rho = 999.0 * cfg.fluid_sg
                _nu  = cfg.fluid_viscosity_cst * 1e-6
                _Re  = _v_m * _D_m / max(1e-12, _nu)
                _eps = cfg.fluid_roughness_ft * _m
                _f   = (64.0 / max(1.0, _Re)) if _Re < 2300 else \
                       (0.25 / (math.log10(_eps / (3.7 * _D_m) + 5.74 / _Re**0.9)) ** 2)
                _fric_pft = _f / _D_m * 0.5 * _rho * _v_m**2 / 6894.757 * _m
                _L_arr    = (lj_mps - pig_mp) * 5280.0
                # head_arr = (j_elev - pig_elev) × sg × 0.433
                # Positive when j is above pig (uphill → less pressure at j → higher pig cap allowed)
                # Negative when j is below pig (downhill → more pressure at j → lower pig cap required)
                # Bernoulli: P_pig_max = MOP_j + head_arr + fric
                _head_arr = (lj_elevs - pig_elevation_ft) * cfg.fluid_sg * 62.4 / 144.0
                liq_cap   = float(np.min(lj_mops + _fric_pft * _L_arr + _head_arr))
                mop_cap   = min(mop_cap, liq_cap)

        if math.isfinite(mop_cap) and mop_cap > 0:
            fill_pressure_psig   = min(fill_pressure_psig,   mop_cap)
            pig_face_target_psig = min(pig_face_target_psig, mop_cap)

        # SCF to fill the volume vacated by pig advance.
        # When boosters run, this mass enters the injection stub, is compressed by the
        # booster chain, and fills the expanding pig face volume. Mass is conserved through
        # the chain, so the injection amount must be calculated at pig face target pressure
        # (not at the stub fill pressure, which is much lower).
        scf_to_fill_new_vol = scf_from_pressure_volume(
            psig_to_psia(pig_face_target_psig), delta_vol, cfg.n2_temperature_f
        )
        scf_inject = scf_deficit + scf_to_fill_new_vol

        # Drive-limited without boosters: also pressurize injection segment to fill_pressure
        # so pig face builds up from injection directly.
        if not pig_result.meter_valve_active and min_suction_floor == 0:
            target_scf = inj_seg.scf_at_pressure(fill_pressure_psig)
            scf_inject = max(scf_inject, target_scf - inj_seg.scf)

        # Stalled with no boosters: inject at max rate regardless of fill calculation —
        # only way to build pig-face pressure and recover.
        if is_stalled and min_suction_floor == 0.0:
            scf_inject = max(scf_inject, cfg.max_injection_scfm * dt_hr * 60.0)

        # Cap by max rate
        max_scf_step = cfg.max_injection_scfm * dt_hr * 60.0
        scf_inject = min(max(0.0, scf_inject), max_scf_step)

        inj_seg.add_scf(scf_inject)
        total_scf_injected += scf_inject

        # Compute injection pressure (the first segment's current pressure)
        injection_psig = inj_seg.pressure_psig
        injection_scfm = scf_inject / max(1e-12, dt_hr * 60.0)

        # --- Step boosters ---
        # Only the pig-adjacent (last active) booster gets a discharge cap — it controls
        # pig face pressure. Earlier boosters in the chain should run at rated discharge to
        # build up intermediate segment pressure for the last stage to draw from.
        # In drive-limited mode (pig below target speed), all boosters run uncapped to
        # build maximum pressure and recover pig speed.
        if pig_result.meter_valve_active and min_suction_floor > 0:
            last_active_mp = max(
                (bs.mp for bs in booster_states if bs.requested and bs.mp < pig_mp),
                default=-1.0,
            )
        else:
            last_active_mp = -1.0  # no cap in drive-limited mode or no-booster mode

        active_mps = active_booster_mps(booster_states)
        booster_step_results = []
        for bs in booster_states:
            if bs.mp < pig_mp:  # only boosters the pig has passed
                tgt_discharge = (
                    pig_face_target_psig
                    if (last_active_mp >= 0 and abs(bs.mp - last_active_mp) < 0.001)
                    else None
                )
                # MOP cap applies to every booster, not just the last active.
                # A booster must never discharge into a segment above the minimum
                # MOP of joints within that segment or anywhere downstream.
                if math.isfinite(mop_cap) and mop_cap > 0:
                    tgt_discharge = min(tgt_discharge, mop_cap) if tgt_discharge is not None \
                                    else mop_cap
                bsr = step_booster(bs, segs, dt_hr, pig_face_psig,
                                   target_discharge_psig=tgt_discharge)
                booster_step_results.append({
                    'mp': bs.mp,
                    'name': bs.config.name,
                    **bsr,
                })
        active_mps = active_booster_mps(booster_states)

        # --- Release spreads from spent booster stations ---
        # Spreads move forward when:
        #   1. Their booster shuts down (suction too low or no SCF to transfer)
        #   2. Proactive: pig is close enough to the next uncovered station that the
        #      spread must leave NOW to arrive before the pig (mob_time + 2h margin).
        if cfg.n_spreads > 0:
            covered_mps = set(spread_assignment.keys())

            for mp in list(spread_assignment.keys()):
                if mp not in spread_assignment:
                    continue  # already released this iteration
                spread = spread_assignment[mp]
                if spread.in_transit:
                    continue

                # Find next uncovered booster station ahead
                next_target = None
                for next_bs in sorted(booster_states, key=lambda b: b.mp):
                    if next_bs.mp > mp and next_bs.mp not in covered_mps:
                        next_target = next_bs.mp
                        break
                if next_target is None:
                    continue  # no forward station to move to

                # Forced release: booster at this station shut down
                booster_shut_down = any(
                    abs(bsr['mp'] - mp) < 0.001
                    and bsr['shutoff_reason'] in ('suction_too_low', 'no_scf_to_transfer')
                    for bsr in booster_step_results
                )

                # Proactive release: pig has passed this station and the spread must
                # leave NOW or it will miss the pig at next_target.
                # Use actual pig speed (floor = target/2) so the spread won't release
                # while the pig is stalled or slow — it stays and lets the booster run.
                # When the pig recovers toward target speed, the timing naturally re-arms.
                effective_speed = max(pig_result.pig_speed_mph, cfg.target_speed_mph * 0.5)
                t_to_next = (next_target - pig_mp) / effective_speed
                # Also require pig to have traveled a minimum distance past this station
                # before we release, so the booster gets at least a few hours of service.
                min_hold_miles = max(5.0, cfg.mob_time_hr * cfg.target_speed_mph / 6.0)
                proactive_release = (
                    pig_mp > mp
                    and pig_mp < next_target
                    and pig_mp - mp >= min_hold_miles
                    and t_to_next <= cfg.mob_time_hr + 2.0
                )

                if booster_shut_down or proactive_release:
                    reason = 'suction_shutoff' if booster_shut_down else 'proactive'
                    del spread_assignment[mp]
                    covered_mps.discard(mp)
                    spread_assignment[next_target] = spread
                    covered_mps.add(next_target)
                    spread.mobilize_to(next_target, t_hr, cfg.mob_time_hr)
                    results.spread_events.append({
                        'spread_id': spread.spread_id,
                        'from_mp': mp,
                        'to_mp': next_target,
                        'reason': reason,
                        't_hr': t_hr,
                        'arrive_at_hr': spread.available_at_hr,
                    })
                    for old_bs in booster_states:
                        if abs(old_bs.mp - mp) < 0.001:
                            old_bs.requested = False
                            break

        # --- Check valve cascade merges ---
        evaluate_check_valves(segs, check_valves, active_mps)

        # --- Pump station shutdowns ---
        # Ultimate endpoint for hydraulic necessity check: BPCV (if upstream) or tankage
        if bpcv_mp is not None and pig_mp < bpcv_mp:
            _ult_psig = bpcv_set_point_psig if bpcv_set_point_psig is not None else cfg.exit_pressure_run_psig
            _ult_mp   = bpcv_mp
        else:
            _ult_psig = cfg.exit_pressure_run_psig
            _ult_mp   = cfg.purge_end_mp

        # Cap effective pig-face at the injection-rate-limited sustainable pressure.
        # The initial N2 fill at 900 psig is transient — once the pig moves, injection
        # alone can only sustain P = Q_inj * 14.7 / (60 * v_tgt * A).  Using the raw
        # 900 psig peak causes all stations to appear hydraulically unnecessary at t=0.
        _v_tgt = mph_to_fts(cfg.target_speed_mph)
        _inj_floor_psia = cfg.max_injection_scfm * ATM_PSI / max(1e-9, 60.0 * _v_tgt * area_ft2)
        _inj_floor_psig = max(0.0, _inj_floor_psia - ATM_PSI)
        _face_for_shutdown = min(pig_face_psig, _inj_floor_psig)

        shutdown_events = evaluate_shutdowns(
            station_states, pig_mp, _face_for_shutdown,
            ultimate_endpoint_psig=_ult_psig,
            ultimate_endpoint_mp=_ult_mp,
            flow_bph=fts_to_bph(mph_to_fts(cfg.target_speed_mph), area_ft2),
            od_in=od_in, wt_in=wt_in,
            sg=cfg.fluid_sg, viscosity_cst=cfg.fluid_viscosity_cst,
            roughness_ft=cfg.fluid_roughness_ft,
            elevation_at=elevation_at,
            sim_time_hr=t_hr,
        )

        # --- MOP check and station pressures ---
        gas_profile = build_gas_pressure_profile(segs.summary())
        station_pressures = _compute_station_pressures(
            station_states, pig_mp, pig_face_psig,
            exit_psig, exit_mp, gas_profile,
            od_in, wt_in,
            cfg.fluid_sg, cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft,
            flow_bph=fts_to_bph(mph_to_fts(cfg.target_speed_mph), area_ft2),
            elevation_at=elevation_at,
        )
        mop_liq = check_mop_liquid_side(
            pig_mp, pig_face_psig, mop_joints,
            exit_psig, exit_mp,
            fts_to_bph(mph_to_fts(pig_result.pig_speed_mph), area_ft2),
            cfg.fluid_sg, cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft,
            cfg.mop_warning_fraction,
            pig_elevation_ft=pig_elevation_ft,
        )
        mop_gas = check_mop_gas_side(
            pig_mp, cfg.purge_start_mp, gas_profile, mop_joints, cfg.mop_warning_fraction,
        )
        all_mop = mop_liq + mop_gas
        mop_sum = mop_summary(all_mop)

        # --- Record step ---
        step = SimStep(
            t_hr=t_hr,
            pig_mp=pig_mp,
            pig_elevation_ft=pig_elevation_ft,
            pig_speed_mph=pig_result.pig_speed_mph,
            total_scf=segs.total_scf(),
            injection_scfm=injection_scfm,
            injection_psig=injection_psig,
            pig_face_psig=pig_face_psig,
            segments=segs.summary(),
            gas_pressure_profile=gas_profile,
            exit_psig=exit_psig,
            exit_mp=exit_mp,
            exit_description=exit_desc,
            mop_violations=mop_sum['n_violations'],
            mop_warnings=mop_sum['n_warnings'],
            worst_mop_mp=mop_sum['worst_violation_mp'],
            worst_mop_margin=mop_sum['worst_violation_margin'],
            mop_results=all_mop,
            booster_states=booster_step_results,
            station_pressures=station_pressures,
            station_shutdown_events=shutdown_events,
            slack_line_risk=pig_result.slack_line_risk,
            meter_valve_active=pig_result.meter_valve_active,
        )
        results.steps.append(step)

        # --- Progress callback ---
        if progress_cb:
            frac = (pig_mp - cfg.purge_start_mp) / max(1e-6, cfg.purge_end_mp - cfg.purge_start_mp)
            progress_cb(min(1.0, frac))

        # --- Advance time and pig ---
        t_hr   += dt_hr
        pig_mp  = new_pig_mp

        # --- Termination ---
        if pig_mp >= cfg.purge_end_mp:
            results.completed = True
            break

        if pig_result.pig_speed_mph <= 0.0 and step_num > 10:
            if stall_start_t_hr is None:
                stall_start_t_hr = t_hr
            stall_duration_hr = t_hr - stall_start_t_hr
            if stall_duration_hr >= cfg.max_stall_hr:
                # Determine whether the stall is a flow bottleneck or pressure issue
                flow_limited_boosters = [
                    bsr for bsr in booster_step_results
                    if bsr.get('running') and bsr.get('flow_limited')
                ]
                if flow_limited_boosters:
                    stations = ', '.join(bsr.get('name', f"MP {bsr['mp']:.2f}")
                                        for bsr in flow_limited_boosters)
                    results.abort_reason = (
                        f"Pig stalled — booster throughput is the bottleneck at: {stations}. "
                        f"Spread SCFM capacity is the limiting factor; add compressors to the spread."
                    )
                else:
                    results.abort_reason = (
                        f"Pig stalled for {stall_duration_hr:.1f} hr — "
                        f"insufficient drive pressure (pig face {pig_face_psig:.0f} psig, "
                        f"exit {exit_psig:.0f} psig at MP {exit_mp:.1f})"
                    )
                break
        else:
            stall_start_t_hr = None  # reset if pig starts moving again

    results.total_scf_injected = total_scf_injected
    results.wall_time_s = time.monotonic() - t_wall_start
    return results
