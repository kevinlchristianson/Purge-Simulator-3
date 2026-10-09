"""
Pig speed and drive pressure solver.

The pig's speed at any moment is determined by a pressure balance:
  P_behind_pig (N2 drive) = P_in_front_of_pig (liquid resistance)

Liquid resistance components (ahead of pig):
  P_front = exit_psig + liquid_friction(pig→exit) + static_head(pig→exit)

The solver bisects on pig velocity to find the speed where N2 drive = liquid resistance.

Constraints:
  - max_speed_mph: physical or operational maximum
  - min_speed_mph: minimum to prevent slack line

The exit (the nearest thing downstream of the pig: a running pump's suction, the BPCV,
or the tank inlet) holds a MINIMUM inlet pressure, `exit_psig`. When the drive would push
the pig faster than max speed, the exit MODULATES: the client's SCADA raises that inlet
pressure (pump slowing, tank inlet valve pinching, BPCV set point rising) by exactly the
surplus, so the pig holds max speed. The device can only hold so much, `exit_max_psig`;
once that is reached the pig overspeeds and the step is flagged (never hidden). A FIXED
exit is the same with exit_max_psig == exit_psig: it adds nothing and the pig overspeeds.
exit_max_psig=None is unbounded (the legacy behavior: the pig is pinned at max speed and
the exit holds whatever that takes).

Slack line risk:
  If the achievable speed < min_speed_mph, the liquid column is pulling faster than
  the pig can push, and a slack line condition is flagged.

BPCV phase:
  Before pig reaches BPCV: BPCV set point is the exit condition for the liquid column.
  After pig passes BPCV: the pig is in the N2 section; BPCV is now a constraint on
  N2 pressure — pig-face pressure must exceed BPCV set point + gas friction from pig to BPCV.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import math
import numpy as np

from .physics import (
    liquid_friction_loss_psi,
    gas_friction_loss_psi,
    static_head_psi,
    pipe_area_ft2,
    bph_to_fts,
    fts_to_bph,
    fts_to_mph,
    mph_to_fts,
    psig_to_psia,
    scfm_from_pig_velocity,
    clearance_candidates,
)


@dataclass
class PigSolverConfig:
    """Static configuration for the pig speed solver."""
    # Pipe geometry (at pig location — updated each step for variable-OD pipe)
    od_in: float
    wt_in: float
    roughness_ft: float

    # Fluid properties
    sg: float
    viscosity_cst: float

    # Speed limits
    min_speed_mph: float = 0.5
    max_speed_mph: float = 6.0
    target_speed_mph: float = 3.0

    # N2 temperature
    temperature_f: float = 45.0


@dataclass
class PigSolverResult:
    """Output from one pig speed solve."""
    pig_speed_mph: float
    pig_speed_fts: float
    drive_psig: float          # required N2 pressure immediately behind pig
    exit_psig: float           # exit inlet pressure actually held this step (minimum + added)
    exit_mp: float
    exit_description: str

    liquid_friction_psi: float
    static_head_psi: float

    # The pig has surplus drive (it is at or above target speed); the exit is what holds it
    # back. Kept under its historical name for the injection controller and the reports.
    meter_valve_active: bool = False
    meter_valve_back_pressure_psi: float = 0.0   # alias of endpoint_added_psi

    # Exit modulation: the minimum the exit holds, what it added this step to keep the pig
    # at max speed, and whether it ran out of room (pig over max speed).
    exit_min_psig: float = 0.0
    endpoint_added_psi: float = 0.0
    overspeed: bool = False
    overspeed_mph: float = 0.0

    # Slack line flag
    slack_line_risk: bool = False
    slack_line_deficit_mph: float = 0.0

    # Injection requirement (SCFM to maintain N2 column)
    injection_scfm: float = 0.0

    # Pig-face drive (psig) that would hold exactly target speed from here (peak-aware
    # resistance at v_target). The smooth lean controller aims the face at this.
    drive_for_target_psig: float = 0.0


def solve_pig_speed(
    cfg: PigSolverConfig,
    pig_mp: float,
    pig_elevation_ft: float,
    exit_psig: float,
    exit_mp: float,
    exit_description: str,
    pig_face_psig: float,          # current N2 pressure behind pig (from segment model)
    elevation_at: callable,        # elevation_at(mp) -> ft
    bpcv_gas_constraint_psig: Optional[float] = None,  # min pig-face N2 after pig passes BPCV
    mop_drive_ceiling_psig: Optional[float] = None,    # max pig-face N2 allowed by MOP
    terrain_mp=None,               # route terrain vertices (mp) between pig and exit
    terrain_elev=None,             # route terrain vertices (elevation ft)
    slack_margin_psi: float = 25.0,
    exit_max_psig: Optional[float] = None,   # most the exit can raise its inlet pressure to
) -> PigSolverResult:
    """
    Solve for pig speed given current system state.

    The pig speed is found by balancing the available drive pressure against
    the liquid-side resistance to produce a speed. Above max speed the exit
    modulates (raises its inlet pressure) to hold the pig there.

    Steps:
      1. Compute liquid resistance at target speed (friction + static head to exit)
      2. Check if pig_face_psig can achieve at least min_speed
      3. Between min and max speed the pig runs where drive = resistance
      4. Past max speed the exit adds back-pressure, up to exit_max_psig, to hold max;
         past that the pig overspeeds and is flagged
      5. Flag slack line if achievable < min

    `exit_psig` is the minimum the exit holds; `exit_max_psig` is the most it can hold
    (None = no limit, the exit holds whatever max speed takes; == exit_psig = fixed).

    MOP is zero-tolerance. `mop_drive_ceiling_psig` is the highest pig-face N2
    pressure the weakest joint (gas column or downstream liquid) can tolerate.
    The drive available to push the pig is capped at this ceiling, so when the
    system *could* supply more pressure than MOP allows, the pig is throttled —
    below min_speed if necessary — rather than overpressuring a joint. This is
    the only place pig speed is allowed to fall below min_speed by design.
    """
    area_ft2 = pipe_area_ft2(cfg.od_in, cfg.wt_in)
    D_ft     = (cfg.od_in - 2 * cfg.wt_in) / 12.0
    eps_ft   = cfg.roughness_ft

    exit_elevation_ft = elevation_at(exit_mp)
    L_liq_ft = max(0.0, exit_mp - pig_mp) * 5280.0

    head_psi = static_head_psi(exit_elevation_ft - pig_elevation_ft, cfg.sg)

    # Peak-aware resistance: the drive must keep the HGL above terrain at EVERY point ahead,
    # not just the exit. Precompute the clearance constraints once (base + length per
    # candidate); friction is linear in length, so resistance(v) = max(base + fric_per_ft*L).
    # With no terrain supplied this degenerates to the single exit term (legacy behavior).
    _cl_base, _cl_len = clearance_candidates(
        pig_mp, pig_elevation_ft, exit_mp, exit_psig, exit_elevation_ft,
        terrain_mp, terrain_elev, cfg.sg, slack_margin_psi,
    )

    # Bisect on speed to find where drive = resistance. `added_psi` is back-pressure the exit
    # has put on top of its minimum: it raises the exit term (candidate 0) only, never the
    # peak-clearing terms behind it.
    def _resistance_at_speed(v_fts: float, added_psi: float = 0.0) -> float:
        fpf = liquid_friction_loss_psi(1.0, D_ft, v_fts, cfg.sg, cfg.viscosity_cst, eps_ft)
        res = _cl_base + fpf * _cl_len
        return float(max(res[0] + added_psi, np.max(res)))

    def _exit_term_at_speed(v_fts: float) -> float:
        # Pig face needed to deliver exit_psig at the exit alone (no peak terms).
        fpf = liquid_friction_loss_psi(1.0, D_ft, v_fts, cfg.sg, cfg.viscosity_cst, eps_ft)
        return float(_cl_base[0] + fpf * _cl_len[0])

    def _drive_available() -> float:
        # If BPCV constraint active (pig past BPCV in gas phase), drive is limited by BPCV.
        # Zero set point means BPCV is wide open — no constraint.
        d = pig_face_psig
        if bpcv_gas_constraint_psig is not None and bpcv_gas_constraint_psig > 0:
            d = min(d, bpcv_gas_constraint_psig)
        # MOP hard ceiling: never use more drive than the weakest joint can hold.
        # If this forces achievable speed below min_speed, the pig slows (or stalls)
        # instead of overpressuring a joint — MOP is zero-tolerance.
        if mop_drive_ceiling_psig is not None and mop_drive_ceiling_psig > 0:
            d = min(d, mop_drive_ceiling_psig)
        return d

    drive = _drive_available()
    v_min = mph_to_fts(cfg.min_speed_mph)
    v_max = mph_to_fts(cfg.max_speed_mph)
    v_tgt = mph_to_fts(cfg.target_speed_mph)

    # Check if drive can sustain minimum speed
    res_at_min = _resistance_at_speed(v_min)
    res_at_tgt = _resistance_at_speed(v_tgt)
    if drive < res_at_min:
        # Pig cannot maintain minimum speed — slack line risk
        # Compute actual achievable speed by bisection
        if drive <= _resistance_at_speed(0.0):
            v_achievable = 0.0
        else:
            # Bisect between 0 and v_min
            lo, hi = 0.0, v_min
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if _resistance_at_speed(mid) <= drive:
                    lo = mid
                else:
                    hi = mid
            v_achievable = lo

        fric = liquid_friction_loss_psi(L_liq_ft, D_ft, v_achievable, cfg.sg, cfg.viscosity_cst, eps_ft)
        return PigSolverResult(
            pig_speed_mph=fts_to_mph(v_achievable),
            pig_speed_fts=v_achievable,
            drive_psig=drive,
            exit_psig=exit_psig,
            exit_mp=exit_mp,
            exit_description=exit_description,
            liquid_friction_psi=fric,
            static_head_psi=head_psi,
            exit_min_psig=exit_psig,
            slack_line_risk=True,
            slack_line_deficit_mph=fts_to_mph(v_min - v_achievable),
            injection_scfm=scfm_from_pig_velocity(v_achievable, area_ft2,
                                                   psig_to_psia(pig_face_psig), cfg.temperature_f),
            drive_for_target_psig=res_at_tgt,
        )

    # Drive can sustain at least minimum. Find achievable speed.
    meter_valve_active = False
    added_psi = 0.0
    overspeed = False

    if drive >= res_at_tgt:
        # Enough drive for at least target speed. The pig is allowed to run FASTER
        # than target — up to max_speed — when drive is excessive (downhill static-head
        # assist, surplus N2, or booster overshoot). Symmetric with the slow-down case:
        # just as the pig may fall below min_speed when drive is MOP-limited, it may rise
        # above target when drive is abundant.
        meter_valve_active = True
        res_at_max = _resistance_at_speed(v_max)
        if drive >= res_at_max:
            # At max speed the exit modulates: the nearest downstream device raises its
            # inlet pressure by the surplus drive over what the liquid column itself takes at
            # max speed (friction + head to the exit), and the pig holds max speed. The device
            # can only hold so much; past its maximum the pig overspeeds, flagged below.
            needed = max(0.0, drive - _exit_term_at_speed(v_max))
            room = math.inf if exit_max_psig is None else max(0.0, exit_max_psig - exit_psig)
            added_psi = min(needed, room)
            if added_psi >= needed - 1e-9:
                v_actual = v_max
            else:
                # Exit at its limit: solve the speed where the column's resistance, with the
                # exit holding its maximum, equals the drive. Friction grows with speed, so
                # bracket upward from max speed and bisect.
                lo, hi = v_max, 2.0 * v_max
                while _resistance_at_speed(hi, added_psi) < drive and hi < 64.0 * v_max:
                    lo, hi = hi, 2.0 * hi
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if _resistance_at_speed(mid, added_psi) <= drive:
                        lo = mid
                    else:
                        hi = mid
                v_actual = lo
                overspeed = v_actual > v_max * (1.0 + 1e-6)
        else:
            # Natural speed between target and max where drive = resistance
            lo, hi = v_tgt, v_max
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if _resistance_at_speed(mid) <= drive:
                    lo = mid
                else:
                    hi = mid
            v_actual = lo
    else:
        # Pig speed is between min and target — bisect
        lo, hi = v_min, v_tgt
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if _resistance_at_speed(mid) <= drive:
                lo = mid
            else:
                hi = mid
        v_actual = lo

    fric = liquid_friction_loss_psi(L_liq_ft, D_ft, v_actual, cfg.sg, cfg.viscosity_cst, eps_ft)
    inj_scfm = scfm_from_pig_velocity(v_actual, area_ft2,
                                       psig_to_psia(pig_face_psig), cfg.temperature_f)

    return PigSolverResult(
        pig_speed_mph=fts_to_mph(v_actual),
        pig_speed_fts=v_actual,
        drive_psig=drive,
        exit_psig=exit_psig + added_psi,
        exit_mp=exit_mp,
        exit_description=exit_description,
        liquid_friction_psi=fric,
        static_head_psi=head_psi,
        meter_valve_active=meter_valve_active,
        meter_valve_back_pressure_psi=added_psi,
        exit_min_psig=exit_psig,
        endpoint_added_psi=added_psi,
        overspeed=overspeed,
        overspeed_mph=fts_to_mph(max(0.0, v_actual - v_max)) if overspeed else 0.0,
        injection_scfm=inj_scfm,
        drive_for_target_psig=res_at_tgt,
    )


def minimum_n2_floor_to_sustain_flow(
    cfg: PigSolverConfig,
    pig_mp: float,
    exit_psig: float,
    exit_mp: float,
    elevation_at: callable,
    target_speed_mph: Optional[float] = None,
) -> float:
    """
    Return the minimum N2 pig-face pressure (psig) required to maintain
    target speed (or cfg.target_speed_mph if not specified).
    Used by the booster chain to set its minimum discharge target.
    """
    v_fts = mph_to_fts(target_speed_mph if target_speed_mph else cfg.target_speed_mph)
    D_ft   = (cfg.od_in - 2 * cfg.wt_in) / 12.0
    L_ft   = max(0.0, exit_mp - pig_mp) * 5280.0

    fric  = liquid_friction_loss_psi(L_ft, D_ft, v_fts, cfg.sg, cfg.viscosity_cst, cfg.roughness_ft)
    head  = static_head_psi(elevation_at(exit_mp) - elevation_at(pig_mp), cfg.sg)
    return exit_psig + fric + head
