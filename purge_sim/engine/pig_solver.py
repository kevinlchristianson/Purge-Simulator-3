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
  - meter_valve throttles receipt when pig could go faster than target
    (meter valve back-pressure limits throughput at tankage)

Slack line risk:
  If the achievable speed < min_speed_mph, the liquid column is pulling faster than
  the pig can push. The meter valve closes to reduce demand.
  If the meter valve cannot fully compensate, a slack line condition is flagged.

BPCV phase:
  Before pig reaches BPCV: BPCV set point is the exit condition for the liquid column.
  After pig passes BPCV: the pig is in the N2 section; BPCV is now a constraint on
  N2 pressure — pig-face pressure must exceed BPCV set point + gas friction from pig to BPCV.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import math

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
    exit_psig: float           # effective exit condition used
    exit_mp: float
    exit_description: str

    liquid_friction_psi: float
    static_head_psi: float

    # Whether the pig is limited by meter valve (going faster than target)
    meter_valve_active: bool = False
    meter_valve_back_pressure_psi: float = 0.0

    # Slack line flag
    slack_line_risk: bool = False
    slack_line_deficit_mph: float = 0.0

    # Injection requirement (SCFM to maintain N2 column)
    injection_scfm: float = 0.0


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
) -> PigSolverResult:
    """
    Solve for pig speed given current system state.

    The pig speed is found by balancing the available drive pressure against
    the liquid-side resistance to produce a speed. The meter valve then throttles
    if that speed exceeds the target.

    Steps:
      1. Compute liquid resistance at target speed (friction + static head to exit)
      2. Check if pig_face_psig can achieve at least min_speed
      3. If pig can exceed target, meter valve activates (back-pressure = excess drive)
      4. Final speed = min(achievable_speed, target_speed)
      5. Flag slack line if achievable < min
    """
    area_ft2 = pipe_area_ft2(cfg.od_in, cfg.wt_in)
    D_ft     = (cfg.od_in - 2 * cfg.wt_in) / 12.0
    eps_ft   = cfg.roughness_ft

    exit_elevation_ft = elevation_at(exit_mp)
    L_liq_ft = max(0.0, exit_mp - pig_mp) * 5280.0

    head_psi = static_head_psi(exit_elevation_ft - pig_elevation_ft, cfg.sg)

    # Bisect on speed to find where drive = resistance
    def _resistance_at_speed(v_fts: float) -> float:
        fric = liquid_friction_loss_psi(L_liq_ft, D_ft, v_fts, cfg.sg, cfg.viscosity_cst, eps_ft)
        return exit_psig + fric + head_psi

    def _drive_available() -> float:
        # If BPCV constraint active (pig past BPCV in gas phase), drive is limited by BPCV.
        # Zero set point means BPCV is wide open — no constraint.
        if bpcv_gas_constraint_psig is not None and bpcv_gas_constraint_psig > 0:
            return min(pig_face_psig, bpcv_gas_constraint_psig)
        return pig_face_psig

    drive = _drive_available()
    v_min = mph_to_fts(cfg.min_speed_mph)
    v_max = mph_to_fts(cfg.max_speed_mph)
    v_tgt = mph_to_fts(cfg.target_speed_mph)

    # Check if drive can sustain minimum speed
    res_at_min = _resistance_at_speed(v_min)
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
            slack_line_risk=True,
            slack_line_deficit_mph=fts_to_mph(v_min - v_achievable),
            injection_scfm=scfm_from_pig_velocity(v_achievable, area_ft2,
                                                   psig_to_psia(pig_face_psig), cfg.temperature_f),
        )

    # Drive can sustain at least minimum. Find achievable speed.
    res_at_tgt = _resistance_at_speed(v_tgt)
    meter_valve_active = False
    meter_back_psi = 0.0

    if drive >= res_at_tgt:
        # Pig would go faster than target — meter valve throttles
        v_actual = v_tgt
        meter_valve_active = True
        meter_back_psi = drive - res_at_tgt
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

    # Clamp to physical maximum
    if v_actual > v_max:
        v_actual = v_max
        meter_valve_active = True
        meter_back_psi = drive - _resistance_at_speed(v_max)

    fric = liquid_friction_loss_psi(L_liq_ft, D_ft, v_actual, cfg.sg, cfg.viscosity_cst, eps_ft)
    inj_scfm = scfm_from_pig_velocity(v_actual, area_ft2,
                                       psig_to_psia(pig_face_psig), cfg.temperature_f)

    return PigSolverResult(
        pig_speed_mph=fts_to_mph(v_actual),
        pig_speed_fts=v_actual,
        drive_psig=drive,
        exit_psig=exit_psig,
        exit_mp=exit_mp,
        exit_description=exit_description,
        liquid_friction_psi=fric,
        static_head_psi=head_psi,
        meter_valve_active=meter_valve_active,
        meter_valve_back_pressure_psi=meter_back_psi,
        injection_scfm=inj_scfm,
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
