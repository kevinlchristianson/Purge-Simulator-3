"""
Pump station shutdown logic.

Pump stations are liquid-side infrastructure. The pig cannot pass through a running
pump (it stalls between suction and discharge of the running unit).

Shutdown rules:
  - A station shuts down when it is NO LONGER HYDRAULICALLY NECESSARY — i.e., the pig
    has enough drive pressure to reach the next active station (or endpoint) without
    the boost this station provides.
  - 1 mile from a running station is a HARD SAFETY LIMIT: a station MUST be shut
    down before the pig gets within 1 mile, regardless of hydraulics.
  - Operators can (and should) shut stations down earlier when they become unnecessary.
  - In the high-drive strategy (pig near MOP), stations A and B may become unnecessary
    long before the pig reaches them and can be shut down far in advance.

After shutdown, the pig's effective exit condition jumps from this station's suction
to the NEXT active station's suction (or BPCV / tankage if none remain). This
dramatically increases required drive pressure, so the hydraulic grade line check
is critical before authorizing any shutdown.

Grade line check (simplified):
  Available drive = pig_face_psig - liquid_friction(pig→next_active) - static_head(pig→next_active)
  Required drive  = next_active_suction_psig
  If available_drive >= required_drive → station is hydraulically unnecessary → can shut down

Station states:
  RUNNING       — providing suction lift to liquid ahead of pig
  SHUT_DOWN     — already offline; pig must push past it on its own
  BYPASSED      — pig is past this station; no longer a factor
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from .physics import (liquid_friction_loss_psi, static_head_psi, pipe_area_ft2, bph_to_fts,
                      required_drive_to_clear)


class StationStatus(Enum):
    RUNNING   = auto()
    SHUT_DOWN = auto()
    BYPASSED  = auto()   # pig has passed this station


@dataclass
class PumpStationConfig:
    """Static configuration for one pump station."""
    mp: float
    name: str = ""
    suction_psig: float = 0.0    # minimum suction pressure (what the pump needs at inlet)
    enabled: bool = True         # False = this station never runs (scenario can disable it)
    # Highest suction the station can run at while it is the pig's exit and modulates
    # (slows its pumps) to hold the pig at max speed. None = the MOP at the station.
    max_suction_psig: Optional[float] = None
    # Note: stations don't have a fixed 'discharge' in the pig context —
    # after shutdown, the pig just has to push liquid past the station entirely.


@dataclass
class PumpStationState:
    """Runtime state of one pump station."""
    config: PumpStationConfig
    status: StationStatus = StationStatus.RUNNING
    shutdown_time_hr: Optional[float] = None
    shutdown_reason: str = ""    # 'hydraulically_unnecessary', 'one_mile_limit', 'manual'

    @property
    def mp(self) -> float:
        return self.config.mp

    @property
    def is_active(self) -> bool:
        return self.status == StationStatus.RUNNING

    @property
    def is_bypassed(self) -> bool:
        return self.status == StationStatus.BYPASSED


def next_active_station(
    stations: List[PumpStationState],
    pig_mp: float,
) -> Optional[PumpStationState]:
    """
    Return the first RUNNING station downstream of the pig.
    Returns None if all remaining stations are shut down (pig pushes to BPCV / tankage).
    """
    ahead = [s for s in stations if s.mp > pig_mp and s.is_active]
    if not ahead:
        return None
    return min(ahead, key=lambda s: s.mp)


def check_hydraulic_necessity(
    pig_mp: float,
    pig_face_psig: float,
    station: PumpStationState,
    next_downstream: Optional[PumpStationState],
    endpoint_psig: float,
    endpoint_mp: float,
    flow_bph: float,
    od_in: float,
    wt_in: float,
    sg: float,
    viscosity_cst: float,
    roughness_ft: float,
    elevation_at: callable,
    terrain_mp=None,
    terrain_elev=None,
    slack_margin_psi: float = 25.0,
) -> tuple[bool, float, float]:
    """
    Check whether `station` is still hydraulically necessary.

    The test: if this station shuts down, can the pig's N2 drive pressure still push
    liquid to the NEXT active station (or the actual endpoint if none remain) — keeping
    the column full over EVERY intervening hill, not just at the target (peak-aware)?
    A pump cannot hold a column upstream of itself, so any hill between the pig and the
    next anchor is the pig's drive to hold; the station is only unnecessary once the pig
    face can clear all of them (plus a slack margin).

    Args:
        pig_mp:          current pig milepost
        pig_face_psig:   pig drive pressure (behind-pig N2 pressure)
        station:         the station we're evaluating for shutdown
        next_downstream: the next RUNNING station after `station` (or None)
        endpoint_psig:   required pressure at the endpoint (BPCV set point or tankage)
        endpoint_mp:     milepost of the endpoint (BPCV or tankage — used when no next station)
        flow_bph:        target flow rate for friction calculation
        od_in, wt_in:    pipe geometry
        sg, viscosity_cst, roughness_ft: fluid properties
        elevation_at:    callable(mp) -> elevation in ft

    Returns:
        (unnecessary, available_drive_psig, required_drive_psig)
    """
    if next_downstream is not None:
        target_mp     = next_downstream.mp
        required_psig = next_downstream.config.suction_psig
    else:
        target_mp     = endpoint_mp   # actual endpoint, NOT zero
        required_psig = endpoint_psig

    area_ft2 = pipe_area_ft2(od_in, wt_in)
    D_ft     = (od_in - 2 * wt_in) / 12.0
    v_fts    = bph_to_fts(flow_bph, area_ft2)

    # Peak-aware: drive needed to deliver required_psig at the target AND clear every hill
    # between the pig and the target (+ slack margin). The station is unnecessary only when
    # the pig face already exceeds that.
    required = required_drive_to_clear(
        pig_mp, elevation_at(pig_mp), target_mp, required_psig, elevation_at(target_mp),
        terrain_mp, terrain_elev, D_ft, v_fts, sg, viscosity_cst, roughness_ft,
        slack_margin_psi,
    )
    unnecessary = pig_face_psig >= required
    return unnecessary, pig_face_psig, required


def evaluate_shutdowns(
    stations: List[PumpStationState],
    pig_mp: float,
    pig_face_psig: float,
    ultimate_endpoint_psig: float,
    ultimate_endpoint_mp: float,
    flow_bph: float,
    od_in: float,
    wt_in: float,
    sg: float,
    viscosity_cst: float,
    roughness_ft: float,
    elevation_at: callable,
    sim_time_hr: float,
    hard_limit_miles: float = 1.0,
    terrain_mp=None,
    terrain_elev=None,
    slack_margin_psi: float = 25.0,
) -> List[dict]:
    """
    Evaluate all running stations for shutdown eligibility this timestep.

    ultimate_endpoint_psig / _mp: the final destination beyond all pump stations —
        the BPCV set point (if pig hasn't passed it) or tankage inlet. Used when a
        station has no downstream station to check against.
    """
    """
    Evaluate all running stations for shutdown eligibility this timestep.

    A station shuts down if:
      1. It is hydraulically unnecessary (pig can reach next station without it), OR
      2. Pig is within hard_limit_miles of it (safety hard limit)

    Stations that the pig has already passed are marked BYPASSED.

    Returns list of dicts describing any shutdowns that occurred.
    """
    events = []

    # Mark bypassed stations
    for s in stations:
        if s.is_active and s.mp < pig_mp:
            s.status = StationStatus.BYPASSED

    # Evaluate still-running stations ahead of pig
    running_ahead = sorted(
        [s for s in stations if s.is_active and s.mp > pig_mp],
        key=lambda s: s.mp,
    )

    for i, station in enumerate(running_ahead):
        distance_to_pig = station.mp - pig_mp

        # Hard limit check — must shut down
        if distance_to_pig <= hard_limit_miles:
            station.status = StationStatus.SHUT_DOWN
            station.shutdown_time_hr = sim_time_hr
            station.shutdown_reason = 'one_mile_limit'
            events.append({
                'mp': station.mp,
                'name': station.config.name,
                'reason': 'one_mile_limit',
                'distance_to_pig_mi': distance_to_pig,
                'sim_time_hr': sim_time_hr,
            })
            continue

        # Find what this station's downstream target would be after shutdown
        # (next running station in the list, skipping i)
        remaining = [s for s in running_ahead[i+1:] if s.is_active]
        next_dn = remaining[0] if remaining else None

        unnecessary, avail, req = check_hydraulic_necessity(
            pig_mp, pig_face_psig, station, next_dn,
            ultimate_endpoint_psig, ultimate_endpoint_mp, flow_bph,
            od_in, wt_in, sg, viscosity_cst, roughness_ft,
            elevation_at, terrain_mp, terrain_elev, slack_margin_psi,
        )

        if unnecessary:
            station.status = StationStatus.SHUT_DOWN
            station.shutdown_time_hr = sim_time_hr
            station.shutdown_reason = 'hydraulically_unnecessary'
            events.append({
                'mp': station.mp,
                'name': station.config.name,
                'reason': 'hydraulically_unnecessary',
                'available_drive_psig': avail,
                'required_drive_psig': req,
                'distance_to_pig_mi': distance_to_pig,
                'sim_time_hr': sim_time_hr,
            })

    return events


def effective_exit_window(
    stations: List[PumpStationState],
    pig_mp: float,
    bpcv_set_point_psig: Optional[float],
    bpcv_mp: Optional[float],
    tankage_psig: float,
    tankage_mp: float,
    bpcv_max_psig: Optional[float] = None,
    tankage_max_psig: Optional[float] = None,
) -> tuple[float, Optional[float], float, str]:
    """
    Return the exit for the pig's current liquid push: the nearest thing downstream of it,
    as (min_psig, max_psig, exit_mp, description).

    Only three things can be immediately downstream of the pig, and only the nearest one
    is in its hydraulics:
      1. Next RUNNING pump station ahead of pig -> its suction_psig (max: max_suction_psig)
      2. BPCV set point, while the pig hasn't passed the BPCV (max: bpcv_max_psig)
      3. Tankage inlet pressure (max: tankage_max_psig)

    min_psig is the inlet pressure the device holds at rest; max_psig is the most it can
    raise it to when it modulates to hold the pig at max speed (None = not known here, the
    caller falls back to the MOP at the exit).
    """
    nxt = next_active_station(stations, pig_mp)
    if nxt is not None:
        return (nxt.config.suction_psig, nxt.config.max_suction_psig, nxt.mp,
                f"Station {nxt.config.name} suction")

    if bpcv_mp is not None and bpcv_set_point_psig is not None and pig_mp < bpcv_mp:
        return bpcv_set_point_psig, bpcv_max_psig, bpcv_mp, "BPCV set point"

    return tankage_psig, tankage_max_psig, tankage_mp, "Tankage inlet"


def effective_exit_condition(
    stations: List[PumpStationState],
    pig_mp: float,
    bpcv_set_point_psig: Optional[float],
    bpcv_mp: Optional[float],
    tankage_psig: float,
    tankage_mp: float,
) -> tuple[float, float, str]:
    """(exit_min_psig, exit_mp, description): effective_exit_window without the maximum."""
    lo, _hi, mp, desc = effective_exit_window(stations, pig_mp, bpcv_set_point_psig, bpcv_mp,
                                              tankage_psig, tankage_mp)
    return lo, mp, desc
