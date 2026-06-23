"""
Gas booster station logic.

Boosters are operator-controlled compressors at pump station locations.
They draw N2 from the upstream purged pipe (upstream segment), compress it,
and discharge into the downstream segment (driving the pig at higher pressure).

Key physics (v29 bug is fixed here):
  - Suction = upstream segment pressure — booster runs only while upstream P ≥ suction_min_psig
  - Discharge = downstream segment receives SCF transferred from upstream segment
  - The upstream segment loses SCF and the downstream segment gains SCF each timestep
  - The mainline check valve at this location is HELD CLOSED while the booster is running
    (discharge pressure always exceeds suction pressure)
  - Once upstream P drops below suction_min_psig, booster shuts off automatically
    (it cannot compressor against its own discharge if suction is too low)

Boosters are NOT automatic — they are operator-initiated. The simulator models:
  1. A "requested" activation (operator turns it on when pig passes)
  2. Automatic de-activation when suction pressure falls below minimum

Spread logistics:
  - Each physical spread (equipment set) occupies one location at a time
  - Spread must mobilize (travel + rig-up) to move to a new location
  - Mobilization time is user-supplied (hours); during mobilization the spread is unavailable
  - Spreads move forward only (can't go back upstream of pig)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import math

from .segment_model import SegmentList, N2Segment
from .physics import scfm_from_pig_velocity, psia_to_psig, psig_to_psia


@dataclass
class BoosterConfig:
    """Static configuration for one booster station."""
    mp: float                      # milepost of the station
    name: str = ""
    discharge_psig: float = 1200.0 # maximum discharge pressure the booster can achieve
    suction_min_psig: float = 50.0 # minimum upstream pressure to keep running
    max_flow_scfm: float = 15_000.0 # maximum throughput (compressor capacity, SCFM)


@dataclass
class BoosterState:
    """Runtime state of one booster station."""
    config: BoosterConfig

    # Operator intent — set by simulator when pig passes station (or pre-positioned)
    requested: bool = False

    # Actual running state
    running: bool = False

    # Segment indices in SegmentList (resolved each step after possible merges)
    upstream_seg_idx: Optional[int] = None
    downstream_seg_idx: Optional[int] = None

    # Diagnostics from last timestep
    suction_psig: float = 0.0
    discharge_psig: float = 0.0
    flow_scfm: float = 0.0
    shutoff_reason: str = ""

    @property
    def mp(self) -> float:
        return self.config.mp


@dataclass
class BoosterSpread:
    """
    A physical equipment spread (compressor trailer set).
    Can be deployed to one station at a time.
    """
    spread_id: int
    current_mp: Optional[float] = None          # None = in transit
    destination_mp: Optional[float] = None      # where it's heading
    mobilize_start_hr: Optional[float] = None   # sim time when mobilization began
    mobilize_duration_hr: float = 0.0           # total mob time for current move
    available_at_hr: float = 0.0                # sim time when it arrives and is rigged up

    @property
    def in_transit(self) -> bool:
        return self.destination_mp is not None and self.current_mp != self.destination_mp

    def is_available_at(self, sim_time_hr: float) -> bool:
        return sim_time_hr >= self.available_at_hr and not self.in_transit

    def mobilize_to(self, dest_mp: float, sim_time_hr: float, mob_time_hr: float) -> None:
        """Start mobilization to a new station. Must be forward-only (dest_mp > current_mp)."""
        if self.current_mp is not None and dest_mp < self.current_mp:
            raise ValueError(
                f"Spread {self.spread_id}: cannot move backward from MP {self.current_mp:.2f} "
                f"to MP {dest_mp:.2f}"
            )
        self.destination_mp = dest_mp
        self.mobilize_start_hr = sim_time_hr
        self.mobilize_duration_hr = mob_time_hr
        self.available_at_hr = sim_time_hr + mob_time_hr

    def arrive(self, sim_time_hr: float) -> None:
        """Called when the spread has reached destination and is rigged up."""
        self.current_mp = self.destination_mp
        self.destination_mp = None
        self.available_at_hr = sim_time_hr


def find_segment_at_boundary(segs: SegmentList, mp: float) -> tuple[Optional[int], Optional[int]]:
    """
    Find the indices of the upstream and downstream segments at a booster milepost.
    Returns (upstream_idx, downstream_idx) or (None, None) if not found.
    A booster boundary must be between two segments (the boundary between them).
    """
    for i in range(len(segs) - 1):
        if abs(segs[i].downstream_mp - mp) < 0.001:
            return i, i + 1
    return None, None


def step_booster(
    state: BoosterState,
    segs: SegmentList,
    dt_hr: float,
    pig_face_pressure_psig: float,
    target_discharge_psig: Optional[float] = None,
    allow_fresh: bool = False,
) -> dict:
    """
    Advance one booster station by dt_hr hours.
    Transfers SCF from upstream segment to downstream segment.

    Returns a dict with:
      running, suction_psig, discharge_psig, flow_scfm, scf_transferred, shutoff_reason
    """
    result = {
        'running': False,
        'suction_psig': 0.0,
        'discharge_psig': 0.0,
        'flow_scfm': 0.0,
        'scf_transferred': 0.0,       # recompressed upstream N2 (free — already in the line)
        'scf_fresh_injected': 0.0,    # fresh N2 from the station's own source (totalize this)
        'shutoff_reason': '',
        'flow_limited': False,  # True = compressor pegged at max SCFM (add a compressor)
    }

    if not state.requested:
        state.running = False
        result['shutoff_reason'] = 'not_requested'
        return result

    up_idx, dn_idx = find_segment_at_boundary(segs, state.mp)
    if up_idx is None:
        # Boundary has been merged away — booster MP is now interior to one segment.
        # This should not happen while booster is active (it prevents the merge), but
        # guard defensively.
        state.running = False
        result['shutoff_reason'] = 'boundary_merged'
        return result

    state.upstream_seg_idx = up_idx
    state.downstream_seg_idx = dn_idx

    seg_up = segs[up_idx]
    seg_dn = segs[dn_idx]

    suction_psig = seg_up.pressure_psig
    result['suction_psig'] = suction_psig

    # Recompression (the compressor) can only run with suction >= suction_min — it cannot
    # pull a near-vacuum and pump against its own discharge. Fresh N2 injection, however,
    # comes from the station's own outside N2 source plumbed in at the discharge, so it
    # needs no suction. A booster therefore stays useful even after the upstream store is
    # drawn down to its floor: it just shifts from recompressing to injecting fresh.
    can_recompress = suction_psig >= state.config.suction_min_psig

    # Effective discharge target.
    # If a target_discharge_psig cap is supplied (set by the simulator for the pig-adjacent
    # booster when pig is at target speed), cap discharge there. This prevents the booster
    # from driving pig face above the minimum needed pressure, which would waste N2
    # (N2 consumption scales linearly with pig face pressure via mass conservation).
    # In drive-limited mode (pig below target speed), no cap is passed so the booster
    # runs at rated discharge to recover pig speed as fast as possible.
    if target_discharge_psig is not None:
        effective_discharge_psig = min(
            state.config.discharge_psig,
            max(target_discharge_psig, suction_psig + 5.0),  # never below suction + headroom
        )
    else:
        effective_discharge_psig = state.config.discharge_psig

    max_scf_per_step = state.config.max_flow_scfm * dt_hr * 60.0
    # Downstream demand: SCF needed to bring the pig-side segment up to the discharge target.
    scf_headroom_dn = max(0.0, seg_dn.scf_at_pressure(effective_discharge_psig) - seg_dn.scf)

    # --- 1) Recompress as much upstream N2 as is available above the suction floor ---
    if can_recompress:
        scf_available = seg_up.scf_at_pressure(state.config.suction_min_psig)
        scf_can_send  = max(0.0, seg_up.scf - scf_available)
        scf_recompress = min(scf_can_send, max_scf_per_step, scf_headroom_dn)
    else:
        scf_recompress = 0.0
    actual_recompress = seg_up.remove_scf(scf_recompress) if scf_recompress > 0 else 0.0
    if actual_recompress > 0:
        seg_dn.add_scf(actual_recompress)

    # --- 2) Fresh N2 makeup (LAST RESORT): cover the remaining discharge demand from the
    #         station's own N2 source, up to remaining compressor capacity. Fresh nitrogen is
    #         expensive (it adds to total N2), so it is only allowed at the pig-adjacent
    #         (front) booster — the one actually driving the pig face. Upstream boosters in
    #         the chain recompress-only: they relay existing gas forward but never manufacture
    #         fresh gas to over-fill an intermediate bay. Reported separately for totalizing. ---
    if allow_fresh:
        remaining_cap      = max(0.0, max_scf_per_step - actual_recompress)
        remaining_headroom = max(0.0, seg_dn.scf_at_pressure(effective_discharge_psig) - seg_dn.scf)
        scf_fresh = min(remaining_cap, remaining_headroom)
    else:
        scf_fresh = 0.0
    if scf_fresh > 0:
        seg_dn.add_scf(scf_fresh)

    total_added = actual_recompress + scf_fresh
    if total_added <= 0.0:
        # No upstream gas to recompress and no downstream demand — booster idles.
        state.running = False
        result['shutoff_reason'] = 'suction_too_low' if not can_recompress else 'no_demand'
        _update_state(state, result)
        return result

    result['running'] = True
    result['flow_scfm'] = total_added / (dt_hr * 60.0) if dt_hr > 0 else 0.0
    result['scf_transferred'] = actual_recompress
    result['scf_fresh_injected'] = scf_fresh
    result['discharge_psig'] = seg_dn.pressure_psig  # updated after adding SCF
    result['flow_limited'] = (total_added >= max_scf_per_step * 0.99)

    state.running = True
    state.suction_psig = suction_psig
    state.discharge_psig = result['discharge_psig']
    state.flow_scfm = result['flow_scfm']
    state.shutoff_reason = ''
    return result


def _update_state(state: BoosterState, result: dict) -> None:
    state.suction_psig = result['suction_psig']
    state.discharge_psig = result['discharge_psig']
    state.flow_scfm = result['flow_scfm']
    state.shutoff_reason = result['shutoff_reason']


def active_booster_mps(booster_states: List[BoosterState]) -> set[float]:
    """Return set of mileposts where a booster is currently running."""
    return {s.mp for s in booster_states if s.running}


def activate_booster_if_pig_passed(
    state: BoosterState,
    pig_mp: float,
) -> bool:
    """
    Set booster to requested when pig has passed its station.
    Returns True if newly activated this call.
    """
    if not state.requested and pig_mp > state.mp:
        state.requested = True
        return True
    return False
