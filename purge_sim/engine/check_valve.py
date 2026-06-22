"""
Check valve state machine.

Check valves are simple flappers — they open when upstream pressure ≥ downstream pressure
and close otherwise. No backflow is ever possible.

There are two flavors of check valves in the system:
  1. Mainline check valves — on the pipe itself (standalone or at pump stations)
  2. Booster bypass check valves — stay closed while the booster is running
     (booster discharge holds the downstream segment at higher pressure)

This module only handles mainline check valves. Booster logic is in booster.py.

Check valves are represented as boundary mileposts between adjacent segments.
When a check valve is at position MP X:
  - segments[i].downstream_mp == X
  - segments[i+1].upstream_mp == X
  - If segments[i].pressure_psig >= segments[i+1].pressure_psig → valve opens → merge
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from .segment_model import SegmentList


@dataclass
class CheckValve:
    """A passive mainline check valve at a fixed milepost."""
    mp: float
    name: str = ""
    is_pump_station: bool = False   # True if this is the pump station mainline check

    # Runtime state — updated each timestep
    is_open: bool = False
    blocked_by_booster: bool = False  # True while a booster at this MP is running


def find_segment_boundary_index(segs: SegmentList, mp: float) -> Optional[int]:
    """
    Return index i such that segs[i].downstream_mp ≈ mp (the boundary before segment i+1).
    Returns None if no such boundary exists (valve boundaries disappear after merges).
    """
    for i, s in enumerate(segs.segments[:-1]):
        if abs(s.downstream_mp - mp) < 0.001:  # 5 ft tolerance in milepost
            return i
    return None


def evaluate_check_valves(
    segs: SegmentList,
    valves: List[CheckValve],
    booster_active_mps: set[float],
) -> int:
    """
    Evaluate all check valves and perform cascading merges where pressure allows.

    A valve opens (segments merge) when:
      - upstream segment P ≥ downstream segment P, AND
      - no active booster is holding the downstream segment at elevated pressure

    After a merge, the combined segment may now equalize with the next downstream
    segment — cascade is handled by SegmentList.cascade_merges().

    Returns the number of merges that occurred this call.
    """
    total_merges = 0

    # Mark which valves are blocked by active boosters
    for valve in valves:
        valve.blocked_by_booster = valve.mp in booster_active_mps

    # Scan from upstream to downstream; after each merge re-scan from that point
    changed = True
    while changed:
        changed = False
        for valve in sorted(valves, key=lambda v: v.mp):
            if valve.blocked_by_booster:
                valve.is_open = False
                continue
            i = find_segment_boundary_index(segs, valve.mp)
            if i is None:
                # This valve's boundary was absorbed by a prior merge — it's open
                valve.is_open = True
                continue
            seg_up = segs[i]
            seg_dn = segs[i + 1]
            if seg_up.pressure_psig >= seg_dn.pressure_psig:
                valve.is_open = True
                merged = segs.merge_if_equal(i)
                if merged:
                    total_merges += 1
                    changed = True
                    break  # restart scan after any merge
            else:
                valve.is_open = False

    return total_merges


def insert_check_valve_boundary(
    segs: SegmentList,
    valve: CheckValve,
) -> bool:
    """
    Ensure a check valve boundary exists in the segment list at valve.mp.
    Splits the containing segment if the boundary doesn't exist yet.
    Call this when a check valve location first becomes relevant
    (e.g., pig has passed it and the gas column now spans across it).

    Returns True if a split was performed, False if boundary already existed.
    """
    for s in segs.segments:
        if abs(s.downstream_mp - valve.mp) < 0.001:
            return False  # already a boundary
        if abs(s.upstream_mp - valve.mp) < 0.001:
            return False  # also already a boundary

    # Check if valve is interior to any segment
    for s in segs.segments:
        if s.upstream_mp < valve.mp < s.downstream_mp:
            segs.split_at(valve.mp)
            return True

    return False
