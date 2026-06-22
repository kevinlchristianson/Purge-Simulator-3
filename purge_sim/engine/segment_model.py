"""
N2 segment model.

The gas column behind the pig is divided into segments by check valves and booster stations.
Each segment is an isolated volume of nitrogen with a tracked SCF inventory.
Pressure is derived from SCF + volume + temperature via Peng-Robinson EOS.

Segment boundaries are always in milepost order (upstream → downstream).
Segment 0 is farthest upstream (injection side).
Last segment is immediately behind the pig.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import math

from .physics import (
    pipe_volume_ft3,
    pressure_psia_from_scf,
    scf_from_pressure_volume,
    psia_to_psig,
    psig_to_psia,
)
from .constants import ATM_PSI, N2_TEMP_F_DEFAULT


@dataclass
class PipeGeometry:
    """Variable-OD pipeline geometry — lookup pipe properties at a given milepost."""

    # Each entry: (start_mp, end_mp, od_in, wt_in)
    segments: List[tuple] = field(default_factory=list)

    def od_wt_at(self, mp: float) -> tuple[float, float]:
        """Return (od_in, wt_in) at a given milepost. Uses last segment if beyond end."""
        for (s, e, od, wt) in self.segments:
            if s <= mp <= e:
                return od, wt
        if self.segments:
            return self.segments[-1][2], self.segments[-1][3]
        raise ValueError(f"No pipe geometry defined for MP {mp:.3f}")

    def volume_ft3_between(self, mp_upstream: float, mp_downstream: float) -> float:
        """Internal volume (ft³) between two mileposts, accounting for variable OD/WT."""
        if mp_downstream <= mp_upstream:
            return 0.0

        total = 0.0
        # Walk through pipe geometry segments, computing volume in each sub-span
        checkpoints = set()
        for (s, e, _, _) in self.segments:
            if s > mp_upstream:
                checkpoints.add(min(s, mp_downstream))
            if e < mp_downstream:
                checkpoints.add(max(e, mp_upstream))
        checkpoints.add(mp_upstream)
        checkpoints.add(mp_downstream)
        sorted_cps = sorted(checkpoints)

        for i in range(len(sorted_cps) - 1):
            a = sorted_cps[i]
            b = sorted_cps[i + 1]
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            od, wt = self.od_wt_at(mid)
            total += pipe_volume_ft3(od, wt, b - a)
        return total


@dataclass
class N2Segment:
    """
    One isolated N2 volume between two check valve / boundary locations.

    Inventory is tracked as SCF (standard cubic feet at 14.7 psia, 60°F).
    Pressure is derived on demand — not stored — so it always reflects current SCF.
    """
    upstream_mp: float    # milepost of upstream boundary (check valve, injection point, or booster suction)
    downstream_mp: float  # milepost of downstream boundary (check valve, booster discharge, or pig)
    scf: float            # N2 inventory (SCF)
    temperature_f: float = N2_TEMP_F_DEFAULT
    geometry: Optional[PipeGeometry] = field(default=None, repr=False)

    @property
    def length_mi(self) -> float:
        return max(0.0, self.downstream_mp - self.upstream_mp)

    @property
    def volume_ft3(self) -> float:
        if self.geometry is not None:
            return self.geometry.volume_ft3_between(self.upstream_mp, self.downstream_mp)
        raise RuntimeError("N2Segment has no PipeGeometry attached — cannot compute volume")

    @property
    def pressure_psia(self) -> float:
        V = self.volume_ft3
        if V <= 0:
            return ATM_PSI
        return pressure_psia_from_scf(self.scf, V, self.temperature_f)

    @property
    def pressure_psig(self) -> float:
        return psia_to_psig(self.pressure_psia)

    def scf_at_pressure(self, target_psig: float) -> float:
        """SCF this segment would hold at target_psig (same volume, same temperature)."""
        return scf_from_pressure_volume(psig_to_psia(target_psig), self.volume_ft3, self.temperature_f)

    def add_scf(self, delta_scf: float) -> None:
        self.scf = max(0.0, self.scf + delta_scf)

    def remove_scf(self, delta_scf: float) -> float:
        """Remove up to delta_scf from this segment. Returns actual amount removed."""
        removable = min(delta_scf, self.scf)
        self.scf = max(0.0, self.scf - removable)
        return removable


def merge_segments(upstream: N2Segment, downstream: N2Segment) -> N2Segment:
    """
    Merge two adjacent segments into one (when check valve opens).
    Combined segment spans from upstream.upstream_mp to downstream.downstream_mp.
    SCF inventories simply add — pressure redistributes over the combined volume.
    """
    merged = N2Segment(
        upstream_mp=upstream.upstream_mp,
        downstream_mp=downstream.downstream_mp,
        scf=upstream.scf + downstream.scf,
        temperature_f=upstream.temperature_f,
        geometry=upstream.geometry,
    )
    return merged


class SegmentList:
    """
    Ordered list of N2Segment objects (upstream → downstream).
    Manages insertions, merges, and the pig-face segment boundary.
    """

    def __init__(self, geometry: PipeGeometry):
        self._geometry = geometry
        self._segments: List[N2Segment] = []

    @property
    def segments(self) -> List[N2Segment]:
        return self._segments

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx):
        return self._segments[idx]

    def initialize_from_pig_at(self, pig_mp: float, purge_start_mp: float,
                                initial_pressure_psig: float) -> None:
        """
        Initialize with a single segment spanning purge_start_mp → pig_mp
        at the given initial pressure.
        """
        self._segments.clear()
        seg = N2Segment(
            upstream_mp=purge_start_mp,
            downstream_mp=pig_mp,
            scf=0.0,
            geometry=self._geometry,
        )
        # Set SCF to match initial pressure
        V = seg.volume_ft3
        seg.scf = scf_from_pressure_volume(psig_to_psia(initial_pressure_psig), V, seg.temperature_f)
        self._segments.append(seg)

    def advance_pig(self, new_pig_mp: float) -> float:
        """
        Extend the last (pig-face) segment to new_pig_mp.
        The new volume is the incremental volume vacated by the pig.
        Returns the incremental volume added (ft³) — caller must add injection SCF.
        """
        if not self._segments:
            raise RuntimeError("SegmentList is empty — cannot advance pig")
        last = self._segments[-1]
        old_mp = last.downstream_mp
        if new_pig_mp <= old_mp:
            return 0.0
        delta_vol = self._geometry.volume_ft3_between(old_mp, new_pig_mp)
        last.downstream_mp = new_pig_mp
        return delta_vol

    def split_at(self, mp: float) -> None:
        """
        Split whichever segment contains mp at that milepost.
        Used when a booster station activates and becomes a boundary.
        SCF is split proportionally by volume.
        """
        for i, seg in enumerate(self._segments):
            if seg.upstream_mp < mp < seg.downstream_mp:
                vol_total = seg.volume_ft3
                vol_left  = self._geometry.volume_ft3_between(seg.upstream_mp, mp)
                frac_left = vol_left / max(1e-12, vol_total)
                left = N2Segment(
                    upstream_mp=seg.upstream_mp,
                    downstream_mp=mp,
                    scf=seg.scf * frac_left,
                    temperature_f=seg.temperature_f,
                    geometry=self._geometry,
                )
                right = N2Segment(
                    upstream_mp=mp,
                    downstream_mp=seg.downstream_mp,
                    scf=seg.scf * (1.0 - frac_left),
                    temperature_f=seg.temperature_f,
                    geometry=self._geometry,
                )
                self._segments[i:i+1] = [left, right]
                return
        raise ValueError(f"No segment contains MP {mp:.3f}")

    def merge_if_equal(self, i: int) -> bool:
        """
        Merge segments[i] and segments[i+1] if upstream[i] P ≥ downstream[i+1] P.
        Returns True if a merge occurred (caller should re-scan after merges).
        """
        segs = self._segments
        if i < 0 or i + 1 >= len(segs):
            return False
        if segs[i].pressure_psig >= segs[i + 1].pressure_psig:
            merged = merge_segments(segs[i], segs[i + 1])
            segs[i:i+2] = [merged]
            return True
        return False

    def cascade_merges(self, start_i: int = 0) -> int:
        """
        Starting at index start_i, merge all adjacent segments where upstream P ≥ downstream P.
        Cascades until no more merges occur.
        Returns number of merges performed.
        """
        total_merges = 0
        i = max(0, start_i)
        while i < len(self._segments) - 1:
            if self.merge_if_equal(i):
                total_merges += 1
                # Don't advance i — the merged segment may now also equalize with the next
            else:
                i += 1
        return total_merges

    def total_scf(self) -> float:
        return sum(s.scf for s in self._segments)

    def pig_face_pressure_psig(self) -> float:
        """Pressure immediately behind the pig (last segment)."""
        if not self._segments:
            return 0.0
        return self._segments[-1].pressure_psig

    def injection_segment(self) -> N2Segment:
        """The first (most upstream) segment — where injection enters."""
        if not self._segments:
            raise RuntimeError("SegmentList is empty")
        return self._segments[0]

    def summary(self) -> list[dict]:
        """Return a list of dicts describing each segment (for logging/UI)."""
        out = []
        for i, s in enumerate(self._segments):
            out.append({
                'index': i,
                'upstream_mp': s.upstream_mp,
                'downstream_mp': s.downstream_mp,
                'length_mi': s.length_mi,
                'volume_ft3': s.volume_ft3,
                'scf': s.scf,
                'pressure_psig': s.pressure_psig,
            })
        return out
