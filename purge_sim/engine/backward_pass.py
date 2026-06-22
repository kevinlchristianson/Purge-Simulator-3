"""
Backward-pass optimizer.

Computes the theoretical minimum N2 required for a successful purge by working
backwards from the endpoint. This gives the floor — the least N2 that could
possibly work — before the forward optimizer finds a spread deployment plan
to meet that floor.

Concept:
  Starting from the pig at the endpoint with N2 at minimum acceptable pressure,
  walk backwards through each check valve / booster boundary. At each boundary,
  the upstream segment must hold enough SCF to:
    1. Sustain N2 pressure above booster suction floor (or pig-face floor if no booster)
    2. Supply whatever the downstream segment needed via booster transfers

  Natural equalization (check valve merge when pressures equalize) reduces the
  required injection: if upstream and downstream pressures equalize on their own,
  no booster energy is needed for that transfer.

Output:
  - N2Floor per segment at each pig position
  - Total minimum SCF injection profile over time
  - Which booster stations are load-bearing (needed for the minimum)

This is a pre-simulation analysis, not a real-time calculation.
The actual simulation may inject more than the minimum (safety margin, rate limits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

from .physics import (
    pipe_volume_ft3, scf_from_pressure_volume, pressure_psia_from_scf,
    psig_to_psia, psia_to_psig,
)
from .segment_model import PipeGeometry
from .booster import BoosterConfig
from .check_valve import CheckValve


@dataclass
class BackwardPassConfig:
    """Inputs for the backward pass."""
    purge_start_mp: float
    purge_end_mp: float
    pipe_geometry: PipeGeometry
    n2_temperature_f: float = 45.0
    endpoint_pressure_psig: float = 10.0    # minimum acceptable pig-face pressure at finish

    # Infrastructure
    booster_configs: List[BoosterConfig] = field(default_factory=list)
    check_valves: List[CheckValve] = field(default_factory=list)

    # Pig travel parameters
    pig_step_mi: float = 1.0   # resolution of the backward pass (miles)


@dataclass
class SegmentFloor:
    """Minimum N2 floor for one segment at a specific pig position."""
    upstream_mp: float
    downstream_mp: float
    min_pressure_psig: float    # minimum pressure needed in this segment
    min_scf: float              # SCF required to hold min_pressure_psig
    volume_ft3: float
    reason: str                 # 'booster_suction', 'pig_face', 'equalization'


@dataclass
class BackwardPassResult:
    """Result of the backward pass analysis."""
    total_min_scf: float          # theoretical minimum SCF injection
    equalization_credit_scf: float # SCF saved by natural equalization (no booster needed)
    floors_by_pig_mp: Dict[float, List[SegmentFloor]]  # pig_mp -> list of floors
    critical_boosters: List[float]  # booster mileposts that are load-bearing
    min_injection_profile: List[Tuple[float, float]]   # [(pig_mp, scf_needed_at_this_step)]


def run_backward_pass(cfg: BackwardPassConfig) -> BackwardPassResult:
    """
    Compute minimum N2 floor by working backwards from the endpoint.

    Algorithm:
    1. Discretize pig positions from end to start in cfg.pig_step_mi increments
    2. At each pig position, the gas column has segments defined by booster/CV boundaries
       that the pig has already passed
    3. Working from the pig face backward:
       a. Pig-face segment needs pressure_psig >= pig_face_min
       b. Each segment feeding a booster needs pressure_psig >= booster.suction_min_psig
       c. If two segments can equalize (no booster between them), use combined volume → lower floor
    4. Sum the SCF floors over all segments → minimum total SCF at that pig position
    5. The injection requirement at each pig step = SCF needed at next position - what's already in the system
    """
    geom = cfg.pipe_geometry
    T_f  = cfg.n2_temperature_f

    # Sort boosters and check valves by milepost
    boosters  = sorted(cfg.booster_configs, key=lambda b: b.mp)
    all_cvs   = sorted(cfg.check_valves, key=lambda v: v.mp)
    booster_mps = {b.mp for b in boosters}

    # Pig positions from end to start
    pig_mps = np.arange(cfg.purge_end_mp, cfg.purge_start_mp, -cfg.pig_step_mi)
    if pig_mps[-1] != cfg.purge_start_mp:
        pig_mps = np.append(pig_mps, cfg.purge_start_mp)

    floors_by_mp: Dict[float, List[SegmentFloor]] = {}
    min_injection_profile: List[Tuple[float, float]] = []
    prev_total_scf = 0.0
    cumulative_injection = 0.0
    equalization_credit = 0.0
    critical_booster_mps: set[float] = set()

    for pig_mp in reversed(pig_mps):   # forward pass for injection profile
        # Build segment boundaries at this pig position
        # Boundaries = [purge_start] + [all booster/CV mps that pig has already passed] + [pig_mp]
        passed_boundaries = sorted(
            [mp for mp in (booster_mps | {cv.mp for cv in all_cvs})
             if cfg.purge_start_mp < mp < pig_mp]
        )
        boundaries = [cfg.purge_start_mp] + passed_boundaries + [pig_mp]

        segments_at_mp: List[SegmentFloor] = []
        total_scf_needed = 0.0

        # Work backwards from pig face
        for k in range(len(boundaries) - 1, 0, -1):
            seg_up_mp = boundaries[k - 1]
            seg_dn_mp = boundaries[k]
            vol = geom.volume_ft3_between(seg_up_mp, seg_dn_mp)

            # Is there a booster at seg_dn_mp?
            booster_at_boundary = next(
                (b for b in boosters if abs(b.mp - seg_dn_mp) < 0.001), None
            )

            if k == len(boundaries) - 1:
                # Pig-face segment: must hold endpoint pressure
                min_p = cfg.endpoint_pressure_psig
                reason = 'pig_face'
            elif booster_at_boundary is not None:
                # Booster at downstream boundary: this segment must be at suction_min
                min_p  = booster_at_boundary.suction_min_psig
                reason = 'booster_suction'
                critical_booster_mps.add(booster_at_boundary.mp)
            else:
                # Check valve only — try equalization with downstream segment
                # If downstream pressure <= this segment's natural equilibrium pressure,
                # they merge → combined volume → lower pressure per SCF
                if segments_at_mp:
                    dn_floor = segments_at_mp[-1]
                    combined_vol = vol + dn_floor.volume_ft3
                    combined_scf = dn_floor.min_scf  # we can share the downstream SCF
                    # Pressure from combined volume
                    combined_p = psia_to_psig(
                        pressure_psia_from_scf(combined_scf, combined_vol, T_f)
                    )
                    if combined_p >= dn_floor.min_pressure_psig:
                        # Equalization works — upstream segment contributes for free
                        equalization_credit += scf_from_pressure_volume(
                            psig_to_psia(max(0.0, combined_p)), vol, T_f
                        )
                        segments_at_mp.append(SegmentFloor(
                            upstream_mp=seg_up_mp,
                            downstream_mp=seg_dn_mp,
                            min_pressure_psig=combined_p,
                            min_scf=0.0,   # no additional injection needed
                            volume_ft3=vol,
                            reason='equalization',
                        ))
                        continue
                min_p  = cfg.endpoint_pressure_psig  # default floor
                reason = 'check_valve'

            min_scf = scf_from_pressure_volume(psig_to_psia(max(0.0, min_p)), vol, T_f)
            total_scf_needed += min_scf

            segments_at_mp.append(SegmentFloor(
                upstream_mp=seg_up_mp,
                downstream_mp=seg_dn_mp,
                min_pressure_psig=min_p,
                min_scf=min_scf,
                volume_ft3=vol,
                reason=reason,
            ))

        floors_by_mp[float(pig_mp)] = segments_at_mp

        # Injection at this step = how much MORE SCF is needed than what was in system
        incremental = max(0.0, total_scf_needed - prev_total_scf)
        cumulative_injection += incremental
        min_injection_profile.append((float(pig_mp), incremental))
        prev_total_scf = total_scf_needed

    return BackwardPassResult(
        total_min_scf=cumulative_injection,
        equalization_credit_scf=equalization_credit,
        floors_by_pig_mp=floors_by_mp,
        critical_boosters=sorted(critical_booster_mps),
        min_injection_profile=min_injection_profile,
    )


def min_n2_for_segment(
    volume_ft3: float,
    min_pressure_psig: float,
    temperature_f: float,
) -> float:
    """Convenience: SCF needed to hold a given volume at min_pressure_psig."""
    return scf_from_pressure_volume(psig_to_psia(max(0.0, min_pressure_psig)), volume_ft3, temperature_f)
