"""
Back Pressure Control Valve (BPCV) model.

The BPCV is a motor-controlled valve at St. Cesaire (MP 205.64) that protects
every downstream pipe joint from MOP exceedance. It is pig-passable.

Set point calculation:
  The set point is the maximum allowable pressure at the BPCV's downstream face,
  computed as the minimum across all downstream joints of:
    MOP_j - static_head(BPCV → j) - friction(BPCV → j at target flow)

Because flow to tankage is constant (metered), friction downstream is constant,
making the set point essentially stable for the duration of the purge.

Phases:
  1. Before pig reaches BPCV: BPCV is part of the liquid hydraulic system.
     Its set point limits how hard the upstream pump / pig drive can push.
     The exit condition for the pig's liquid push is: P_pig = set_point + liquid_friction(pig→BPCV)
  2. After pig passes BPCV: BPCV is now in the N2 gas column.
     It creates a pressure drop in the N2 column. The pig-face N2 pressure must
     remain above the BPCV set point on its downstream face.
     Effectively the BPCV becomes an intermediate constraint: N2 behind pig must
     push through BPCV to tankage at the right rate.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .physics import static_head_psi, liquid_friction_loss_psi, psia_to_psig


@dataclass
class BCPVDownstreamJoint:
    """One downstream pipe joint that the BPCV must protect."""
    mp: float
    mop_psig: float
    elevation_ft: float


@dataclass
class BPCVConfig:
    """Static configuration for the BPCV."""
    mp: float                           # milepost of the valve
    elevation_ft: float                 # elevation at valve location
    name: str = "BPCV"
    downstream_joints: List[BCPVDownstreamJoint] = field(default_factory=list)

    # Downstream flow conditions (constant during purge)
    downstream_flow_bph: float = 0.0   # target receipt rate
    downstream_od_in: float = 22.0     # pipe OD downstream of BPCV
    downstream_wt_in: float = 0.313    # pipe WT downstream of BPCV
    downstream_sg: float = 0.85        # fluid SG downstream (liquid still present there)
    downstream_viscosity_cst: float = 2.7
    downstream_roughness_ft: float = 0.00015


@dataclass
class BPCVState:
    """Runtime state of the BPCV."""
    config: BPCVConfig
    pig_has_passed: bool = False

    # Computed each timestep
    set_point_psig: float = 0.0        # target pressure at downstream face
    upstream_required_psig: float = 0.0 # what upstream must deliver to hold set point
    is_active: bool = True             # False after pig passes (pig-passable, then opens)


def compute_bpcv_set_point(
    cfg: BPCVConfig,
    target_flow_bph: float,
    fluid_sg: float,
    fluid_viscosity_cst: float,
) -> float:
    """
    Compute BPCV set point (psig at the valve downstream face).

    Walks every downstream joint. For each joint, computes the max allowable
    pressure at the BPCV downstream face that still keeps that joint below its MOP.
    Returns the minimum across all joints (most restrictive).

    Static head: (elev_BPCV - elev_j) * grad   [negative when j is lower than BPCV → adds pressure]
    Friction: integrated from BPCV to joint at target_flow_bph
    """
    if not cfg.downstream_joints:
        return 0.0

    import math
    from .physics import fts_to_bph, bph_to_fts, pipe_area_ft2

    area_ft2 = pipe_area_ft2(cfg.downstream_od_in, cfg.downstream_wt_in)
    D_ft = (cfg.downstream_od_in - 2 * cfg.downstream_wt_in) / 12.0
    eps_ft = cfg.downstream_roughness_ft

    v_fts = bph_to_fts(target_flow_bph, area_ft2) if target_flow_bph > 0 else 0.0

    min_allowable = math.inf
    for joint in cfg.downstream_joints:
        # Hydrostatic: going from BPCV elevation to joint elevation
        # Positive elev delta means BPCV is higher → liquid gains head going down → more pressure at joint
        elev_delta = cfg.elevation_ft - joint.elevation_ft
        head_psi = static_head_psi(elev_delta, fluid_sg)  # > 0 if joint is lower

        # Friction from BPCV to joint
        L_ft = max(0.0, joint.mp - cfg.mp) * 5280.0
        fric_psi = liquid_friction_loss_psi(L_ft, D_ft, v_fts, fluid_sg, fluid_viscosity_cst, eps_ft)

        # Allowable at BPCV downstream face = MOP_j - head_gain - friction
        allowable = joint.mop_psig - head_psi - fric_psi
        if allowable < min_allowable:
            min_allowable = allowable

    return float(min_allowable) if math.isfinite(min_allowable) else 0.0


def step_bpcv(
    state: BPCVState,
    pig_mp: float,
    target_flow_bph: float,
    fluid_sg: float,
    fluid_viscosity_cst: float,
) -> dict:
    """
    Update BPCV state for current pig position.

    Returns dict:
      set_point_psig     — target pressure at downstream face of BPCV
      pig_has_passed     — whether pig is now past the BPCV
      upstream_constraint_psig — if pig hasn't passed: max allowable pig-face pressure before BPCV
                                 if pig has passed: minimum N2 pressure required at pig face
                                   = set_point + gas friction from pig to BPCV (handled in pig_solver)
    """
    cfg = state.config

    if pig_mp >= cfg.mp:
        state.pig_has_passed = True

    set_pt = compute_bpcv_set_point(cfg, target_flow_bph, fluid_sg, fluid_viscosity_cst)
    state.set_point_psig = set_pt

    return {
        'set_point_psig': set_pt,
        'pig_has_passed': state.pig_has_passed,
    }


def build_bpcv_from_ili(
    bpcv_mp: float,
    bpcv_elevation_ft: float,
    ili_joints_downstream: list[dict],
    downstream_od_in: float = 22.0,
    downstream_wt_in: float = 0.313,
    downstream_sg: float = 0.85,
    downstream_viscosity_cst: float = 2.7,
    downstream_roughness_ft: float = 0.00015,
) -> BPCVConfig:
    """
    Build a BPCVConfig from ILI joint data downstream of the BPCV.

    ili_joints_downstream: list of dicts with keys 'mp', 'mop_psig', 'elevation_ft'
    """
    joints = [
        BCPVDownstreamJoint(
            mp=float(j['mp']),
            mop_psig=float(j['mop_psig']),
            elevation_ft=float(j['elevation_ft']),
        )
        for j in ili_joints_downstream
        if j.get('mop_psig') is not None
    ]
    return BPCVConfig(
        mp=bpcv_mp,
        elevation_ft=bpcv_elevation_ft,
        downstream_joints=joints,
        downstream_od_in=downstream_od_in,
        downstream_wt_in=downstream_wt_in,
        downstream_sg=downstream_sg,
        downstream_viscosity_cst=downstream_viscosity_cst,
        downstream_roughness_ft=downstream_roughness_ft,
    )
