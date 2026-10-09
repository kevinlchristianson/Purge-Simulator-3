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
import math
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

    # Minimum acceptable gauge pressure at elevation peaks upstream of the BPCV.
    # The BPCV's PRIMARY job is to hold enough backpressure to keep the HGL above
    # zero (plus this margin) at the high-elevation points between the pig and the valve.
    # Typically 10 psig — enough to confirm liquid fill and avoid slack line / column separation.
    min_upstream_pressure_psig: float = 10.0


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

    from .physics import bph_to_fts, pipe_area_ft2

    area_ft2 = pipe_area_ft2(cfg.downstream_od_in, cfg.downstream_wt_in)
    D_ft = (cfg.downstream_od_in - 2.0 * cfg.downstream_wt_in) / 12.0
    eps_ft = cfg.downstream_roughness_ft

    v_fts = bph_to_fts(target_flow_bph, area_ft2) if target_flow_bph > 0 else 0.0

    # Vectorized over all downstream joints.
    # Friction is linear in L (D, v, fluid properties are constant downstream of BPCV),
    # so compute friction-per-foot once, then multiply by the L array.
    m = 0.3048
    D_m, v_m = D_ft * m, v_fts * m
    rho = 999.0 * float(fluid_sg)
    nu  = float(fluid_viscosity_cst) * 1e-6
    Re  = v_m * D_m / max(1e-12, nu)
    if Re < 2300.0:
        f = 64.0 / max(1.0, Re)
    else:
        eps_m = eps_ft * m
        f = 0.25 / (math.log10(eps_m / (3.7 * D_m) + 5.74 / Re**0.9)) ** 2
    friction_psi_per_ft = f / (D_ft * m) * 0.5 * rho * v_m**2 / 6894.757 * m

    # Build arrays once from joint list (cache-friendly sequential access)
    mps   = np.fromiter((j.mp         for j in cfg.downstream_joints), dtype=float, count=len(cfg.downstream_joints))
    mops  = np.fromiter((j.mop_psig   for j in cfg.downstream_joints), dtype=float, count=len(cfg.downstream_joints))
    elevs = np.fromiter((j.elevation_ft for j in cfg.downstream_joints), dtype=float, count=len(cfg.downstream_joints))

    L_ft_arr  = np.maximum(0.0, (mps - cfg.mp) * 5280.0)
    fric_arr  = friction_psi_per_ft * L_ft_arr
    head_arr  = (cfg.elevation_ft - elevs) * float(fluid_sg) * 62.4 / 144.0
    allowable = mops - head_arr - fric_arr

    result = float(np.min(allowable))
    return result if math.isfinite(result) else 0.0


def compute_bpcv_upstream_min_set_point(
    cfg: BPCVConfig,
    pig_mp: float,
    elevation_profile: np.ndarray,
    target_flow_bph: float,
    fluid_sg: float,
    fluid_viscosity_cst: float,
    upstream_od_in: float = 24.0,
    upstream_wt_in: float = 0.281,
    upstream_roughness_ft: float = 0.00015,
) -> float:
    """
    Minimum BPCV set point (psig) to keep pressure >= min_upstream_pressure_psig
    at every elevation peak between the pig and the BPCV.

    Derived from Bernoulli in the flow direction (pig → point j → BPCV):

        P(j) = P_BPCV + (elev_BPCV - elev_j)×SG×0.433 + friction(j→BPCV)

    Rearranging for the minimum BPCV pressure that ensures P(j) >= P_min:

        P_BPCV_min = P_min + (elev_j - elev_BPCV)×SG×0.433 - friction(j→BPCV)

    Takes the maximum over all elevation points between pig and BPCV.
    Points lower than the BPCV never constrain (term is negative → BPCV can be lower).
    """
    if elevation_profile is None or len(elevation_profile) == 0:
        return 0.0

    mask = (elevation_profile[:, 0] >= pig_mp) & (elevation_profile[:, 0] <= cfg.mp)
    if not np.any(mask):
        return 0.0

    upstream_mps  = elevation_profile[mask, 0]
    upstream_elevs = elevation_profile[mask, 1]

    # Friction factor — single value since pipe properties are approximately uniform upstream
    from .physics import pipe_area_ft2, bph_to_fts
    area_ft2 = pipe_area_ft2(upstream_od_in, upstream_wt_in)
    D_ft = (upstream_od_in - 2.0 * upstream_wt_in) / 12.0
    v_fts = bph_to_fts(target_flow_bph, area_ft2) if target_flow_bph > 0 else 0.0

    m = 0.3048
    D_m, v_m = D_ft * m, v_fts * m
    rho = 999.0 * float(fluid_sg)
    nu  = float(fluid_viscosity_cst) * 1e-6
    Re  = v_m * D_m / max(1e-12, nu)
    if Re < 2300.0:
        f = 64.0 / max(1.0, Re)
    else:
        eps_m = upstream_roughness_ft * m
        f = 0.25 / (math.log10(eps_m / (3.7 * D_m) + 5.74 / Re**0.9)) ** 2
    # friction in psi per foot of pipe
    friction_psi_per_ft = f / D_m * 0.5 * rho * v_m**2 / 6894.757 * m

    L_to_bpcv = np.maximum(0.0, (cfg.mp - upstream_mps) * 5280.0)
    fric_arr  = friction_psi_per_ft * L_to_bpcv
    head_arr  = (upstream_elevs - cfg.elevation_ft) * float(fluid_sg) * 62.4 / 144.0

    required = cfg.min_upstream_pressure_psig + head_arr - fric_arr
    return max(0.0, float(np.max(required)))


def step_bpcv(
    state: BPCVState,
    pig_mp: float,
    target_flow_bph: float,
    fluid_sg: float,
    fluid_viscosity_cst: float,
    elevation_profile: Optional[np.ndarray] = None,
    upstream_od_in: float = 24.0,
    upstream_wt_in: float = 0.281,
    upstream_roughness_ft: float = 0.00015,
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

    # Minimum set point: enough backpressure to keep the HGL above min_upstream_pressure_psig
    # at every elevation peak between the pig and the BPCV. This is the PRIMARY constraint —
    # the operator sets BPCV just high enough to prevent slack line at the worst peak.
    if elevation_profile is not None and not state.pig_has_passed:
        set_pt_min = compute_bpcv_upstream_min_set_point(
            cfg, pig_mp, elevation_profile, target_flow_bph,
            fluid_sg, fluid_viscosity_cst,
            upstream_od_in, upstream_wt_in, upstream_roughness_ft,
        )
    else:
        set_pt_min = 0.0

    # Maximum set point: the highest the BPCV can hold without violating downstream MOP.
    # This is a hard cap — the BPCV must never exceed it regardless of upstream conditions.
    set_pt_max = compute_bpcv_set_point(cfg, target_flow_bph, fluid_sg, fluid_viscosity_cst)
    if not math.isfinite(set_pt_max) or set_pt_max <= 0:
        set_pt_max = math.inf   # no downstream MOP constraint known

    set_pt = max(set_pt_min, 0.0)
    if math.isfinite(set_pt_max):
        set_pt = min(set_pt, set_pt_max)

    state.set_point_psig = set_pt

    return {
        'set_point_psig': set_pt,
        'pig_has_passed': state.pig_has_passed,
        # The valve's modulation window while it is the pig's exit: it holds set_point_psig
        # (just enough to keep the upstream peaks full) and may raise it, to slow the pig,
        # as far as its downstream MOP allows (inf when no downstream joints are known).
        'set_point_min_psig': set_pt,
        'set_point_max_psig': max(set_pt, set_pt_max),
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
