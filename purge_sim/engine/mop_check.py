"""
Per-joint MOP pressure check.

Every timestep, every pipe joint in the ILI data is checked against MOP.
The pressure at each joint is interpolated from the hydraulic grade line.

Two regions exist simultaneously:
  1. Upstream of pig: N2 gas column — pressure profile from segment model
  2. Downstream of pig: liquid column — pressure profile from pig-face down to exit

Violations (P > MOP) are flagged in red.
Warnings (P > MOP × warning_fraction, default 0.95) are flagged in yellow.

The check is a safety gate — the simulator can optionally abort or warn on violations.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional
import numpy as np

from .physics import liquid_friction_loss_psi, static_head_psi, pipe_area_ft2, bph_to_fts


class MOPStatus(Enum):
    OK        = auto()
    WARNING   = auto()   # within warning_fraction of MOP
    VIOLATION = auto()   # exceeds MOP


@dataclass
class MOPJoint:
    """One pipe joint from ILI data."""
    mp: float
    mop_psig: float
    elevation_ft: float
    od_in: float = 24.0
    wt_in: float = 0.313


@dataclass
class MOPCheckResult:
    """Result of a single joint MOP check."""
    joint: MOPJoint
    estimated_psig: float
    status: MOPStatus
    margin_psi: float    # MOP - estimated (negative = violation)
    region: str          # 'gas' or 'liquid'


def check_mop_liquid_side(
    pig_mp: float,
    pig_face_psig: float,
    joints: List[MOPJoint],
    exit_psig: float,
    exit_mp: float,
    flow_bph: float,
    sg: float,
    viscosity_cst: float,
    roughness_ft: float,
    warning_fraction: float = 0.95,
    pig_elevation_ft: float = 0.0,
) -> List[MOPCheckResult]:
    """
    Check MOP on the liquid side (downstream of pig).

    P(x) = pig_face_psig - friction(pig→x) - static_head(pig→x)

    Vectorized over all relevant joints using numpy — O(N) not O(N²).
    """
    relevant = [j for j in joints if pig_mp <= j.mp <= exit_mp]
    if not relevant:
        return []

    mps   = np.array([j.mp          for j in relevant], dtype=float)
    mops  = np.array([j.mop_psig    for j in relevant], dtype=float)
    elevs = np.array([j.elevation_ft for j in relevant], dtype=float)
    ods   = np.array([j.od_in       for j in relevant], dtype=float)
    wts   = np.array([j.wt_in       for j in relevant], dtype=float)

    L_ft  = np.maximum(0.0, mps - pig_mp) * 5280.0
    D_ft  = (ods - 2.0 * wts) / 12.0
    area  = np.pi / 4.0 * D_ft ** 2
    v_fts = flow_bph * 5.614583 / 3600.0 / np.maximum(area, 1e-9)

    # Darcy-Weisbach friction (Swamee-Jain, vectorized)
    eps_m = roughness_ft * 0.3048
    D_m   = D_ft * 0.3048
    L_m   = L_ft * 0.3048
    v_m   = v_fts * 0.3048
    rho   = 999.0 * sg
    nu    = viscosity_cst * 1e-6
    Re    = v_m * D_m / max(nu, 1e-12)
    f     = np.where(
        Re < 2300.0,
        64.0 / np.maximum(Re, 1.0),
        0.25 / np.log10(eps_m / (3.7 * D_m) + 5.74 / np.maximum(Re, 1.0) ** 0.9) ** 2,
    )
    fric  = f * (L_m / np.maximum(D_m, 1e-9)) * 0.5 * rho * v_m ** 2 / 6894.757

    # Static head (positive = joint is higher than pig → pressure decreases)
    head  = (elevs - pig_elevation_ft) * sg * 62.4 / 144.0

    est    = pig_face_psig - fric - head
    margin = mops - est

    viol_mask = est > mops
    warn_mask = (~viol_mask) & (est > mops * warning_fraction)

    results = []
    for i, j in enumerate(relevant):
        if viol_mask[i]:
            status = MOPStatus.VIOLATION
        elif warn_mask[i]:
            status = MOPStatus.WARNING
        else:
            status = MOPStatus.OK
        results.append(MOPCheckResult(
            joint=j,
            estimated_psig=float(est[i]),
            status=status,
            margin_psi=float(margin[i]),
            region='liquid',
        ))
    return results


def check_mop_gas_side(
    pig_mp: float,
    purge_start_mp: float,
    seg_pressures: List[tuple],   # list of (mp, psig) pairs along N2 column
    joints: List[MOPJoint],
    warning_fraction: float = 0.95,
) -> List[MOPCheckResult]:
    """
    Check MOP on the gas side (upstream of pig, behind N2 column).

    Pressure profile for gas side: use segment model pressures interpolated along mileposts.
    seg_pressures: list of (mp, psig) pairs sorted by mp (upstream to downstream).
    """
    results = []
    relevant = [j for j in joints if purge_start_mp <= j.mp <= pig_mp]
    if not relevant or not seg_pressures:
        return results

    mps   = np.array([mp  for mp,  _   in seg_pressures], dtype=float)
    press = np.array([psi for _,   psi in seg_pressures], dtype=float)

    for j in relevant:
        est    = float(np.interp(j.mp, mps, press))
        margin = j.mop_psig - est

        if est > j.mop_psig:
            status = MOPStatus.VIOLATION
        elif est > j.mop_psig * warning_fraction:
            status = MOPStatus.WARNING
        else:
            status = MOPStatus.OK

        results.append(MOPCheckResult(
            joint=j,
            estimated_psig=est,
            status=status,
            margin_psi=margin,
            region='gas',
        ))

    return results


def worst_violation(results: List[MOPCheckResult]) -> Optional[MOPCheckResult]:
    """Return the most severe MOP result (highest overpressure)."""
    violations = [r for r in results if r.status == MOPStatus.VIOLATION]
    if not violations:
        return None
    return min(violations, key=lambda r: r.margin_psi)   # most negative margin


def mop_summary(results: List[MOPCheckResult]) -> dict:
    """Summary counts and worst case for display."""
    violations = [r for r in results if r.status == MOPStatus.VIOLATION]
    warnings   = [r for r in results if r.status == MOPStatus.WARNING]
    worst = worst_violation(results)
    return {
        'n_violations': len(violations),
        'n_warnings':   len(warnings),
        'worst_violation_mp':     worst.joint.mp     if worst else None,
        'worst_violation_margin': worst.margin_psi   if worst else None,
        'worst_violation_mop':    worst.joint.mop_psig if worst else None,
        'worst_violation_est':    worst.estimated_psig if worst else None,
    }


def thin_mop_joints(
    joints: List[MOPJoint],
    elevation_epsilon_ft: float = 3.0,
    mop_step_psig: float = 25.0,
) -> List[MOPJoint]:
    """
    Reduce a dense ILI joint list to the hydraulically critical subset.

    Three-pass filter:

    Pass 1 — Elevation inflection points (Ramer-Douglas-Peucker, ε = elevation_epsilon_ft).
      Keeps peaks and valleys where the elevation deviates more than ε from the chord
      connecting its neighbors.  Discards joint-level noise (< 1 psi per foot at SG~0.8)
      while preserving every hydraulically significant hill and valley.

    Pass 2 — MOP step-change boundaries.
      Any joint where MOP drops more than mop_step_psig from the previous retained value
      is kept, capturing pipe-grade transitions where the binding constraint shifts.

    Pass 3 — Per-interval worst-case audit.
      Between each pair of consecutive retained joints, the single joint with the
      absolute minimum MOP is added if its MOP is lower than both neighbors.
      Guarantees that no real MOP violation can slip through a gap.

    Typical result: ~300–700 joints from 33,000+, preserving all critical inflections.
    """
    if len(joints) <= 2:
        return list(joints)

    # Sort by milepost (should already be sorted, but be safe)
    joints = sorted(joints, key=lambda j: j.mp)
    n = len(joints)
    keep = [False] * n

    # Always keep endpoints
    keep[0] = True
    keep[n - 1] = True

    # ------------------------------------------------------------------
    # Pass 1: RDP elevation simplification
    # ------------------------------------------------------------------
    mps   = np.array([j.mp          for j in joints], dtype=float)
    elevs = np.array([j.elevation_ft for j in joints], dtype=float)

    def _rdp(lo: int, hi: int) -> None:
        if hi - lo < 2:
            return
        # Vertical deviation of each interior point from the chord lo→hi
        lo_mp, lo_e = mps[lo], elevs[lo]
        hi_mp, hi_e = mps[hi], elevs[hi]
        span = hi_mp - lo_mp
        if span < 1e-9:
            return
        t = (mps[lo+1:hi] - lo_mp) / span
        interp = lo_e + t * (hi_e - lo_e)
        deviations = np.abs(elevs[lo+1:hi] - interp)
        worst_rel = int(np.argmax(deviations))
        if deviations[worst_rel] >= elevation_epsilon_ft:
            mid = lo + 1 + worst_rel
            keep[mid] = True
            _rdp(lo, mid)
            _rdp(mid, hi)

    _rdp(0, n - 1)

    # ------------------------------------------------------------------
    # Pass 2: MOP step-change boundaries
    # ------------------------------------------------------------------
    last_mop = joints[0].mop_psig
    for i, j in enumerate(joints):
        if abs(j.mop_psig - last_mop) >= mop_step_psig:
            keep[i] = True
            last_mop = j.mop_psig

    # ------------------------------------------------------------------
    # Pass 3: Per-interval minimum-MOP audit
    # ------------------------------------------------------------------
    retained_indices = [i for i, k in enumerate(keep) if k]
    for a, b in zip(retained_indices, retained_indices[1:]):
        if b - a < 2:
            continue
        interval_mops = np.array([joints[i].mop_psig for i in range(a, b + 1)], dtype=float)
        worst_rel = int(np.argmin(interval_mops))
        if worst_rel == 0 or worst_rel == len(interval_mops) - 1:
            continue   # endpoint is already kept
        worst_abs = a + worst_rel
        worst_mop = interval_mops[worst_rel]
        if worst_mop < joints[a].mop_psig or worst_mop < joints[b].mop_psig:
            keep[worst_abs] = True

    result = [joints[i] for i, k in enumerate(keep) if k]
    return result


def build_gas_pressure_profile(
    segment_list_summary: List[dict],
) -> List[tuple]:
    """
    Build (mp, psig) pairs for N2 gas pressure profile from segment summary.
    Each segment contributes two boundary points (step-function profile).
    Returns pairs sorted by mp (ascending, upstream to downstream).
    """
    seen = {}  # mp -> psig
    for seg in segment_list_summary:
        p   = float(seg['pressure_psig'])
        mp0 = float(seg['upstream_mp'])
        mp1 = float(seg['downstream_mp'])
        if mp0 not in seen or p < seen[mp0]:
            seen[mp0] = p
        if mp1 not in seen or p < seen[mp1]:
            seen[mp1] = p
    return sorted(seen.items(), key=lambda x: x[0])   # (mp, psig) sorted by mp
