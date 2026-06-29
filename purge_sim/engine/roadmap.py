"""
Pre-run pressure roadmap (the "corridor").

Before a single timestep runs, the route's terrain and MOP profile already fix
the pressure envelope the pig must live inside at every milepost. This module
precomputes that envelope once, as two curves over pig position x:

    ceiling(x) -- the HIGHEST pig-face drive pressure allowed before some joint
                  exceeds MOP. Answers "what pressure at this high point causes
                  an MOP violation?" Built from two sub-limits:
                    * liquid side  -- every joint AHEAD of the pig sees
                      P_face + static-head - friction; on a rise the column
                      drops its full head onto downstream valleys.
                    * gas side     -- the N2 column BEHIND the pig cannot exceed
                      the MOP of the pipe it sits in. Because boosters bound the
                      pig-adjacent gas inside one inter-booster "bay", the limit
                      is the minimum MOP across that whole bay.

    floor(x)   -- the LOWEST drive pressure that still keeps the pig moving:
                  enough to lift the liquid column over the highest peak between
                  the pig and its exit. Answers "what pressure at this low point
                  causes a stall?"

Where floor(x) > ceiling(x) the location is INFEASIBLE: you cannot both keep the
pig moving and stay under MOP. That is reported before the run so the plan can be
fixed (add a booster, lower target speed, re-batch) instead of discovering it as a
mid-run violation -- or papering over it with a vent.

The controller (speed solver, injection, boosters) then simply tracks ceiling(x)
looking ahead, so pressure is drawn down BEFORE low-MOP pipe instead of built and
relieved. With the corridor respected there is nothing to vent.

Almost all of the corridor is static terrain + MOP (static head dominates friction
~10:1 here), so it is computed once and read by lookup during the loop.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import math
import numpy as np

from .physics import liquid_friction_loss_psi, mph_to_fts, clearance_candidates


@dataclass
class Roadmap:
    """Precomputed pressure corridor along the route (function of pig position)."""
    mp: np.ndarray                 # grid mileposts (sorted)
    elev_ft: np.ndarray            # elevation at each grid mp
    mop_psig: np.ndarray           # joint MOP at each grid mp

    ceiling_psig: np.ndarray       # max safe pig-face drive (min of gas/liquid)
    floor_psig: np.ndarray         # min drive to avoid stall
    ceiling_gas_psig: np.ndarray   # gas-side component (diagnostic)
    ceiling_liq_psig: np.ndarray   # liquid-side component (diagnostic)

    infeasible_mask: np.ndarray    # bool: floor > ceiling
    infeasible_spans: List[Tuple[float, float, float]] = field(default_factory=list)
    # each: (mp_lo, mp_hi, worst_deficit_psi)  deficit = floor - ceiling

    # ---- fast lookups used by the controller during the loop ----
    def ceiling_at(self, mp: float) -> float:
        """Max safe pig-face drive at pig position `mp` (linear interp, clamped)."""
        return float(np.interp(mp, self.mp, self.ceiling_psig))

    def floor_at(self, mp: float) -> float:
        """Min drive to avoid a stall at pig position `mp` (linear interp, clamped)."""
        return float(np.interp(mp, self.mp, self.floor_psig))

    def min_ceiling_ahead(self, mp: float, horizon_mi: float) -> float:
        """Lowest ceiling between `mp` and `mp + horizon_mi`.

        This is what lets the controller act PROACTIVELY: if a low-MOP bay is
        coming, the ceiling ahead is already low, so drive is drawn down in time
        rather than built and vented.
        """
        lo = int(np.searchsorted(self.mp, mp, side='left'))
        hi = int(np.searchsorted(self.mp, mp + horizon_mi, side='right'))
        if hi <= lo:
            return self.ceiling_at(mp)
        return float(np.min(self.ceiling_psig[lo:hi]))

    def capacity_projected_floor(self, mp: float, dP_dx_max: float,
                                 horizon_mi: float = 200.0) -> float:
        """Drive to aim for NOW so the pig can just reach every upcoming floor in time, given
        the equipment's max pressure-build slope ``dP_dx_max`` (psi of drive gained per mile of
        travel).

        For a future point x', reaching floor(x') in the distance (x'-mp) at build slope
        dP_dx_max requires the pig face to already be at  floor(x') - dP_dx_max*(x'-mp)  now.
        The target is the max of that over all x' ahead — the binding upcoming climb. This is
        the capacity-derived replacement for a fixed look-ahead horizon: a fast build slope
        discounts distant climbs heavily (build just-in-time); a slow one barely discounts them
        (start early). Never below the floor right here.
        """
        here = self.floor_at(mp)
        lo = int(np.searchsorted(self.mp, mp, side='left'))
        hi = int(np.searchsorted(self.mp, mp + horizon_mi, side='right'))
        if hi <= lo:
            return here
        proj = self.floor_psig[lo:hi] - max(0.0, dP_dx_max) * (self.mp[lo:hi] - mp)
        return float(max(here, float(np.max(proj))))

    def max_floor_ahead(self, mp: float, horizon_mi: float) -> float:
        """Highest floor between `mp` and `mp + horizon_mi` (worst upcoming demand).

        This lets the controller COAST safely: it only stops injecting when the gas
        already behind the pig exceeds the worst drive the next stretch will demand,
        so it never coasts down into a climb it then can't make (which would stall
        the pig). On a genuine descent the floor stays low ahead, so coasting
        continues and the climb column is drawn down.
        """
        lo = int(np.searchsorted(self.mp, mp, side='left'))
        hi = int(np.searchsorted(self.mp, mp + horizon_mi, side='right'))
        if hi <= lo:
            return self.floor_at(mp)
        return float(np.max(self.floor_psig[lo:hi]))

    @property
    def feasible(self) -> bool:
        return not bool(self.infeasible_mask.any())

    def report(self) -> str:
        """Human-readable pre-run feasibility summary."""
        lines = []
        lo_mp, hi_mp = float(self.mp[0]), float(self.mp[-1])
        lines.append(f"Pressure roadmap: {len(self.mp)} points, MP {lo_mp:.1f}-{hi_mp:.1f}")
        c, f = self.ceiling_psig, self.floor_psig
        finite_c = c[np.isfinite(c)]
        if finite_c.size:
            lines.append(f"  ceiling: min {finite_c.min():.0f}  max {finite_c.max():.0f} psig")
        lines.append(f"  floor:   min {f.min():.0f}  max {f.max():.0f} psig")
        if self.feasible:
            margin = c - f
            margin = margin[np.isfinite(margin)]
            tight = margin.min() if margin.size else float('nan')
            lines.append(f"  FEASIBLE end-to-end. Tightest corridor margin: {tight:.0f} psi")
        else:
            n = len(self.infeasible_spans)
            lines.append(f"  INFEASIBLE at {n} span(s) (floor exceeds ceiling -- "
                         f"cannot move the pig without an MOP violation):")
            for mp_lo, mp_hi, deficit in self.infeasible_spans:
                lines.append(f"    MP {mp_lo:6.1f}-{mp_hi:6.1f}: short by {deficit:.0f} psi")
            lines.append("  Fix: add booster coverage, lower target speed, or re-batch "
                         "so the column need not be lifted as high at once.")
        return "\n".join(lines)


def build_roadmap(
    mj_mp: Optional[np.ndarray],
    mj_mop: Optional[np.ndarray],
    mj_elev: Optional[np.ndarray],
    cfg,
    elevation_at: Callable[[float], float],
) -> Optional[Roadmap]:
    """Build the pressure corridor from the (already thinned, sorted) MOP joints.

    Args:
        mj_mp, mj_mop, mj_elev: sorted joint arrays (the same ones the loop uses).
        cfg:          SimConfig.
        elevation_at: elevation interpolator, elevation_at(mp) -> ft.

    Returns None if there are no MOP joints (no corridor to compute).
    """
    if mj_mp is None or len(mj_mp) == 0:
        return None

    mp   = np.asarray(mj_mp, dtype=float)
    mop  = np.asarray(mj_mop, dtype=float)
    elev = np.asarray(mj_elev, dtype=float)
    n    = len(mp)

    grad   = cfg.fluid_sg * 62.4 / 144.0          # psi per ft of static head
    v_min  = mph_to_fts(cfg.min_speed_mph)
    start  = cfg.purge_start_mp
    end    = cfg.purge_end_mp

    # Bay boundaries: booster stations partition the gas column. The pig-adjacent
    # N2 stays inside one bay until the pig reaches the next station, so the gas
    # there can never legally exceed the minimum MOP across the whole bay.
    boundaries = sorted({start, end} | {b.mp for b in cfg.booster_configs})
    bnd = np.asarray(boundaries, dtype=float)

    # Pump stations ahead act as the pig's exit (suction) while active.
    sta_mp  = np.asarray(sorted(s.mp for s in cfg.pump_stations), dtype=float)
    sta_suc = {s.mp: s.suction_psig for s in cfg.pump_stations}

    maop      = cfg.maop_psig if math.isfinite(cfg.maop_psig) else math.inf
    max_drive = cfg.max_drive_psig if math.isfinite(cfg.max_drive_psig) else math.inf
    drive_cap = min(maop, max_drive)

    ceil_gas = np.full(n, math.inf)
    ceil_liq = np.full(n, math.inf)
    floor    = np.zeros(n)

    # representative pipe diameter for friction (varies slowly; use per-x value)
    for i in range(n):
        x   = mp[i]
        E_x = elev[i]
        od, wt = cfg.pipe_geometry.od_wt_at(x)
        D_ft = (od - 2.0 * wt) / 12.0
        # friction per foot at min speed (Darcy dp scales linearly with length)
        fric_per_ft = liquid_friction_loss_psi(
            1.0, D_ft, v_min, cfg.fluid_sg, cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft
        )

        # --- gas-side ceiling: min MOP across the bay containing x ---
        bi = int(np.searchsorted(bnd, x, side='right'))
        bay_lo = bnd[bi - 1] if bi - 1 >= 0 else start
        bay_hi = bnd[bi]     if bi < len(bnd) else end
        glo = int(np.searchsorted(mp, bay_lo, side='left'))
        ghi = int(np.searchsorted(mp, bay_hi, side='right'))
        if ghi > glo:
            ceil_gas[i] = float(np.min(mop[glo:ghi]))

        # --- liquid-side ceiling: every joint ahead of the pig ---
        # pressure at joint j (pig at x) = P_face + (E_x - E_j)*grad - friction(x->j)
        # require <= MOP_j  ->  P_face <= MOP_j + friction + (E_j - E_x)*grad
        j0 = i + 1
        if j0 < n:
            L_ft = (mp[j0:] - x) * 5280.0
            fric = fric_per_ft * L_ft
            head = (elev[j0:] - E_x) * grad
            ceil_liq[i] = float(np.min(mop[j0:] + fric + head))

        # --- floor: peak-aware min drive to keep the column full to the exit ---
        # Same shared model as the solver / shutdown check: deliver exit_psig at the exit AND
        # clear every intervening hill (+ slack margin). exit = nearest pump station ahead
        # (its suction), else tankage end (the peak-clearing term covers Tug/Sutton either way).
        ahead_sta = sta_mp[sta_mp > x] if sta_mp.size else np.empty(0)
        if ahead_sta.size:
            exit_mp   = float(ahead_sta[0])
            exit_psig = float(sta_suc[exit_mp])
            exit_elev = elevation_at(exit_mp)
        else:
            exit_mp   = end
            exit_psig = cfg.exit_pressure_run_psig
            exit_elev = elevation_at(end)
        _b, _l = clearance_candidates(x, E_x, exit_mp, exit_psig, exit_elev,
                                      mp, elev, cfg.fluid_sg)
        floor[i] = float(np.max(_b + fric_per_ft * _l))

    # Drive ceiling = GAS-side MOP only (the N2 column behind the pig vs. local pipe MOP).
    # The liquid-side limit (the crude column ahead pressing its head onto descending joints)
    # is moot for a gas displacement: the pump stations / BPCV regulate the liquid side as it
    # is received, so it does not cap the pig drive. ceil_liq is kept as a diagnostic only.
    # (Including it wrongly cratered the ceiling on the Tug descent and stalled the pig there;
    # past LS the gas-side MOP is ~930, far above the ~400 psi needed to crest Tug.)
    ceiling = np.minimum(ceil_gas, drive_cap)

    # --- infeasible spans (floor exceeds ceiling) ---
    infeasible = floor > ceiling
    spans: List[Tuple[float, float, float]] = []
    i = 0
    while i < n:
        if infeasible[i]:
            j = i
            worst = 0.0
            while j < n and infeasible[j]:
                worst = max(worst, float(floor[j] - ceiling[j]))
                j += 1
            spans.append((float(mp[i]), float(mp[j - 1]), worst))
            i = j
        else:
            i += 1

    return Roadmap(
        mp=mp, elev_ft=elev, mop_psig=mop,
        ceiling_psig=ceiling, floor_psig=floor,
        ceiling_gas_psig=ceil_gas, ceiling_liq_psig=ceil_liq,
        infeasible_mask=infeasible, infeasible_spans=spans,
    )
