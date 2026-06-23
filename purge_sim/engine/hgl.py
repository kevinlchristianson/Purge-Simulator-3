"""
Hydraulic Grade Line (HGL) — full-pipeline pressure profile at a single timestep.

This is the one spatially-resolved view of the whole route's state: the N2 gas
pressure BEHIND the pig and the liquid pressure AHEAD of it, on one milepost axis,
plus the total head (elevation + pressure head) used for the dual-axis profile chart.

It is the primary tool for both optimization and debugging:
  * see where the column is over-built (gas pressure higher than the floor needs),
  * see whether the liquid HGL ahead of the pig is near slack (column-separation),
  * see whether the BPCV is holding the minimum backpressure or clamping excess.

Pressure model (consistent with pig_solver / roadmap):
  GAS (mp <= pig):   step-function from the segment pressures (gas friction ~ 0).
  LIQUID (mp > pig): walk forward from the pig face,
        P(x+dx) = P(x) - friction(dx) + (elev_x - elev_{x+dx}) * grad
     i.e. pressure drops with friction and rises as the line descends. At the pig's
     exit (next pump suction / BPCV set point / tankage) this equals the exit pressure
     the solver balanced against.

Total head (ft) = elevation_ft + pressure_psig / grad,  grad = sg * 62.4 / 144.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .physics import liquid_friction_loss_psi, mph_to_fts


@dataclass
class HGLProfile:
    """Full-route pressure + total-head profile at one timestep."""
    mp: np.ndarray              # milepost grid (ascending)
    elevation_ft: np.ndarray    # ground elevation at each mp
    pressure_psig: np.ndarray   # gas (behind pig) + liquid (ahead) pressure
    total_head_ft: np.ndarray   # elevation_ft + pressure head
    is_gas: np.ndarray          # bool mask: True where behind the pig (N2 side)

    pig_mp: float = 0.0
    exit_mp: float = 0.0
    exit_psig: float = 0.0

    def slack_margin_psig(self) -> float:
        """Lowest liquid pressure ahead of the pig (negative => slack/column separation)."""
        liq = self.pressure_psig[~self.is_gas]
        return float(np.min(liq)) if liq.size else float("nan")


def _gas_lookup(gas_profile: List[tuple], mp: float) -> float:
    """Step-function: pressure of the segment whose span contains `mp`."""
    p = gas_profile[0][1] if gas_profile else 0.0
    for gmp, gp in gas_profile:          # ascending by mp
        if gmp <= mp:
            p = gp
        else:
            break
    return p


def compute_hgl(step, cfg, elevation_at=None) -> HGLProfile:
    """Build the full-pipeline HGL for one SimStep.

    Args:
        step: a SimStep (uses pig_mp, pig_face_psig, pig_speed_mph, exit_mp,
              exit_psig, gas_pressure_profile).
        cfg:  SimConfig (pipe geometry, fluid, elevation profile, purge span).
        elevation_at: optional elevation interpolator; built from cfg if omitted.

    Returns an HGLProfile across [purge_start_mp, purge_end_mp].
    """
    elev_prof = np.asarray(cfg.elevation_profile, dtype=float)
    if elevation_at is None:
        ep_mp, ep_el = elev_prof[:, 0], elev_prof[:, 1]
        elevation_at = lambda x: float(np.interp(x, ep_mp, ep_el))

    grad = cfg.fluid_sg * 62.4 / 144.0                   # psi per ft of head
    v_fts = mph_to_fts(max(step.pig_speed_mph, cfg.min_speed_mph))

    start, end = cfg.purge_start_mp, cfg.purge_end_mp
    pig_mp = float(step.pig_mp)

    # Milepost grid: elevation points in range + pig + exit, unique & sorted.
    grid = set(float(m) for m in elev_prof[:, 0] if start <= m <= end)
    grid.update([start, end, pig_mp, float(step.exit_mp)])
    mp = np.array(sorted(grid), dtype=float)

    elevation_ft = np.array([elevation_at(x) for x in mp], dtype=float)
    pressure = np.zeros_like(mp)
    is_gas = mp <= pig_mp + 1e-6

    # --- Gas side (behind the pig): step-function of segment pressures ---
    gas_profile = step.gas_pressure_profile or []
    for i in np.where(is_gas)[0]:
        pressure[i] = _gas_lookup(gas_profile, mp[i])

    # --- Liquid side (ahead of the pig): walk forward from the pig face ---
    ahead = np.where(~is_gas)[0]
    p_prev = float(step.pig_face_psig)
    mp_prev = pig_mp
    el_prev = elevation_at(pig_mp)
    for i in ahead:
        od, wt = cfg.pipe_geometry.od_wt_at(0.5 * (mp_prev + mp[i]))
        d_ft = (od - 2.0 * wt) / 12.0
        l_ft = max(0.0, (mp[i] - mp_prev)) * 5280.0
        fric = liquid_friction_loss_psi(l_ft, d_ft, v_fts, cfg.fluid_sg,
                                        cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft)
        head = (el_prev - elevation_ft[i]) * grad        # descending => +pressure
        p_prev = p_prev - fric + head
        pressure[i] = p_prev
        mp_prev, el_prev = mp[i], elevation_ft[i]

    total_head_ft = elevation_ft + pressure / grad

    return HGLProfile(
        mp=mp, elevation_ft=elevation_ft, pressure_psig=pressure,
        total_head_ft=total_head_ft, is_gas=is_gas,
        pig_mp=pig_mp, exit_mp=float(step.exit_mp), exit_psig=float(step.exit_psig),
    )
