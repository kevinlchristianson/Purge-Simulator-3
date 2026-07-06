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

from .physics import liquid_friction_loss_psi, mph_to_fts, fts_to_bph, pipe_area_ft2


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


def build_liquid_hgl(grid_mp, pig_mp, pig_face_psig, pumps, bpcv_mp, bpcv_psig,
                     end_mp, tank_psig, elev_at, fric_head, grad, margin_ft: float = 20.0,
                     clamp_to_ground: bool = True):
    """Liquid HGL (ft of head) ahead of the pig as PIECEWISE-STRAIGHT friction gradients —
    each fluid column has its OWN grade line, exactly like the real operating profile:

        pig -> first running pump   : the PIG'S drive gradient (a pump cannot lift its own
                                      upstream column — only the pig holds it; this can fall
                                      BELOW ground = slack / column separation)
        pump -> next pump / BPCV     : one straight gradient (off the pump's discharge)
        below the BPCV -> tankage    : one straight gradient to the line exit

    A pump adds a square step UP (suction -> discharge); the BPCV adds a step DOWN. Each
    segment's slope is just the friction gradient (steeper = higher flow). `pumps` is a sorted
    list of (mp, suction_psig, discharge_psig), all upstream of the BPCV.

    clamp_to_ground=True floors the line at ground+margin (a cosmetic "full line" view);
    False plots EXACTLY what the model produces, so slack shows as the HGL dipping below
    ground.
    """
    pumps = sorted(pumps)
    out = []
    for x in grid_mp:
        if bpcv_mp is not None and x > bpcv_mp:            # below the valve: tankage gradient
            h = elev_at(end_mp) + tank_psig / grad + fric_head(x, end_mp)
        else:
            up = None                                     # the pump whose discharge feeds x
            for p in pumps:
                if p[0] <= x:
                    up = p
            if up is not None:                            # off a pump's discharge (falls downstream)
                h = elev_at(up[0]) + up[2] / grad - fric_head(up[0], x)
            else:                                         # upstream of the first running pump (or
                # none ahead): the PIG drives the column. A pump cannot hold the column upstream
                # of its own suction, and the BPCV only sets the exit drop — so this stretch is
                # the pig's drive gradient, which goes slack where the drive can't hold a hill.
                h = elev_at(pig_mp) + pig_face_psig / grad - fric_head(pig_mp, x)
        out.append((x, max(h, elev_at(x) + margin_ft) if clamp_to_ground else h))
    return out


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

    # --- Liquid side (ahead of the pig): the REAL operating HGL ---
    # The liquid ahead of the pig is held up by the RUNNING pump stations and the BPCV /
    # tankage ahead — NOT by the pig alone. A pig-only forward walk (pig_face - friction +
    # descent head) falsely droops below ground over a big climb and reads as slack even
    # though the downstream pumps hold the line full. Use the same pump/BPCV-aware model as
    # the profile chart (build_liquid_hgl) so the MOP-check sheet and the animation agree.
    ahead = np.where(~is_gas)[0]
    if ahead.size:
        od0, wt0 = cfg.pipe_geometry.od_wt_at(pig_mp)
        D_ft = (od0 - 2.0 * wt0) / 12.0
        area0 = pipe_area_ft2(od0, wt0)
        flow_bph = fts_to_bph(v_fts, area0)

        def _fric_head(a, b):
            L = abs(b - a) * 5280.0
            return liquid_friction_loss_psi(L, D_ft, v_fts, cfg.fluid_sg,
                                            cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft) / grad

        tank_psig = getattr(cfg, 'exit_pressure_run_psig', float(step.exit_psig))
        bpcv = getattr(cfg, 'bpcv', None)
        bpcv_mp = bpcv.mp if (bpcv is not None and pig_mp < bpcv.mp) else None
        bpcv_sp = tank_psig
        if bpcv_mp is not None:
            try:
                from .bpcv import (compute_bpcv_upstream_min_set_point,
                                   compute_bpcv_set_point)
                smin = compute_bpcv_upstream_min_set_point(
                    bpcv, pig_mp, elev_prof, flow_bph, cfg.fluid_sg, cfg.fluid_viscosity_cst)
                smax = compute_bpcv_set_point(
                    bpcv, flow_bph, cfg.fluid_sg, cfg.fluid_viscosity_cst)
                bpcv_sp = max(0.0, smin)
                if smax and np.isfinite(smax) and smax > 0:
                    bpcv_sp = min(bpcv_sp, smax)
            except Exception:
                bpcv_sp = tank_psig

        sps = getattr(step, 'station_pressures', None) or []
        pumps = [(float(s['mp']), float(s.get('suction_psig', 0.0)), float(s['discharge_psig']))
                 for s in sps if s.get('status') == 'running' and s['mp'] > pig_mp]

        ahead_mp = [float(mp[i]) for i in ahead]
        hgl_pts = build_liquid_hgl(ahead_mp, pig_mp, float(step.pig_face_psig), pumps,
                                   bpcv_mp, bpcv_sp, end, tank_psig,
                                   elevation_at, _fric_head, grad, clamp_to_ground=False)
        for i, (_m, h) in zip(ahead, hgl_pts):
            pressure[i] = (h - elevation_ft[i]) * grad

    total_head_ft = elevation_ft + pressure / grad

    return HGLProfile(
        mp=mp, elevation_ft=elevation_ft, pressure_psig=pressure,
        total_head_ft=total_head_ft, is_gas=is_gas,
        pig_mp=pig_mp, exit_mp=float(step.exit_mp), exit_psig=float(step.exit_psig),
    )
