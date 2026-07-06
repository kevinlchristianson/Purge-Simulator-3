"""
Matplotlib chart functions for the results panel.
All functions accept a Figure and populate it; they do not create or show figures.
"""

from __future__ import annotations

from typing import List
import numpy as np

from ..engine.simulator import SimResults
from ..engine.backward_pass import BackwardPassResult
from ..engine.mop_check import MOPStatus


def plot_pig_speed(fig, times_hr: List[float], mps: List[float], speeds_mph: List[float],
                   meter_valve_flags: List[bool], slack_flags: List[bool]):
    """Two-panel: speed vs time (top) and speed vs milepost (bottom)."""
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    times = np.array(times_hr)
    mps_a = np.array(mps)
    spd   = np.array(speeds_mph)
    mv    = np.array(meter_valve_flags)
    sl    = np.array(slack_flags)

    ax1.plot(times, spd, 'b-', linewidth=1.2, label='Pig speed')
    if mv.any():
        ax1.scatter(times[mv], spd[mv], color='orange', s=8, zorder=5,
                    label='Meter valve active')
    if sl.any():
        ax1.scatter(times[sl], spd[sl], color='red', marker='x', s=20, zorder=6,
                    label='Slack line risk')
    ax1.set_xlabel("Time (hr)")
    ax1.set_ylabel("Speed (mph)")
    ax1.set_title("Pig Speed vs Time")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(mps_a, spd, 'b-', linewidth=1.2)
    if mv.any():
        ax2.scatter(mps_a[mv], spd[mv], color='orange', s=8, zorder=5)
    if sl.any():
        ax2.scatter(mps_a[sl], spd[sl], color='red', marker='x', s=20, zorder=6)
    ax2.set_xlabel("Milepost")
    ax2.set_ylabel("Speed (mph)")
    ax2.set_title("Pig Speed vs Milepost")
    ax2.grid(alpha=0.3)

    fig.tight_layout()


def plot_pipeline_profile(fig, step, cfg, roadmap=None, annotation: str = "",
                          title: str = None, start_label: str = None, end_label: str = None,
                          xlim=None):
    """Dual-axis pipeline profile at a SINGLE timestep.

    Left Y: Elevation / Total Head (ft).  Right Y: Pressure (psig).
    X: Milepost (pump stations + configured start/end labels ticked).

    Parameters
    ----------
    title : str or None
        Chart title; auto-generated from pipe size and distance if None.
    start_label, end_label : str or None
        Labels for the purge-start and purge-end on the X axis.
        Derived from pump stations if None; falls back to "MP X.X".
    xlim : (float, float) or None
        Override X-axis range. None → (purge_start-3, purge_end+3).
        Set to (ep_mp[0]-3, ep_mp[-1]+3) for a full-pipeline context view.
    """
    from ..engine.physics import (pipe_area_ft2, mph_to_fts, fts_to_bph, bph_to_fts,
                                   liquid_friction_loss_psi)

    grad = cfg.fluid_sg * 62.4 / 144.0
    elev = np.asarray(cfg.elevation_profile, dtype=float)
    ep_mp, ep_el = elev[:, 0], elev[:, 1]
    def elev_at(x): return float(np.interp(x, ep_mp, ep_el))
    start, end = cfg.purge_start_mp, cfg.purge_end_mp
    pig   = float(step.pig_mp)
    pface = float(step.pig_face_psig)
    sps   = step.station_pressures or []
    inj_scfm = max(0.0, getattr(step, 'injection_scfm', 0.0))

    ax  = fig.add_subplot(111)
    axp = ax.twinx()

    od, wt = cfg.pipe_geometry.od_wt_at(pig)
    D_ft = (od - 2.0 * wt) / 12.0
    area = pipe_area_ft2(od, wt)
    flow_bph = fts_to_bph(mph_to_fts(step.pig_speed_mph), area)
    v_fts = (bph_to_fts(fts_to_bph(mph_to_fts(max(step.pig_speed_mph, cfg.min_speed_mph)), area),
                        area) if area > 0 else 0.0)

    # ============ Y-AXIS AUTO-SCALING ============================================
    # Head axis: cover the elevation range + full MOP headroom.
    # Use only the purge section for scaling so zoomed views aren't skewed by far tails.
    purge_mask = (ep_mp >= start) & (ep_mp <= end)
    el_pu = ep_el[purge_mask] if purge_mask.any() else ep_el
    min_el = float(np.min(el_pu))
    max_el = float(np.max(el_pu))

    mop_max = float(np.max(roadmap.mop_psig)) if roadmap is not None else max(700.0, pface * 1.2)
    p_top = max(1000.0, mop_max * 1.05)

    # y_max = highest point on ground + full MOP above it
    y_max = max_el + p_top / grad * 1.05
    y_max = max(y_max, 2000.0)
    y_min = max(0.0, min_el - 300.0)

    # Near-sea-level pipelines: give extra vertical room so the elevation profile
    # doesn't crowd the bottom of the chart and HGL reads clearly above it.
    if min_el < 500.0:
        y_min = 0.0
        y_max = max(y_max, max_el * 3.0)

    # Pressure axis: maintain grad-consistent scale (p_range = grad * head_range)
    # so that slope-of-HGL = hydraulic gradient visually.
    p_bot = p_top - grad * (y_max - y_min)
    p_bot = max(p_bot, -p_top)

    # ============ RIGHT AXIS (psig): N2 behind pig — flat per segment ============
    gas_lab = False
    for seg in (step.segments or []):
        if seg['upstream_mp'] >= pig:
            continue
        a, b, p = seg['upstream_mp'], min(seg['downstream_mp'], pig), seg['pressure_psig']
        axp.plot([a, b], [p, p], color='deepskyblue', lw=1.8, zorder=4,
                 label=None if gas_lab else 'N2 Gas Pressure')
        gas_lab = True

    # AHEAD of the pig: liquid HGL
    from ..engine.hgl import build_liquid_hgl
    from ..engine.bpcv import compute_bpcv_upstream_min_set_point, compute_bpcv_set_point

    bpcv = getattr(cfg, 'bpcv', None)
    bpcv_mp = bpcv.mp if (bpcv is not None and pig < bpcv.mp) else None
    mt_psig = getattr(cfg, 'exit_pressure_run_psig', step.exit_psig)

    def _fric_head(a, b):
        L = abs(b - a) * 5280.0
        return liquid_friction_loss_psi(L, D_ft, v_fts, cfg.fluid_sg,
                                        cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft) / grad

    bpcv_sp = mt_psig
    if bpcv_mp is not None:
        smin = compute_bpcv_upstream_min_set_point(
            bpcv, pig, elev, flow_bph, cfg.fluid_sg, cfg.fluid_viscosity_cst)
        smax = compute_bpcv_set_point(bpcv, flow_bph, cfg.fluid_sg, cfg.fluid_viscosity_cst)
        bpcv_sp = max(0.0, smin)
        if smax and np.isfinite(smax) and smax > 0:
            bpcv_sp = min(bpcv_sp, smax)

    pumps = [(s['mp'], s['suction_psig'], s['discharge_psig']) for s in sps
             if s.get('status') == 'running' and s['mp'] > pig]
    grid = sorted(set([m for m in ep_mp if pig < m <= end]
                      + [p[0] for p in pumps] + ([bpcv_mp] if bpcv_mp else []) + [end]))
    hgl_pts = build_liquid_hgl(grid, pig, pface, pumps, bpcv_mp, bpcv_sp, end, mt_psig,
                               elev_at, _fric_head, grad, clamp_to_ground=False)

    ox = [pig] + [m for m, _ in hgl_pts]
    oy = [pface] + [(h - elev_at(m)) * grad for m, h in hgl_pts]
    for pmp, suc, dis in pumps:
        ox += [pmp, pmp]; oy += [suc, dis]
    order = np.argsort(ox)
    ox = list(np.asarray(ox)[order]); oy = list(np.asarray(oy)[order])
    axp.plot(ox, oy, color='deepskyblue', lw=1.8, zorder=4,
             label=None if gas_lab else 'N2 Gas Pressure')

    if roadmap is not None:
        axp.plot(np.asarray(roadmap.mp, float), np.asarray(roadmap.mop_psig, float),
                 color='darkorange', lw=1.1, zorder=3, label='MOP (psig)')

    for s in sps:
        axp.plot([s['mp']], [s['discharge_psig']], 'o', color='royalblue', ms=5, zorder=6)
        axp.annotate(f"{s['discharge_psig']:.0f}", (s['mp'], s['discharge_psig']),
                     color='royalblue', fontsize=10, fontweight='bold', ha='center',
                     xytext=(0, 7), textcoords='offset points')

    axp.plot([pig], [pface], 'o', color='limegreen', ms=9, markeredgecolor='darkgreen', zorder=10)
    axp.annotate(f"{pface:.0f}", (pig, pface), color='darkgreen', fontsize=12,
                 fontweight='bold', ha='left', xytext=(7, 0), textcoords='offset points')

    # ============ FLOATING RATE MARKERS ==========================================
    # Both markers scale from 0 (bottom of their axis) to 80% up at max rate.
    # This gives a visual "thermometer" effect independent of axis scale.

    # Injection rate (green triangle, left margin on head axis)
    _max_inj = max(1.0, getattr(cfg, 'max_injection_scfm', 5000.0))
    _inj_y = min(inj_scfm / _max_inj, 1.0) * 0.80 * (y_max - y_min) + y_min
    ax.plot([start + 1.0], [_inj_y], '^', color='green', ms=11, zorder=11,
            clip_on=False)
    ax.annotate(f"{inj_scfm:,.0f} SCFM", (start + 1.0, _inj_y), color='darkgreen',
                fontsize=9, fontweight='bold', ha='left', va='center',
                xytext=(10, 0), textcoords='offset points')

    # Flow rate (magenta dot, right margin on pressure axis)
    _max_flow_bph = fts_to_bph(mph_to_fts(max(getattr(cfg, 'max_speed_mph', 5.0) or 5.0, 1.0)),
                                area)
    _flow_p = p_bot + min(flow_bph / max(1.0, _max_flow_bph), 1.0) * 0.80 * (p_top - p_bot)
    axp.plot([end - 1.0], [_flow_p], 'o', color='magenta', ms=11, zorder=11, clip_on=False)
    axp.annotate(f"{flow_bph:,.0f} BPH", (end - 1.0, _flow_p), color='magenta',
                 fontsize=9, fontweight='bold', ha='right', va='center',
                 xytext=(-10, 0), textcoords='offset points')

    # ============ LEFT AXIS (ft): elevation + MOP head + HGL ====================
    ax.plot(ep_mp, ep_el, color='navy', lw=1.3, zorder=2, label='Ground Elevation')

    if roadmap is not None:
        mp_m = np.asarray(roadmap.mp, float)
        ax.plot(mp_m, np.interp(mp_m, ep_mp, ep_el) + np.asarray(roadmap.mop_psig, float) / grad,
                color='red', lw=1.0, zorder=2, label='Max Operating Head')

    hgl_ft = [elev_at(x) + p / grad for x, p in zip(ox, oy)]
    ax.plot(ox, hgl_ft, color='royalblue', lw=1.9, zorder=5, label='Operating Head (HGL)')

    _gnd = np.array([elev_at(x) for x in ox]); _hgl = np.array(hgl_ft, dtype=float)
    if np.any(_hgl < _gnd - 0.5):
        ax.fill_between(ox, _hgl, _gnd, where=(_hgl < _gnd), interpolate=True,
                        color='red', alpha=0.35, zorder=4, label='SLACK')

    ax.axvline(pig, color='limegreen', lw=2.4, zorder=6)
    ax.plot([pig], [elev_at(pig) + pface / grad], marker='s', color='orange', ms=6,
            markeredgecolor='k', zorder=8, label='PIG')

    used = False
    for bs in (step.booster_states or []):
        if bs.get('running'):
            bmp = bs['mp']
            bh = elev_at(bmp) + bs.get('discharge_psig', pface) / grad
            ax.plot([bmp], [bh], marker='s', color='gold', ms=11,
                    markeredgecolor='saddlebrown', zorder=7,
                    label=None if used else 'Booster / Compressor')
            scfm = bs.get('flow_scfm', 0.0) or 0.0
            ax.annotate(f"{scfm:,.0f} SCFM", (bmp, bh), color='saddlebrown', fontsize=8,
                        fontweight='bold', ha='center', xytext=(0, 11),
                        textcoords='offset points', zorder=9)
            used = True

    if bpcv is not None:
        ax.axvline(bpcv.mp, color='crimson', lw=1.0, ls=':', zorder=3)
        ax.annotate("BPCV", (bpcv.mp, y_min + (y_max - y_min) * 0.05),
                    color='crimson', fontsize=9, fontweight='bold', rotation=90, ha='center')

    # ============ HGL SLOPE RULERS (large-bore multi-pump systems only) ===========
    # These slope rulers let engineers read off equivalent flow from the HGL slope.
    # Only relevant for large-diameter liquid-full lines with multiple pump stations.
    if od >= 12.0 and len(getattr(cfg, 'pump_stations', [])) > 1:
        def _slope_fthead_per_mi(bph, D, A):
            v = bph_to_fts(bph, A) if A > 0 else 0.0
            return liquid_friction_loss_psi(5280.0, D, v, cfg.fluid_sg,
                                            cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft) / grad

        ruler_anchor = y_min + (y_max - y_min) * 0.90  # top 90% of head axis
        x_left, x_right = start - 3, end + 3
        for q, col in [(4000, '#9467bd'), (5000, '#8c564b'),
                       (6000, '#7f7f7f'), (7000, '#bcbd22')]:
            s_val = _slope_fthead_per_mi(q, D_ft, area)
            ax.plot([x_left, x_right],
                    [ruler_anchor, ruler_anchor - s_val * (x_right - x_left)],
                    ls=':', color=col, lw=1.1, alpha=0.85, zorder=2,
                    label=f"{q:,.0f} BPH slope")

    # ============ X-AXIS TICKS ===================================================
    # Label EVERY station (pumps AND boosters) at its milepost, plus start/end.
    # start_label / end_label override the auto-derived names.
    _sl = start_label or _derive_label(cfg, cfg.purge_start_mp, is_start=True)
    _el = end_label   or _derive_label(cfg, cfg.purge_end_mp,   is_start=False)

    tick_map: dict = {start: _sl, end: _el}
    for ps in getattr(cfg, 'pump_stations', []):
        if start < ps.mp < end:
            tick_map[ps.mp] = ps.name
    for b in getattr(cfg, 'booster_configs', []):
        if start < b.mp < end:
            tick_map.setdefault(b.mp, b.name)   # keep a pump's name if same MP

    sorted_ticks = sorted(tick_map.items())
    ax.set_xticks([m for m, _ in sorted_ticks])
    ax.set_xticklabels([n for _, n in sorted_ticks], fontsize=9)

    # Minor tick marks every 5 miles (unlabeled) for distance reference.
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='x', which='minor', length=4, color='gray')
    ax.tick_params(axis='x', which='major', length=7)

    # ============ AXES FRAMING ===================================================
    x_lo = xlim[0] if xlim else start - 3
    x_hi = xlim[1] if xlim else end + 3
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_min, y_max)
    axp.set_ylim(p_bot, p_top)

    ax.set_xlabel("Milepost")
    ax.set_ylabel("Elevation / Total Head, ft")
    axp.set_ylabel("Pressure, psig")

    _pipe_desc = f"{od:.3g}″"
    _dist = end - start
    ax.set_title(title or f"{_pipe_desc} Pipeline Profile — {_dist:.0f}-mi N₂ Purge",
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)

    t_hr = getattr(step, 't_hr', 0.0)
    ax.text(0.01, 0.98, f"t = {t_hr:.1f} h  |  pig MP {pig:.1f}  |  {step.pig_speed_mph:.2f} mph",
            transform=ax.transAxes, fontsize=8, color='dimgray', va='top')
    if annotation:
        ax.text(0.50, 0.94, annotation, transform=ax.transAxes, fontsize=10,
                ha='center', va='top', bbox=dict(boxstyle='round', fc='lightyellow',
                                                  ec='goldenrod', alpha=0.9))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axp.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.10), ncol=5, framealpha=0.9)
    fig.tight_layout(rect=[0, 0.13, 1, 1])


def _derive_label(cfg, mp: float, is_start: bool) -> str:
    """Return a human label for a purge endpoint from cfg infrastructure."""
    # Pump station at this MP
    for ps in getattr(cfg, 'pump_stations', []):
        if abs(ps.mp - mp) < 2.0:
            return ps.name
    # BPCV at end
    if not is_start:
        bpcv = getattr(cfg, 'bpcv', None)
        if bpcv and abs(bpcv.mp - mp) < 2.0:
            return bpcv.name
    return f"MP {mp:.0f}"


def plot_pressure_profile(fig, results: SimResults):
    """
    Pressure profile at the final timestep + MOP envelope.
    Shows: pig-face pressure history, MOP min across all joints, violation markers.
    """
    steps = results.steps
    if not steps:
        return

    mps_  = [s.pig_mp for s in steps]
    pface = [s.pig_face_psig for s in steps]
    inj   = [s.injection_psig for s in steps]

    ax1 = fig.add_subplot(211)
    ax1.plot(mps_, pface, 'b-', linewidth=1.2, label='Pig-face N2 (psig)')
    ax1.plot(mps_, inj,  'g--', linewidth=1.0, label='Injection pressure (psig)')

    # MOP annotations: worst violation per step
    viol_mps = [s.pig_mp for s in steps if s.mop_violations > 0]
    viol_p   = [s.pig_face_psig for s in steps if s.mop_violations > 0]
    if viol_mps:
        ax1.scatter(viol_mps, viol_p, color='red', s=15, zorder=6, label='MOP violation')

    ax1.set_xlabel("Pig milepost")
    ax1.set_ylabel("Pressure (psig)")
    ax1.set_title("Pig-face Drive Pressure vs Position")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # MOP profile from last step
    last = steps[-1]
    if last.mop_results:
        mop_mps = [r.joint.mp for r in last.mop_results]
        mop_est = [r.estimated_psig for r in last.mop_results]
        mop_lim = [r.joint.mop_psig for r in last.mop_results]
        colors  = [('red' if r.status == MOPStatus.VIOLATION else
                    'orange' if r.status == MOPStatus.WARNING else 'steelblue')
                   for r in last.mop_results]
        ax2 = fig.add_subplot(212)
        ax2.plot(mop_mps, mop_lim, 'r--', linewidth=1.0, label='MOP limit')
        ax2.scatter(mop_mps, mop_est, c=colors, s=6, zorder=5, label='Estimated pressure')
        ax2.set_xlabel("Milepost")
        ax2.set_ylabel("Pressure (psig)")
        ax2.set_title("MOP Check at Final Pig Position (red=violation, orange=warning)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    fig.tight_layout()


def plot_n2_inventory(fig, times_hr: List[float], mps: List[float], results: SimResults):
    """Total N2 SCF in system over time + injection SCFM."""
    steps = results.steps
    if not steps:
        return

    times = np.array(times_hr)
    total_scf = np.array([s.total_scf for s in steps])
    inj_scfm  = np.array([s.injection_scfm for s in steps])

    ax1 = fig.add_subplot(211)
    ax1.plot(times, total_scf / 1000.0, 'b-', linewidth=1.2, label='Total N2 in system (kSCF)')
    ax1.set_xlabel("Time (hr)")
    ax1.set_ylabel("kSCF")
    ax1.set_title("N2 Inventory in Gas Column")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(212)
    ax2.plot(times, inj_scfm, 'g-', linewidth=1.2, label='Injection rate (SCFM)')
    ax2.set_xlabel("Time (hr)")
    ax2.set_ylabel("SCFM")
    ax2.set_title("Required N2 Injection Rate")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()


def plot_mop_check(fig, results: SimResults):
    """Heatmap-style MOP margin across all joints and all timesteps."""
    steps = results.steps
    if not steps:
        return
    # Collect worst margin at each joint
    joint_margins: dict[float, list] = {}
    for step in steps:
        for r in step.mop_results:
            joint_margins.setdefault(r.joint.mp, []).append(r.margin_psi)

    mop_mps   = sorted(joint_margins.keys())
    min_margin = [min(joint_margins[mp]) for mp in mop_mps]

    ax = fig.add_subplot(111)
    colors = ['red' if m < 0 else ('orange' if m < 50 else 'steelblue') for m in min_margin]
    ax.bar(mop_mps, min_margin, color=colors, width=0.05, alpha=0.8)
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--', label='MOP limit')
    ax.set_xlabel("Milepost")
    ax.set_ylabel("MOP margin (psi)")
    ax.set_title("Worst MOP Margin by Joint (red = violation)")
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()


def plot_booster_usage(fig, results: SimResults):
    """
    Three-panel booster breakdown:
      Top:    Pig position vs time — flat segments = stalls
      Middle: Flow rate (SCFM) per booster station over time
      Bottom: Suction pressure per active booster over time
    """
    steps = results.steps
    if not steps:
        return

    times_arr = np.array([s.t_hr for s in steps])
    mps_arr   = np.array([s.pig_mp for s in steps])
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                   '#9467bd', '#8c564b', '#e377c2']

    # --- Aggregate per-station timeseries (all timesteps, 0 when not running) ---
    station_order: list[float] = []
    station_meta: dict[float, dict] = {}
    # Pre-pass to discover all stations and their first appearance order
    for step in steps:
        for bs in step.booster_states:
            mp = bs['mp']
            if mp not in station_meta:
                station_order.append(mp)
                station_meta[mp] = {
                    'name': bs.get('name', '') or f'MP {mp:.2f}',
                    'flows': np.zeros(len(steps)),
                    'suctions': np.full(len(steps), np.nan),
                    'flow_limited_mask': np.zeros(len(steps), dtype=bool),
                }

    for i, step in enumerate(steps):
        for bs in step.booster_states:
            mp = bs['mp']
            if bs.get('running'):
                station_meta[mp]['flows'][i] = bs.get('flow_scfm', 0)
                station_meta[mp]['suctions'][i] = bs.get('suction_psig', np.nan)
                station_meta[mp]['flow_limited_mask'][i] = bs.get('flow_limited', False)

    # ---- Panel 1: pig progress ----
    ax1 = fig.add_subplot(311)
    ax1.plot(times_arr, mps_arr, 'k-', linewidth=1.2, label='Pig position')

    # Shade stall periods (speed < 0.5 mph)
    speeds = np.array([s.pig_speed_mph for s in steps])
    stall_mask = speeds < 0.5
    if stall_mask.any():
        ax1.fill_between(times_arr, mps_arr, where=stall_mask,
                         color='red', alpha=0.25, label='Stall (<0.5 mph)')

    # Spread mobilization events
    for ev in results.spread_events:
        ax1.axvline(ev['t_hr'], color='purple', linewidth=0.8, alpha=0.5, linestyle=':')
        ax1.annotate(f"S{ev['spread_id']+1}→{ev['to_mp']:.0f}",
                     (ev['t_hr'], ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 0),
                     fontsize=6, color='purple', rotation=90, va='bottom')

    ax1.set_ylabel("Pig MP")
    ax1.set_title("Booster Station Usage")
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(alpha=0.3)

    # ---- Panel 2: flow rate per station ----
    ax2 = fig.add_subplot(312, sharex=ax1)
    for idx, mp in enumerate(station_order):
        sd = station_meta[mp]
        c  = colors_list[idx % len(colors_list)]
        lbl = sd['name'] or f'MP {mp:.2f}'
        ax2.plot(times_arr, sd['flows'] / 1000.0, color=c, linewidth=1.2,
                 label=f"{lbl}")
        # Mark flow-limited periods
        fl_mask = sd['flow_limited_mask']
        if fl_mask.any():
            ax2.scatter(times_arr[fl_mask], sd['flows'][fl_mask] / 1000.0,
                        color=c, marker='^', s=10, zorder=5, alpha=0.7)

    ax2.set_ylabel("Flow (kSCFM)")
    ax2.legend(fontsize=7, loc='upper left', ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_title("Booster Flow Rate  (▲ = flow-limited at compressor max)")

    # ---- Panel 3: suction pressure per station ----
    ax3 = fig.add_subplot(313, sharex=ax1)
    for idx, mp in enumerate(station_order):
        sd = station_meta[mp]
        c  = colors_list[idx % len(colors_list)]
        lbl = sd['name'] or f'MP {mp:.2f}'
        ax3.plot(times_arr, sd['suctions'], color=c, linewidth=1.0,
                 label=f"{lbl}", alpha=0.85)

    ax3.set_xlabel("Time (hr)")
    ax3.set_ylabel("Suction (psig)")
    ax3.legend(fontsize=7, loc='upper right', ncol=2)
    ax3.grid(alpha=0.3)
    ax3.set_title("Booster Suction Pressure")

    fig.tight_layout()


def plot_backward_pass(fig, bp: BackwardPassResult):
    """Plot minimum N2 floor profile from the backward pass."""
    if not bp.min_injection_profile:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No backward pass data", ha='center', va='center',
                transform=ax.transAxes)
        return

    mps   = [x[0] for x in bp.min_injection_profile]
    inj   = np.cumsum([x[1] for x in bp.min_injection_profile])

    ax = fig.add_subplot(111)
    ax.plot(mps, inj / 1000.0, 'b-', linewidth=1.5, label='Cumulative min SCF (kSCF)')
    ax.set_xlabel("Pig Milepost")
    ax.set_ylabel("kSCF")
    ax.set_title(f"Backward Pass — Minimum N2 Required\n"
                 f"Total: {bp.total_min_scf:,.0f} SCF | "
                 f"Equalization credit: {bp.equalization_credit_scf:,.0f} SCF")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
