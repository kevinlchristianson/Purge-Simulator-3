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
                   throttle_flags: List[bool], slack_flags: List[bool],
                   overspeed_flags: List[bool] = None):
    """Two-panel: speed vs time (top) and speed vs milepost (bottom).

    throttle_flags: steps where the exit raised its inlet pressure to hold the pig at max
    speed; overspeed_flags: steps where it ran out of room (or is fixed) and the pig ran
    over max speed; slack_flags: slack-line risk."""
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    times = np.array(times_hr)
    mps_a = np.array(mps)
    spd   = np.array(speeds_mph)
    mv    = np.array(throttle_flags, dtype=bool)
    sl    = np.array(slack_flags, dtype=bool)
    ov    = np.array(overspeed_flags if overspeed_flags is not None else [False] * len(spd), dtype=bool)

    ax1.plot(times, spd, 'b-', linewidth=1.2, label='Pig speed')
    if mv.any():
        ax1.scatter(times[mv], spd[mv], color='orange', s=8, zorder=5,
                    label='Exit holding pressure (max speed)')
    if ov.any():
        ax1.scatter(times[ov], spd[ov], color='magenta', marker='^', s=18, zorder=6,
                    label='Over max speed')
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
    if ov.any():
        ax2.scatter(mps_a[ov], spd[ov], color='magenta', marker='^', s=18, zorder=6)
    if sl.any():
        ax2.scatter(mps_a[sl], spd[sl], color='red', marker='x', s=20, zorder=6)
    ax2.set_xlabel("Milepost")
    ax2.set_ylabel("Speed (mph)")
    ax2.set_title("Pig Speed vs Milepost")
    ax2.grid(alpha=0.3)

    fig.tight_layout()


# Colors and type for the animated profile, matched to the client web page
# (static/report_template.html, light theme) so the GIF and the page read as one product.
PROFILE_STYLE = {
    "text": "#17202c", "muted": "#5b6676", "border": "#dce1e8", "grid": "#e6e9ee",
    "gas": "#00489a", "liquid": "#1e9638", "mop": "#cf222e", "ground": "#8c7a5b",
    "hgl": "#0e7c86", "station": "#d4580b", "booster": "#8250df", "bad": "#cf222e",
    "inject": "#1a7f37", "flow": "#bf3989", "panel": "#ffffff",
}
_PROFILE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.edgecolor": PROFILE_STYLE["border"],
    "axes.labelcolor": PROFILE_STYLE["muted"],
    "xtick.color": PROFILE_STYLE["muted"],
    "ytick.color": PROFILE_STYLE["muted"],
    "figure.facecolor": PROFILE_STYLE["panel"],
    "axes.facecolor": PROFILE_STYLE["panel"],
}


def _nice_step(span: float, n: int = 6) -> float:
    raw = max(span, 1e-9) / n
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def plot_pipeline_profile(fig, step, cfg, roadmap=None, annotation: str = "",
                          title: str = None, start_label: str = None, end_label: str = None,
                          xlim=None, full_system: bool = False):
    """Pipeline profile at a SINGLE timestep (the animated GIF / MP4 frame).

    Drawn like the client web page's profile: pressure (psig) on the left axis; ground
    elevation and the hydraulic grade line (ft) on the right axis, locked to the psig scale
    (0.433 psi/ft x SG) so the HGL sits above the ground by exactly the liquid pressure.
    The injection rate (SCFM) and liquid flow rate (BPH) float as gauges at the left and
    right edges: each rides from the bottom of the chart at zero to 80% of the way up at
    its maximum.

    Parameters
    ----------
    title : str or None
        Chart title; auto-generated from pipe size and distance if None.
    start_label, end_label : str or None
        Names for the purge start and end, printed under the axis ends.
        Derived from pump stations if None; falls back to "MP X.X".
    xlim : (float, float) or None
        Override X-axis range. None -> (purge_start-3, purge_end+3).
    full_system : bool
        When the liquid leaves past the pig stop (cfg.fluid_exit_mp), draw the whole
        hydraulic system: the liquid line runs on to that endpoint, the view spans
        launch to endpoint (unless xlim says otherwise), the pig stop is marked and the
        slack floor is drawn.
    """
    import matplotlib.pyplot as plt
    with plt.rc_context(_PROFILE_RC):
        _draw_pipeline_profile(fig, step, cfg, roadmap, annotation, title,
                               start_label, end_label, xlim, full_system)


def _draw_pipeline_profile(fig, step, cfg, roadmap, annotation, title,
                           start_label, end_label, xlim, full_system):
    from ..engine.physics import (pipe_area_ft2, mph_to_fts, fts_to_bph, bph_to_fts,
                                   liquid_friction_loss_psi)
    from ..engine.hgl import build_liquid_hgl, hydraulic_end_mp
    from ..engine.bpcv import compute_bpcv_upstream_min_set_point, compute_bpcv_set_point
    C = PROFILE_STYLE

    grad = cfg.fluid_sg * 62.4 / 144.0
    elev = np.asarray(cfg.elevation_profile, dtype=float)
    ep_mp, ep_el = elev[:, 0], elev[:, 1]
    def elev_at(x): return float(np.interp(x, ep_mp, ep_el))
    start, end = cfg.purge_start_mp, cfg.purge_end_mp
    hyd_end = hydraulic_end_mp(cfg)
    tank_mp = hyd_end if hyd_end is not None else end     # where the liquid leaves the line
    full_system = bool(full_system and hyd_end is not None)
    liq_end = tank_mp if full_system else end              # how far this view draws the liquid
    if xlim is None:
        xlim = (start - 3, liq_end + 3) if full_system else (start - 3, end + 3)
    x_lo, x_hi = xlim
    pig   = float(step.pig_mp)
    pface = float(step.pig_face_psig)
    sps   = step.station_pressures or []
    inj_scfm = max(0.0, getattr(step, 'injection_scfm', 0.0) or 0.0)

    od, wt = cfg.pipe_geometry.od_wt_at(pig)
    D_ft = (od - 2.0 * wt) / 12.0
    area = pipe_area_ft2(od, wt)
    flow_bph = fts_to_bph(mph_to_fts(step.pig_speed_mph), area)
    v_fts = (bph_to_fts(fts_to_bph(mph_to_fts(max(step.pig_speed_mph, cfg.min_speed_mph)), area),
                        area) if area > 0 else 0.0)

    def _fric_head(a, b):
        L = abs(b - a) * 5280.0
        return liquid_friction_loss_psi(L, D_ft, v_fts, cfg.fluid_sg,
                                        cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft) / grad

    # ---- the liquid ahead of the pig: the same pump/BPCV-aware HGL as hgl.compute_hgl ----
    bpcv = getattr(cfg, 'bpcv', None)
    bpcv_mp = bpcv.mp if (bpcv is not None and pig < bpcv.mp) else None
    mt_psig = getattr(cfg, 'exit_pressure_run_psig', step.exit_psig)
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
    grid = sorted(set([m for m in ep_mp if pig < m <= liq_end]
                      + [p[0] for p in pumps if p[0] <= liq_end]
                      + ([bpcv_mp] if bpcv_mp and bpcv_mp <= liq_end else []) + [liq_end]))
    grid = [m for m in grid if m > pig] or [liq_end]
    hgl_pts = build_liquid_hgl(grid, pig, pface, pumps, bpcv_mp, bpcv_sp, tank_mp, mt_psig,
                               elev_at, _fric_head, grad, clamp_to_ground=False)
    ox = [pig] + [m for m, _ in hgl_pts]
    oy = [pface] + [(h - elev_at(m)) * grad for m, h in hgl_pts]
    for pmp, suc, dis in pumps:
        if pmp <= liq_end:
            ox += [pmp, pmp]; oy += [suc, dis]
    order = np.argsort(ox, kind='stable')
    ox = np.asarray(ox, float)[order]; oy = np.asarray(oy, float)[order]
    liquid_on = pig < liq_end - 1e-6

    # ---- one scale for both axes: psig on the left, feet on the right ----------------
    # Everything is drawn in psig; a height in feet is (ft - datum) x grad. The datum is the
    # lowest ground in view, so the ground and the HGL sit on the same scale as pressure.
    view = (ep_mp >= x_lo) & (ep_mp <= x_hi)
    vm, ve = (ep_mp[view], ep_el[view]) if view.any() else (ep_mp, ep_el)
    datum = float(np.min(ve))
    to_p = lambda ft: (np.asarray(ft, float) - datum) * grad          # feet -> psig scale
    if roadmap is not None and len(roadmap.mp):
        r_mp = np.asarray(roadmap.mp, float); r_mop = np.asarray(roadmap.mop_psig, float)
        mop_v = np.interp(vm, r_mp, r_mop)
    else:
        r_mp = r_mop = None
        mop_v = np.full(vm.shape, max(700.0, pface * 1.2))
    p_top = max(float(np.max(to_p(ve) + mop_v)), float(np.max(mop_v)), pface,
                max([s['discharge_psig'] for s in sps] or [0.0])) * 1.08
    hgl_p = to_p([elev_at(x) for x in ox]) + oy
    floor_psig = float(getattr(cfg, 'slack_floor_psig', 25.0))
    p_lo = min(0.0, float(np.min(oy)) if liquid_on else 0.0, floor_psig if full_system else 0.0)
    p_bot = p_lo - 0.05 * (p_top - p_lo)

    # ---- frame: header, legend row, plot --------------------------------------------
    W, H = fig.get_size_inches()
    top_in, bot_in = 1.05, 0.62                               # header / axis labels, inches
    ax = fig.add_axes([0.065, bot_in / H, 0.87, 1 - (top_in + bot_in) / H])
    axr = ax.twinx()
    for a in (ax, axr):
        for side in ('top',):
            a.spines[side].set_visible(False)
        a.tick_params(length=0, labelsize=9)
    ax.set_xlim(x_lo, x_hi); axr.set_xlim(x_lo, x_hi)
    ax.set_ylim(p_bot, p_top)
    axr.set_ylim(datum + p_bot / grad, datum + p_top / grad)
    ax.grid(True, color=C["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    step_x = _nice_step(x_hi - x_lo, 8)
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax.xaxis.set_major_locator(MultipleLocator(step_x))
    ax.yaxis.set_major_locator(MultipleLocator(_nice_step(p_top - p_bot, 6)))
    axr.yaxis.set_major_locator(MultipleLocator(_nice_step((p_top - p_bot) / grad, 6)))
    comma = FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.yaxis.set_major_formatter(comma); axr.yaxis.set_major_formatter(comma)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Milepost", fontsize=9.5)
    ax.set_ylabel("psig", fontsize=9.5)
    axr.set_ylabel("Elevation / HGL (ft)", fontsize=9.5, rotation=270, labelpad=14)

    # ---- ground and grade lines ------------------------------------------------------
    gx = ep_mp[(ep_mp >= x_lo - 1) & (ep_mp <= x_hi + 1)]
    gy = to_p(np.interp(gx, ep_mp, ep_el))
    ax.fill_between(gx, p_bot, gy, color=C["ground"], alpha=0.12, lw=0, zorder=1)
    ax.plot(gx, gy, color=C["ground"], lw=1.3, zorder=2, label="Elevation")
    if r_mp is not None:
        ax.plot(r_mp, r_mop, color=C["mop"], lw=1.3, ls=(0, (6, 4)), zorder=3, label="MOP")

    # N2 behind the pig: flat per segment
    gas_x, gas_y = [], []
    for seg in sorted(step.segments or [], key=lambda s: s['upstream_mp']):
        if seg['upstream_mp'] >= pig:
            continue
        a, b, p = seg['upstream_mp'], min(seg['downstream_mp'], pig), seg['pressure_psig']
        gas_x += [a, b]; gas_y += [p, p]
    if not gas_x and pig > start:
        gas_x, gas_y = [start, pig], [pface, pface]
    if gas_x:
        ax.plot(gas_x, gas_y, color=C["gas"], lw=2.4, zorder=5, solid_capstyle='round',
                label="N2 pressure (behind pig)")
    if liquid_on:
        ax.plot(ox, oy, color=C["liquid"], lw=2.4, zorder=5, solid_joinstyle='round',
                label="Liquid pressure (ahead of pig)")
        ax.plot(ox, hgl_p, color=C["hgl"], lw=2.0, ls=(0, (7, 3)), zorder=4,
                label="Hydraulic grade line")
        g_at = to_p([elev_at(x) for x in ox])
        if np.any(hgl_p < g_at - 0.5 * grad):
            ax.fill_between(ox, hgl_p, g_at, where=(hgl_p < g_at), interpolate=True,
                            color=C["bad"], alpha=0.28, lw=0, zorder=4, label="Slack (HGL below ground)")
    if full_system:
        ax.axhline(floor_psig, color=C["bad"], lw=1.0, ls=(0, (2, 3)), zorder=3,
                   label=f"Slack floor ({floor_psig:,.0f} psig)")

    # HGL slope rulers (large-bore multi-pump systems): read the flow off the HGL slope
    if od >= 12.0 and len(getattr(cfg, 'pump_stations', [])) > 1:
        ruler_y = p_bot + (p_top - p_bot) * 0.93
        for q, col in [(4000, '#9467bd'), (5000, '#8c564b'), (6000, '#7f7f7f'), (7000, '#bcbd22')]:
            v = bph_to_fts(q, area) if area > 0 else 0.0
            s_psi = liquid_friction_loss_psi(5280.0, D_ft, v, cfg.fluid_sg,
                                             cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft)
            ax.plot([x_lo, x_hi], [ruler_y, ruler_y - s_psi * (x_hi - x_lo)], ls=':', color=col,
                    lw=1.0, alpha=0.8, zorder=2, label=f"{q:,.0f} BPH slope")

    # ---- vertical markers, labeled along the top like the web page ------------------
    trans = ax.get_xaxis_transform()
    def vmark(x, label, color, ls, lw, row=0, bold=False):
        if not (x_lo <= x <= x_hi):
            return
        ax.axvline(x, color=color, lw=lw, ls=ls, zorder=6)
        if label:
            ax.text(x, 0.985 - row * 0.045, " " + label, transform=trans, color=color, fontsize=8.5,
                    fontweight='bold' if bold else 'normal', va='top', ha='left', zorder=9,
                    clip_on=True)
    vmark(pig, "Pig", C["text"], '-', 1.8, bold=True)
    if full_system:
        vmark(end, "Pig stop", C["muted"], (0, (6, 3)), 1.2, row=1)
        vmark(tank_mp, "", C["muted"], (0, (6, 3)), 1.2)
    names_row = 1 if not full_system else 2
    for k, s in enumerate(sorted(sps, key=lambda s: s['mp'])):
        vmark(s['mp'], s.get('name', ''), C["station"], (0, (6, 3)), 1.2, row=names_row + (k % 2))
        if x_lo <= s['mp'] <= x_hi and s.get('discharge_psig') is not None:
            ax.plot([s['mp']], [s['discharge_psig']], 'o', color=C["station"], ms=5, zorder=8)
            ax.annotate(f"{s['discharge_psig']:,.0f}", (s['mp'], s['discharge_psig']), color=C["station"],
                        fontsize=8.5, fontweight='bold', ha='center', xytext=(0, 7),
                        textcoords='offset points', zorder=9)
    if bpcv is not None:
        vmark(bpcv.mp, "BPCV", C["mop"], (0, (6, 3)), 1.2, row=names_row)

    # boosters running now: discharge pressure and flow
    for bs in (step.booster_states or []):
        if bs.get('running') and x_lo <= bs['mp'] <= x_hi:
            bp = bs.get('discharge_psig', pface)
            ax.plot([bs['mp']], [bp], marker='s', color=C["booster"], ms=9, mec='white', mew=1.2, zorder=8)
            ax.annotate(f"{bs.get('name', 'Booster')}  {bs.get('flow_scfm', 0.0) or 0.0:,.0f} SCFM",
                        (bs['mp'], bp), color=C["booster"], fontsize=8, fontweight='bold', ha='center',
                        xytext=(0, 10), textcoords='offset points', zorder=9)

    # pig face pressure
    ax.plot([pig], [pface], 'o', color=C["liquid"], ms=8, mec='white', mew=1.5, zorder=10)
    ax.annotate(f"{pface:,.0f} psig", (pig, pface), color=C["text"], fontsize=9, fontweight='bold',
                ha='left', va='center', xytext=(8, 0), textcoords='offset points', zorder=11,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C["border"], lw=0.8, alpha=0.92))

    # ---- rate floaters: a gauge at each edge, 0 at the bottom, max at 80% up -----------
    _max_inj = max(1.0, getattr(cfg, 'max_injection_scfm', 5000.0) or 5000.0)
    _max_flow = fts_to_bph(mph_to_fts(max(getattr(cfg, 'max_speed_mph', 5.0) or 5.0, 1.0)), area)
    def floater(xf, frac, marker, color, text, ha):
        lo_f, hi_f = 0.06, 0.06 + 0.80
        ax.plot([xf, xf], [lo_f, hi_f], transform=ax.transAxes, color=color, alpha=0.18, lw=5,
                solid_capstyle='round', zorder=7, clip_on=False)
        yf = lo_f + min(max(frac, 0.0), 1.0) * (hi_f - lo_f)
        ax.plot([xf], [yf], marker=marker, color=color, ms=11, mec='white', mew=1.2,
                transform=ax.transAxes, zorder=11, clip_on=False)
        ax.annotate(text, (xf, yf), xycoords='axes fraction', color=color, fontsize=9,
                    fontweight='bold', ha=ha, va='center',
                    xytext=(11 if ha == 'left' else -11, 0), textcoords='offset points', zorder=12,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, lw=0.9, alpha=0.95))
    floater(0.018, inj_scfm / _max_inj, '^', C["inject"], f"N2 injection  {inj_scfm:,.0f} SCFM", 'left')
    floater(0.982, flow_bph / max(1.0, _max_flow), 'o', C["flow"], f"Liquid flow  {flow_bph:,.0f} BPH", 'right')

    if annotation:
        ax.text(0.5, 0.86, annotation, transform=ax.transAxes, fontsize=9.5, color=C["text"],
                ha='center', va='center', zorder=12,
                bbox=dict(boxstyle='round,pad=0.4', fc='#fff8e1', ec='#e3c46b', lw=0.8))

    # ---- names of the ends, under the axis ------------------------------------------
    _sl = start_label or _derive_label(cfg, cfg.purge_start_mp, is_start=True)
    _el = end_label or _derive_label(cfg, cfg.purge_end_mp, is_start=False)
    ends = [(start, _sl, 'left'), (end, _el + (" (pig stop)" if full_system else ""), 'left' if full_system else 'right')]
    if full_system:
        ends.append((tank_mp, f"Hydraulic endpoint, MP {tank_mp:.1f}", 'right'))
    for x, lbl, ha in ends:
        if x_lo <= x <= x_hi:
            ax.annotate(lbl, (x, 0), xycoords=trans, xytext=(0, -30), textcoords='offset points',
                        ha=ha, va='top', fontsize=8.5, color=C["muted"], fontweight='bold')

    # ---- header: title, the moment's read-out, legend ------------------------------
    _dist = end - start
    _title = title or f"{od:.3g}″ pipeline profile — {_dist:.0f}-mile N₂ purge"
    if full_system:
        _title += f"  ·  whole hydraulic system to MP {tank_mp:.1f}"
    fig.text(0.065, 1 - 0.16 / H, _title, fontsize=12.5, fontweight='bold', color=C["text"],
             ha='left', va='top')
    t_hr = getattr(step, 't_hr', 0.0)
    read = (f"{t_hr:.1f} hr into the run  ·  pig at MP {pig:.2f} moving {step.pig_speed_mph:.2f} mph"
            f"  ·  N2 drive {pface:,.0f} psig  ·  exit {step.exit_psig:,.0f} psig at MP {step.exit_mp:.2f}")
    flags = []
    if getattr(step, 'overspeed', False):
        flags.append("over max speed")
    if getattr(step, 'slack_line_risk', False):
        flags.append("slack-line risk")
    fig.text(0.065, 1 - 0.46 / H, read, fontsize=9.5, color=C["muted"], ha='left', va='top')
    if flags:
        fig.text(0.935, 1 - 0.46 / H, "  ·  ".join(flags), fontsize=9.5, color=C["bad"],
                 fontweight='bold', ha='right', va='top')
    h1, l1 = ax.get_legend_handles_labels()
    _order = ["N2 pressure", "Liquid pressure", "MOP", "Elevation", "Hydraulic grade", "Slack"]
    _rank = lambda l: next((i for i, k in enumerate(_order) if l.startswith(k)), len(_order))
    pairs = sorted(zip(h1, l1), key=lambda hl: _rank(hl[1]))
    h1, l1 = [h for h, _ in pairs], [l for _, l in pairs]
    leg = fig.legend(h1, l1, loc='upper left', bbox_to_anchor=(0.058, 1 - 0.66 / H), ncol=min(len(l1), 8),
                     frameon=False, fontsize=8.5, handlelength=2.0, columnspacing=1.4,
                     handletextpad=0.5, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(C["text"])


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
