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


def plot_pipeline_profile(fig, step, cfg, roadmap=None, annotation: str = ""):
    """Reference-style dual-axis pipeline profile at a SINGLE timestep.

    Left Y: Elevation / Total Head (ft).  Right Y: Pressure (psig), aligned so 0 psig
    sits on the 4,000-ft head gridline.  X: Milepost (all 9 stations ticked).

    Renders, matching the 24ML reference:
      - ground elevation (navy) and the MOP envelope (orange), as head
      - AHEAD of the pig: the Operating Head Gradient / liquid HGL (blue, station dots),
        built from the pump-supported station discharge pressures (stays above ground)
      - BEHIND the pig: ONLY the N2 gas pressure (teal) — the sole physical data there
      - N2 interface (green vertical) + pig (orange square) with face-pressure label
      - active booster/compressor markers (gold), the BPCV at SC (red dashed)
      - floating flow-rate marker (magenta, right margin) and SP injection marker (green)
      - per-station operating-pressure labels
    """
    from ..engine.physics import (pipe_area_ft2, mph_to_fts, fts_to_bph, bph_to_fts,
                                   liquid_friction_loss_psi)

    grad = cfg.fluid_sg * 62.4 / 144.0          # psi per ft of head
    elev = np.asarray(cfg.elevation_profile, dtype=float)
    ep_mp, ep_el = elev[:, 0], elev[:, 1]
    def elev_at(x): return float(np.interp(x, ep_mp, ep_el))
    start, end = cfg.purge_start_mp, cfg.purge_end_mp
    pig  = float(step.pig_mp)
    pface = float(step.pig_face_psig)
    sps  = step.station_pressures or []

    ax  = fig.add_subplot(111)                   # LEFT: elevation / head (ft)
    axp = ax.twinx()                             # RIGHT: pressure (psig) — everything else

    # friction model for the ahead operating-pressure walk
    od, wt = cfg.pipe_geometry.od_wt_at(pig)
    D_ft = (od - 2.0 * wt) / 12.0
    area = pipe_area_ft2(od, wt)
    flow_bph = fts_to_bph(mph_to_fts(step.pig_speed_mph), area)            # actual (for marker)
    # friction for the HGL is drawn at the SAME flow the engine sized the stations at
    # (pig speed floored at min) so the grade lines and station suction/discharge are consistent.
    v_fts = bph_to_fts(fts_to_bph(mph_to_fts(max(step.pig_speed_mph, cfg.min_speed_mph)), area),
                       area) if area > 0 else 0.0

    # ============ RIGHT AXIS (psig): operating pressure everywhere ===================
    # BEHIND the pig: N2 gas — FLAT per segment (uniform pressure inside a gas segment).
    gas_lab = False
    for seg in (step.segments or []):
        if seg['upstream_mp'] >= pig:
            continue
        a, b, p = seg['upstream_mp'], min(seg['downstream_mp'], pig), seg['pressure_psig']
        axp.plot([a, b], [p, p], color='deepskyblue', lw=1.8, zorder=4,
                 label=None if gas_lab else 'Operating Pressure')
        gas_lab = True

    # AHEAD of the pig: the engine builds the liquid HGL as the upper envelope of the physical
    # supports (pig drive + pump suction/discharge lifts + BPCV backpressure), clamped so it
    # never rides below ground. The chart just plots it — operating pressure on the psi axis
    # (= (HGL - ground) * grad, always >= 0) and the HGL itself on the head axis (later).
    from ..engine.hgl import build_liquid_hgl
    from ..engine.bpcv import compute_bpcv_upstream_min_set_point, compute_bpcv_set_point

    bpcv = getattr(cfg, 'bpcv', None)
    bpcv_mp = bpcv.mp if (bpcv is not None and pig < bpcv.mp) else None
    mt_psig = getattr(cfg, 'exit_pressure_run_psig', step.exit_psig)

    def _fric_head(a, b):                         # friction head (ft) between mileposts a,b
        L = abs(b - a) * 5280.0
        return liquid_friction_loss_psi(L, D_ft, v_fts, cfg.fluid_sg,
                                        cfg.fluid_viscosity_cst, cfg.fluid_roughness_ft) / grad

    # BPCV set point (its backpressure, capped at downstream MOP) — the same the engine uses
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
                               elev_at, _fric_head, grad)

    ox = [pig] + [m for m, _ in hgl_pts]
    oy = [pface] + [(h - elev_at(m)) * grad for m, h in hgl_pts]
    for pmp, suc, dis in pumps:                    # clean vertical lift edge at each pump
        ox += [pmp, pmp]; oy += [suc, dis]
    order = np.argsort(ox)
    ox = list(np.asarray(ox)[order]); oy = list(np.asarray(oy)[order])
    axp.plot(ox, oy, color='deepskyblue', lw=1.8, zorder=4,
             label=None if gas_lab else 'Operating Pressure')

    # MOP (psig) on the right axis
    if roadmap is not None:
        axp.plot(np.asarray(roadmap.mp, float), np.asarray(roadmap.mop_psig, float),
                 color='darkorange', lw=1.1, zorder=3, label='MOP (psig)')

    # station operating-pressure dots + labels (right axis)
    for s in sps:
        axp.plot([s['mp']], [s['discharge_psig']], 'o', color='royalblue', ms=5, zorder=6)
        axp.annotate(f"{s['discharge_psig']:.0f}", (s['mp'], s['discharge_psig']),
                     color='royalblue', fontsize=10, fontweight='bold', ha='center',
                     xytext=(0, 7), textcoords='offset points')

    # interface pressure: its OWN dot on the green line, on the psi scale
    axp.plot([pig], [pface], 'o', color='limegreen', ms=9, markeredgecolor='darkgreen',
             zorder=10)
    axp.annotate(f"{pface:.0f}", (pig, pface), color='darkgreen', fontsize=12,
                 fontweight='bold', ha='left', xytext=(7, 0), textcoords='offset points')

    # flow + SP injection markers (right axis, value/10 like the reference)
    axp.plot([end - 3], [flow_bph / 10.0], 'o', color='magenta', ms=12, zorder=11)
    axp.annotate(f"Flow {flow_bph:,.0f} BPH", (end - 3, flow_bph / 10.0), color='magenta',
                 fontsize=10, fontweight='bold', ha='right', va='center',
                 xytext=(-12, 0), textcoords='offset points')
    inj_scfm = max(0.0, getattr(step, 'injection_scfm', 0.0))
    axp.plot([start + 3], [inj_scfm / 10.0], '^', color='green', ms=13, zorder=11)
    axp.annotate(f"SP inj {inj_scfm:,.0f} SCFM", (start + 3, inj_scfm / 10.0), color='green',
                 fontsize=10, fontweight='bold', ha='left', va='center',
                 xytext=(12, 0), textcoords='offset points')

    # ============ LEFT AXIS (ft): elevation + HGL only ===============================
    ax.plot(ep_mp, ep_el, color='navy', lw=1.3, zorder=2, label='Ground Elevation')
    # MOP as HEAD (red) on the left axis: the ceiling the HGL must stay under.
    if roadmap is not None:
        mp_m = np.asarray(roadmap.mp, float)
        ax.plot(mp_m, np.interp(mp_m, ep_mp, ep_el) + np.asarray(roadmap.mop_psig, float) / grad,
                color='red', lw=1.0, zorder=2, label='Max Operating Head')
    # HGL ahead = elev + operating_pressure/grad — carries the pump lifts and BPCV drop
    hgl_ft = [elev_at(x) + p / grad for x, p in zip(ox, oy)]
    ax.plot(ox, hgl_ft, color='royalblue', lw=1.9, zorder=5,
            label='Operating Head Gradient (HGL)')

    # N2 interface (green vertical) + pig square (on the HGL/left at its head)
    ax.axvline(pig, color='limegreen', lw=2.4, zorder=6)
    ax.plot([pig], [elev_at(pig) + pface / grad], marker='s', color='orange', ms=12,
            markeredgecolor='k', zorder=8, label='PIG')

    # active booster / compressor markers (gold) + their N2 flow rate (SCFM)
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
        ax.annotate("BPCV", (bpcv.mp, 250), color='crimson', fontsize=9, fontweight='bold',
                    rotation=90, ha='center')

    # --- all 9 station ticks --------------------------------------------------------
    bsta = sorted(cfg.booster_configs, key=lambda b: b.mp)
    ticks = [(start, 'SP')] + [(b.mp, b.name) for b in bsta] + [(end, 'MT')]
    ax.set_xticks([m for m, _ in ticks])
    ax.set_xticklabels([n for _, n in ticks], fontsize=9)

    # --- axes framing: align 0 psig with the 4,000-ft head gridline ------------------
    ax.set_xlim(start - 3, end + 3)
    ax.set_ylim(0, 6500)
    frac = 4000.0 / 6500.0
    mop_max = float(np.max(roadmap.mop_psig)) if roadmap is not None else 700.0
    p_top = max(1000.0, mop_max * 1.05)
    p_bot = -frac / (1.0 - frac) * p_top         # 0 psig lands on the 4,000-ft line
    axp.set_ylim(p_bot, p_top)
    ax.set_xlabel("Milepost")
    ax.set_ylabel("Elevation / Total Head, ft")
    axp.set_ylabel("Pressure, psig")
    ax.set_title("24-inch Main Line Profile — Nitrogen Displacement of Crude Oil Line-fill",
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)

    t_hr = getattr(step, 't_hr', 0.0)
    ax.text(0.01, 0.02, f"t = {t_hr:.1f} h    pig MP {pig:.1f}    {step.pig_speed_mph:.2f} mph",
            transform=ax.transAxes, fontsize=8, color='dimgray')
    if annotation:
        ax.text(0.30, 0.80, annotation, transform=ax.transAxes, fontsize=10,
                ha='left', va='top', bbox=dict(boxstyle='round', fc='lightyellow',
                                               ec='goldenrod'))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axp.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right', ncol=2)
    fig.tight_layout()


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
