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
