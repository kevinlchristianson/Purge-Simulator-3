"""
Results panel — tabbed display of simulation output.

Tabs:
  1. Overview    — key metrics table, completion status
  2. Pig Speed   — speed vs milepost + time
  3. Pressure    — pig-face pressure + MOP profile overlay
  4. N2 Inventory— SCF per segment over time
  5. Logistics   — booster/station events timeline
  6. Optimizer   — spread deployment plan (text + timeline chart)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    _MPL_OK = True
except Exception:
    _MPL_OK = False

from ..engine.simulator import SimResults
from ..engine.backward_pass import BackwardPassResult
from ..engine.optimizer import SpreadPlan
from .charts import (
    plot_pig_speed, plot_pressure_profile, plot_n2_inventory,
    plot_mop_check, plot_booster_usage, plot_backward_pass,
)


class ResultsPanel(ttk.Frame):
    """Right panel with tabbed simulation results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        self._tabs: dict[str, ttk.Frame] = {}
        for name in ["Overview", "Pig Speed", "Pressure + MOP", "N2 Inventory",
                     "Boosters", "Optimizer"]:
            frame = ttk.Frame(self._notebook)
            self._notebook.add(frame, text=name)
            self._tabs[name] = frame

        self._build_overview_tab()
        self._build_placeholder_charts()

    # ------------------------------------------------------------------
    # Tab construction
    # ------------------------------------------------------------------

    def _build_overview_tab(self):
        f = self._tabs["Overview"]
        self._overview_text = tk.Text(f, wrap=tk.WORD, state=tk.DISABLED,
                                       font=("Consolas", 10))
        scroll = ttk.Scrollbar(f, command=self._overview_text.yview)
        self._overview_text.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._overview_text.pack(fill=tk.BOTH, expand=True)

    def _build_placeholder_charts(self):
        if not _MPL_OK:
            for tab_name in ["Pig Speed", "Pressure + MOP", "N2 Inventory",
                             "Logistics", "Optimizer"]:
                ttk.Label(self._tabs[tab_name],
                          text="matplotlib required for charts\n(pip install matplotlib)").pack(expand=True)
            return

        self._figures: dict[str, Figure] = {}
        self._canvases: dict[str, FigureCanvasTkAgg] = {}

        for tab_name in ["Pig Speed", "Pressure + MOP", "N2 Inventory", "Boosters"]:
            fig = Figure(figsize=(10, 7 if tab_name == "Boosters" else 5), dpi=96)
            canvas = FigureCanvasTkAgg(fig, master=self._tabs[tab_name])
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas, self._tabs[tab_name])
            toolbar.update()
            self._figures[tab_name] = fig
            self._canvases[tab_name] = canvas

        # Optimizer tab has text + optional chart
        opt_frame = self._tabs["Optimizer"]
        self._opt_text = tk.Text(opt_frame, wrap=tk.WORD, state=tk.DISABLED,
                                  font=("Consolas", 10), height=12)
        self._opt_text.pack(fill=tk.X)
        if _MPL_OK:
            opt_fig = Figure(figsize=(10, 3), dpi=96)
            opt_canvas = FigureCanvasTkAgg(opt_fig, master=opt_frame)
            opt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._figures["Optimizer"] = opt_fig
            self._canvases["Optimizer"] = opt_canvas

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update(self, results: SimResults):
        """Refresh all tabs with new simulation results."""
        self._update_overview(results)
        if _MPL_OK and results.steps:
            self._update_charts(results)

    def _update_overview(self, results: SimResults):
        steps = results.steps
        if not steps:
            self._set_text(self._overview_text, "No results to display.")
            return

        cfg  = results.config
        last = steps[-1]
        dur_hr    = last.t_hr
        route_mi  = cfg.purge_end_mp - cfg.purge_start_mp
        dist_mi   = last.pig_mp - cfg.purge_start_mp
        speeds    = [s.pig_speed_mph for s in steps]
        faces     = [s.pig_face_psig  for s in steps]
        n_viol    = sum(s.mop_violations for s in steps)
        n_warn    = sum(s.mop_warnings   for s in steps)

        # ── Stall analysis (speed < 0.5 mph) ──────────────────────────────
        STALL_MPH = 0.5
        stall_periods = []
        in_stall = False
        s_start = s_mp = 0.0
        for s in steps:
            if s.pig_speed_mph < STALL_MPH:
                if not in_stall:
                    in_stall = True
                    s_start  = s.t_hr
                    s_mp     = s.pig_mp
            else:
                if in_stall:
                    stall_periods.append((s_start, s.t_hr, s_mp))
                    in_stall = False
        if in_stall:
            stall_periods.append((s_start, last.t_hr, s_mp))
        total_stall_hr = sum(e - b for b, e, _ in stall_periods)

        # ── Booster aggregation ────────────────────────────────────────────
        # booster_stats[mp] = {name, first_hr, last_hr, total_scf, n_run, n_fl}
        booster_stats: dict[float, dict] = {}
        total_booster_scf = 0.0
        for step in steps:
            for bs in step.booster_states:
                mp = bs['mp']
                if mp not in booster_stats:
                    booster_stats[mp] = {
                        'name': bs.get('name', '') or f'MP {mp:.2f}',
                        'first_hr': None, 'last_hr': None,
                        'total_scf': 0.0, 'n_run': 0, 'n_fl': 0,
                    }
                st = booster_stats[mp]
                if bs.get('running'):
                    if st['first_hr'] is None:
                        st['first_hr'] = step.t_hr
                    st['last_hr']  = step.t_hr
                    xfer = bs.get('scf_transferred', 0.0)
                    st['total_scf'] += xfer
                    total_booster_scf += xfer
                    st['n_run'] += 1
                    if bs.get('flow_limited'):
                        st['n_fl'] += 1

        # ── Format lines ──────────────────────────────────────────────────
        lines = [
            "═" * 58,
            f"  SIMULATION {'COMPLETE ✓' if results.completed else 'ABORTED ✗  — ' + results.abort_reason}",
            "═" * 58,
            "",
            "── HEADLINE METRICS ─────────────────────────────────────────",
            f"  Duration:          {dur_hr:.2f} hr",
            f"  Distance:          {dist_mi:.2f} / {route_mi:.1f} mi  "
            f"({100*dist_mi/max(1,route_mi):.1f}%)",
            f"  N2 injected:       {results.total_scf_injected/1e6:.3f} M SCF",
            f"  N2 moved boosters: {total_booster_scf/1e6:.3f} M SCF",
            f"  N2 remaining/pipe: {last.total_scf/1e6:.3f} M SCF  (at final step)",
            f"  Avg pig speed:     {sum(speeds)/len(speeds):.2f} mph  "
            f"(min {min(speeds):.2f}, max {max(speeds):.2f})",
            f"  MOP violations:    {n_viol:,}   warnings: {n_warn:,}",
            "",
            "── PIG STALL ANALYSIS (< 0.5 mph) ──────────────────────────",
        ]

        if stall_periods:
            lines.append(f"  Total stall time:  {total_stall_hr:.2f} hr  "
                         f"({100*total_stall_hr/max(1e-9,dur_hr):.1f}% of run)")
            for b, e, mp in stall_periods:
                lines.append(f"    t={b:.2f}–{e:.2f}h  ({e-b:.2f}hr)  at MP {mp:.2f}")
        else:
            lines.append("  No stalls detected.")

        lines += [
            "",
            "── BOOSTER STATION SUMMARY ──────────────────────────────────",
            f"  {'Station':<16} {'On(h)':>6} {'Off(h)':>6} {'Dur':>5}  "
            f"{'MSCF':>6}  {'kSCFM':>6}  {'FL%':>4}",
            "  " + "-" * 54,
        ]
        for mp in sorted(booster_stats):
            st = booster_stats[mp]
            if st['first_hr'] is None:
                continue
            dur_b  = (st['last_hr'] - st['first_hr']) if st['last_hr'] else 0.0
            # avg SCFM when running: total_scf / (n_run steps × dt_hr × 60)
            # approximate dt from total duration / total steps
            dt_est = dur_hr / max(1, len(steps))
            avg_kscfm = st['total_scf'] / max(1, st['n_run'] * dt_est * 60) / 1000.0
            pct_fl = 100 * st['n_fl'] / max(1, st['n_run'])
            lines.append(
                f"  {st['name']:<16} {st['first_hr']:>6.2f} {st['last_hr']:>6.2f} "
                f"{dur_b:>5.1f}h  {st['total_scf']/1e6:>6.3f}  "
                f"{avg_kscfm:>6.1f}  {pct_fl:>3.0f}%"
            )
        if not any(st['first_hr'] is not None for st in booster_stats.values()):
            lines.append("  (no boosters ran)")

        lines += [
            "",
            "── SPREAD MOBILIZATIONS ─────────────────────────────────────",
        ]
        if results.spread_events:
            for ev in results.spread_events:
                lines.append(
                    f"  Spread {ev['spread_id']+1}  MP {ev['from_mp']:.2f}→{ev['to_mp']:.2f}"
                    f"  depart {ev['t_hr']:.2f}h  arrive {ev['arrive_at_hr']:.2f}h"
                    f"  [{ev['reason']}]"
                )
        else:
            lines.append("  (no spread moves)")

        lines += [
            "",
            "── PUMP STATION SHUTDOWNS ───────────────────────────────────",
        ]
        all_events = [e for s in steps for e in s.station_shutdown_events]
        if all_events:
            for ev in all_events:
                lines.append(
                    f"  t={ev['sim_time_hr']:7.2f}h  MP {ev['mp']:6.2f}  "
                    f"{ev.get('name',''):<8}  {ev['reason']}"
                    f"  (pig {ev.get('distance_to_pig_mi',0):.2f} mi away)"
                )
        else:
            lines.append("  (none)")

        lines += [
            "",
            "── PRESSURE ─────────────────────────────────────────────────",
            f"  Avg pig-face:  {sum(faces)/len(faces):.0f} psig",
            f"  Max pig-face:  {max(faces):.0f} psig",
            f"  Avg injection: {sum(s.injection_psig for s in steps)/len(steps):.0f} psig",
            f"  Max injection: {max(s.injection_psig for s in steps):.0f} psig",
            "",
            f"  Wall clock:  {results.wall_time_s:.1f} s   Steps: {len(steps):,}",
        ]

        self._set_text(self._overview_text, '\n'.join(lines))

    def _update_charts(self, results: SimResults):
        steps = results.steps
        times = [s.t_hr for s in steps]
        mps   = [s.pig_mp for s in steps]

        # Pig Speed
        fig = self._figures["Pig Speed"]
        fig.clear()
        plot_pig_speed(fig, times, mps, [s.pig_speed_mph for s in steps],
                       [s.meter_valve_active for s in steps],
                       [s.slack_line_risk for s in steps])
        self._canvases["Pig Speed"].draw()

        # Pressure + MOP
        fig = self._figures["Pressure + MOP"]
        fig.clear()
        plot_pressure_profile(fig, results)
        self._canvases["Pressure + MOP"].draw()

        # N2 Inventory
        fig = self._figures["N2 Inventory"]
        fig.clear()
        plot_n2_inventory(fig, times, mps, results)
        self._canvases["N2 Inventory"].draw()

        # Boosters
        fig = self._figures["Boosters"]
        fig.clear()
        plot_booster_usage(fig, results)
        self._canvases["Boosters"].draw()

    def show_optimizer_result(self, plan: SpreadPlan, text: str):
        """Display optimizer plan text and timeline."""
        self._set_text(self._opt_text, text)
        self._notebook.select(self._tabs["Optimizer"])

        if _MPL_OK and "Optimizer" in self._figures:
            fig = self._figures["Optimizer"]
            fig.clear()
            _plot_spread_timeline(fig, plan)
            self._canvases["Optimizer"].draw()

    def show_backward_pass(self, bp: BackwardPassResult):
        """Display backward pass N2 floor profile."""
        if not _MPL_OK:
            return
        fig = self._figures.get("N2 Inventory")
        if fig:
            fig.clear()
            plot_backward_pass(fig, bp)
            self._canvases["N2 Inventory"].draw()
        self._notebook.select(self._tabs["N2 Inventory"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_text(widget: tk.Text, text: str):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)


def _plot_spread_timeline(fig: "Figure", plan: SpreadPlan):
    """Simple Gantt-style spread deployment timeline."""
    if not plan.assignments:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No assignments", ha='center', va='center', transform=ax.transAxes)
        return

    ax = fig.add_subplot(111)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for a in plan.assignments:
        c = colors[a.spread_id % len(colors)]
        ax.barh(f"Spread {a.spread_id+1}", plan.mob_time_hr,
                left=a.mob_start_hr, color=c, alpha=0.7, label=f"Spread {a.spread_id+1}")
        if a.pig_arrival_hr:
            ax.axvline(a.pig_arrival_hr, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(a.arrive_at_station_hr + 0.1, a.spread_id,
                f"MP {a.station_mp:.1f}", va='center', fontsize=8)

    ax.set_xlabel("Time (hours)")
    ax.set_title("Spread Deployment Timeline")
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
