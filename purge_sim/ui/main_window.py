"""
Main application window.

Layout:
  Left panel  (30%): configuration inputs (collapsible sections)
  Right panel (70%): tabbed results (charts, tables, optimizer output)

The window persists between runs — clicking "Run" updates the right panel
without rebuilding the left panel. Scenario save/load lives in the toolbar.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional

from .config_panel import ConfigPanel
from .results_panel import ResultsPanel
from .scenario_manager import ScenarioManager
from ..engine.simulator import SimConfig, SimResults, simulate
from ..engine.backward_pass import BackwardPassConfig, run_backward_pass
from ..engine.optimizer import OptimizerConfig, run_optimizer, format_spread_plan
from ..engine.log_export import export_run_log
from ..data.scenario import Scenario, ScenarioMeta, ScenarioInputs, save_scenario, load_scenario


WINDOW_TITLE = "Purge Simulator v30"
WINDOW_MIN_W = 1280
WINDOW_MIN_H = 800
DEFAULT_SCENARIO_DIR = os.path.join(os.path.expanduser("~"), "PurgeSimScenarios")


class MainWindow:
    """Root Tkinter window and application controller."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.minsize(WINDOW_MIN_W, WINDOW_MIN_H)
        self.root.geometry(f"{WINDOW_MIN_W}x{WINDOW_MIN_H}")

        self._last_results: Optional[SimResults] = None
        self._current_scenario_path: Optional[str] = None
        self._running = False

        self._build_menu()
        self._build_toolbar()
        self._build_panels()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Scenario",    command=self._on_new_scenario)
        file_menu.add_command(label="Open Scenario…",  command=self._on_open_scenario)
        file_menu.add_command(label="Save Scenario",   command=self._on_save_scenario)
        file_menu.add_command(label="Save Scenario As…", command=self._on_save_scenario_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Run Log…",  command=self._on_export_log)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="Run Simulation",  command=self._on_run,     accelerator="F5")
        run_menu.add_command(label="Run Optimizer",   command=self._on_optimize, accelerator="F6")
        run_menu.add_command(label="Backward Pass",   command=self._on_backward_pass)
        menubar.add_cascade(label="Run", menu=run_menu)

        self.root.bind("<F5>", lambda e: self._on_run())
        self.root.bind("<F6>", lambda e: self._on_optimize())

        self.root.config(menu=menubar)

    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, relief=tk.RIDGE)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self._btn_run = ttk.Button(toolbar, text="▶  Run (F5)", command=self._on_run, width=14)
        self._btn_run.pack(side=tk.LEFT, padx=4, pady=3)

        self._btn_opt = ttk.Button(toolbar, text="⚙  Optimize (F6)", command=self._on_optimize, width=16)
        self._btn_opt.pack(side=tk.LEFT, padx=4, pady=3)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)

        ttk.Button(toolbar, text="Open…",  command=self._on_open_scenario, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save",   command=self._on_save_scenario,  width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save As…", command=self._on_save_scenario_as, width=9).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)
        ttk.Button(toolbar, text="Compare…",   command=self._on_compare,    width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export Log…", command=self._on_export_log, width=12).pack(side=tk.LEFT, padx=2)

    def _build_panels(self):
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: config
        left_frame = ttk.LabelFrame(pane, text="Configuration", padding=4)
        pane.add(left_frame, weight=3)
        self.config_panel = ConfigPanel(left_frame, on_change=self._on_config_change)
        self.config_panel.pack(fill=tk.BOTH, expand=True)

        # Right: results
        right_frame = ttk.LabelFrame(pane, text="Results", padding=4)
        pane.add(right_frame, weight=7)
        self.results_panel = ResultsPanel(right_frame)
        self.results_panel.pack(fill=tk.BOTH, expand=True)

        self.scenario_manager = ScenarioManager(DEFAULT_SCENARIO_DIR)

    def _build_statusbar(self):
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self._status_var = tk.StringVar(value="Ready")
        self._progress_var = tk.DoubleVar(value=0.0)

        ttk.Label(status_frame, textvariable=self._status_var, anchor=tk.W).pack(side=tk.LEFT, padx=6)
        self._progress_bar = ttk.Progressbar(
            status_frame, variable=self._progress_var,
            maximum=100, length=200, mode='determinate'
        )
        self._progress_bar.pack(side=tk.RIGHT, padx=6, pady=2)

    # ------------------------------------------------------------------
    # Run actions
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._running:
            return
        try:
            cfg = self.config_panel.build_sim_config()
        except ValueError as e:
            messagebox.showerror("Configuration Error", str(e))
            return

        self._set_status("Running simulation…", clear_progress=True)
        self._btn_run.config(state=tk.DISABLED)
        self._running = True

        def _worker():
            try:
                results = simulate(cfg, progress_cb=self._on_progress)
                self.root.after(0, self._on_run_complete, results)
            except Exception as ex:
                self.root.after(0, self._on_run_error, str(ex))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_optimize(self):
        if self._running:
            return
        try:
            cfg = self.config_panel.build_sim_config()
            bp_cfg = self.config_panel.build_backward_pass_config()
        except ValueError as e:
            messagebox.showerror("Configuration Error", str(e))
            return

        self._set_status("Running optimizer…")

        def _worker():
            try:
                bp    = run_backward_pass(bp_cfg)
                opt_cfg = self.config_panel.build_optimizer_config(bp)
                plan  = run_optimizer(opt_cfg)
                text  = format_spread_plan(plan)
                self.root.after(0, self.results_panel.show_optimizer_result, plan, text)
                self.root.after(0, self._set_status, "Optimizer complete")
            except Exception as ex:
                self.root.after(0, self._on_run_error, str(ex))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_backward_pass(self):
        try:
            bp_cfg = self.config_panel.build_backward_pass_config()
        except ValueError as e:
            messagebox.showerror("Configuration Error", str(e))
            return
        self._set_status("Running backward pass…")

        def _worker():
            try:
                bp = run_backward_pass(bp_cfg)
                self.root.after(0, self.results_panel.show_backward_pass, bp)
                self.root.after(0, self._set_status,
                                f"Backward pass complete — min N2: {bp.total_min_scf:,.0f} SCF")
            except Exception as ex:
                self.root.after(0, self._on_run_error, str(ex))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_progress(self, fraction: float):
        self.root.after(0, self._progress_var.set, fraction * 100)

    def _on_run_complete(self, results: SimResults):
        self._last_results = results
        self._running = False
        self._btn_run.config(state=tk.NORMAL)
        self._progress_var.set(100)
        n_steps = len(results.steps)
        status = (f"Complete — {n_steps} steps, "
                  f"{results.total_scf_injected:,.0f} SCF injected, "
                  f"{results.wall_time_s:.1f}s")
        if not results.completed:
            status += f" | ABORTED: {results.abort_reason}"
        self._set_status(status)
        self.results_panel.update(results)

    def _on_run_error(self, msg: str):
        self._running = False
        self._btn_run.config(state=tk.NORMAL)
        self._set_status(f"Error: {msg}")
        messagebox.showerror("Simulation Error", msg)

    # ------------------------------------------------------------------
    # Scenario actions
    # ------------------------------------------------------------------

    def _on_new_scenario(self):
        self.config_panel.reset()
        self._current_scenario_path = None
        self.root.title(WINDOW_TITLE)

    def _on_open_scenario(self):
        path = filedialog.askopenfilename(
            title="Open Scenario",
            initialdir=DEFAULT_SCENARIO_DIR,
            filetypes=[("Scenario files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            scenario = load_scenario(path)
            self.config_panel.load_from_scenario(scenario.inputs)
            self._current_scenario_path = path
            self.root.title(f"{WINDOW_TITLE} — {scenario.meta.name}")
            self._set_status(f"Loaded: {scenario.meta.name}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _on_save_scenario(self):
        if self._current_scenario_path:
            self._do_save(self._current_scenario_path)
        else:
            self._on_save_scenario_as()

    def _on_save_scenario_as(self):
        os.makedirs(DEFAULT_SCENARIO_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save Scenario As",
            initialdir=DEFAULT_SCENARIO_DIR,
            defaultextension=".json",
            filetypes=[("Scenario files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._do_save(path)

    def _do_save(self, path: str):
        try:
            inputs  = self.config_panel.collect_inputs()
            name    = os.path.splitext(os.path.basename(path))[0]
            scenario = Scenario(
                meta=ScenarioMeta(name=name),
                inputs=inputs,
            )
            save_scenario(scenario, path)
            self._current_scenario_path = path
            self.root.title(f"{WINDOW_TITLE} — {name}")
            self._set_status(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _on_compare(self):
        messagebox.showinfo("Compare Scenarios",
                            "Open two or more scenarios and use File > Save to compare them side by side.\n"
                            "(Full scenario comparison UI coming in a future update.)")

    def _on_export_log(self):
        if not self._last_results:
            messagebox.showwarning("No Results", "Run a simulation first before exporting a log.")
            return
        scenario_name = ""
        if self._current_scenario_path:
            scenario_name = os.path.splitext(os.path.basename(self._current_scenario_path))[0]
        default_name = f"{scenario_name or 'run'}_log.txt"
        path = filedialog.asksaveasfilename(
            title="Export Run Log",
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            export_run_log(self._last_results, path, scenario_name=scenario_name)
            self._set_status(f"Log exported: {path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_config_change(self):
        # Auto-validate on every config change
        pass

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, clear_progress: bool = False):
        self._status_var.set(msg)
        if clear_progress:
            self._progress_var.set(0)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        self.root.mainloop()
