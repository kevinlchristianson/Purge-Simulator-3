"""
Configuration panel — all simulation inputs organized into collapsible sections.

Sections:
  1. Data Source      — ILI file, KMZ, TXT, or manual entry
  2. Pipeline         — NPS/WT segments, roughness, purge start/end
  3. Fluid            — type, SG, viscosity
  4. N2 Gas           — temperature, initial pressure, injection limits
  5. Pig Speed        — min/max/target, throttle down
  6. Exit Condition   — run/end pressure, behavior, MAOP
  7. Infrastructure   — check valves, pump stations, BPCV
  8. Gas Boosters     — stations, discharge/suction settings
  9. Spreads          — count, mob time
 10. Simulation       — timestep, adaptive dt
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional, List
import numpy as np

from ..engine.simulator import SimConfig
from ..engine.segment_model import PipeGeometry
from ..engine.check_valve import CheckValve
from ..engine.booster import BoosterConfig
from ..engine.pump_stations import PumpStationConfig
from ..engine.backward_pass import BackwardPassConfig
from ..engine.optimizer import OptimizerConfig
from ..data.scenario import ScenarioInputs
from ..data.ili_parser import parse_ili, ILIData
from ..data.profile_parser import parse_profile


class _CollapsibleSection(ttk.Frame):
    """A frame with a toggle button that shows/hides its content."""

    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, **kwargs)
        self._visible = True
        self._title = title

        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        self._toggle_btn = ttk.Button(header, text=f"▼ {title}", command=self._toggle,
                                       style='Toolbutton')
        self._toggle_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.content = ttk.Frame(self)
        self.content.pack(fill=tk.BOTH, padx=8, pady=4)

    def _toggle(self):
        if self._visible:
            self.content.pack_forget()
            self._toggle_btn.config(text=f"▶ {self._title}")
        else:
            self.content.pack(fill=tk.BOTH, padx=8, pady=4)
            self._toggle_btn.config(text=f"▼ {self._title}")
        self._visible = not self._visible


class ConfigPanel(ttk.Frame):
    """Left panel containing all simulation configuration inputs."""

    def __init__(self, parent, on_change: Callable = None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_change = on_change or (lambda: None)
        self._ili_data: Optional[ILIData] = None
        self._profile_data = None

        # Scrollable canvas
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)
        self._scroll_frame.bind("<Configure>",
                                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_sections()

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_sections(self):
        p = self._scroll_frame

        self._sec_data   = _CollapsibleSection(p, "Data Source")
        self._sec_data.pack(fill=tk.X, pady=2)
        self._build_data_section(self._sec_data.content)

        self._sec_pipe   = _CollapsibleSection(p, "Pipeline")
        self._sec_pipe.pack(fill=tk.X, pady=2)
        self._build_pipe_section(self._sec_pipe.content)

        self._sec_fluid  = _CollapsibleSection(p, "Fluid")
        self._sec_fluid.pack(fill=tk.X, pady=2)
        self._build_fluid_section(self._sec_fluid.content)

        self._sec_n2     = _CollapsibleSection(p, "N2 Gas")
        self._sec_n2.pack(fill=tk.X, pady=2)
        self._build_n2_section(self._sec_n2.content)

        self._sec_speed  = _CollapsibleSection(p, "Pig Speed")
        self._sec_speed.pack(fill=tk.X, pady=2)
        self._build_speed_section(self._sec_speed.content)

        self._sec_exit   = _CollapsibleSection(p, "Exit Condition")
        self._sec_exit.pack(fill=tk.X, pady=2)
        self._build_exit_section(self._sec_exit.content)

        self._sec_infra  = _CollapsibleSection(p, "Infrastructure")
        self._sec_infra.pack(fill=tk.X, pady=2)
        self._build_infra_section(self._sec_infra.content)

        self._sec_boost  = _CollapsibleSection(p, "Gas Boosters")
        self._sec_boost.pack(fill=tk.X, pady=2)
        self._build_booster_section(self._sec_boost.content)

        self._sec_spread = _CollapsibleSection(p, "Spreads & Logistics")
        self._sec_spread.pack(fill=tk.X, pady=2)
        self._build_spread_section(self._sec_spread.content)

        self._sec_sim    = _CollapsibleSection(p, "Simulation Control")
        self._sec_sim.pack(fill=tk.X, pady=2)
        self._build_sim_section(self._sec_sim.content)

    def _lf(self, parent, row, label, var, width=12, tooltip=""):
        ttk.Label(parent, text=label, width=22, anchor=tk.W).grid(row=row, column=0, sticky=tk.W, pady=1)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=1, sticky=tk.W, pady=1)
        return e

    def _build_data_section(self, f):
        self._data_source_var = tk.StringVar(value="ILI")
        for i, opt in enumerate(["ILI Excel", "KMZ", "TXT/CSV", "Client Excel", "Manual"]):
            ttk.Radiobutton(f, text=opt, variable=self._data_source_var, value=opt.split()[0],
                            command=self._on_data_source_change).grid(row=0, column=i, sticky=tk.W)

        self._data_file_var = tk.StringVar()
        ttk.Label(f, text="File:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(f, textvariable=self._data_file_var, width=28).grid(row=1, column=1, columnspan=3, sticky=tk.EW)
        ttk.Button(f, text="Browse…", command=self._on_browse_data).grid(row=1, column=4, padx=4)
        ttk.Button(f, text="Load", command=self._on_load_data).grid(row=1, column=5, padx=4)

        self._data_status_var = tk.StringVar(value="No file loaded")
        ttk.Label(f, textvariable=self._data_status_var, foreground="gray").grid(
            row=2, column=0, columnspan=6, sticky=tk.W)

    def _build_pipe_section(self, f):
        r = 0
        self._start_mp_var = tk.DoubleVar(value=0.0)
        self._end_mp_var   = tk.DoubleVar(value=236.0)
        self._lf(f, r, "Purge Start MP:", self._start_mp_var); r += 1
        self._lf(f, r, "Purge End MP:",   self._end_mp_var);   r += 1

        ttk.Label(f, text="Pipe Segments (OD, WT, start MP, end MP):").grid(
            row=r, column=0, columnspan=4, sticky=tk.W); r += 1
        self._seg_text = tk.Text(f, height=5, width=36)
        self._seg_text.grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=2)
        self._seg_text.insert(tk.END, "# OD_in, WT_in, start_mp, end_mp\n24.0, 0.313, 0.0, 103.28\n22.0, 0.313, 103.28, 207.21\n24.0, 0.313, 207.21, 236.0")

    def _build_fluid_section(self, f):
        r = 0
        self._fluid_var = tk.StringVar(value="Diesel")
        ttk.Label(f, text="Fluid type:", width=22, anchor=tk.W).grid(row=r, column=0, sticky=tk.W)
        fluid_cb = ttk.Combobox(f, textvariable=self._fluid_var,
                                 values=["Diesel", "Gasoline", "Crude Oil", "Water", "NGL"], width=14)
        fluid_cb.grid(row=r, column=1, sticky=tk.W); r += 1

        self._sg_var   = tk.DoubleVar(value=0.85)
        self._visc_var = tk.DoubleVar(value=2.7)
        self._rough_var = tk.DoubleVar(value=0.00015)
        self._lf(f, r, "Specific Gravity:", self._sg_var);    r += 1
        self._lf(f, r, "Viscosity (cSt):", self._visc_var);   r += 1
        self._lf(f, r, "Roughness (ft):",  self._rough_var);  r += 1

    def _build_n2_section(self, f):
        r = 0
        self._n2_temp_var       = tk.DoubleVar(value=45.0)
        self._max_inj_p_var     = tk.DoubleVar(value=200.0)
        self._max_inj_scfm_var  = tk.DoubleVar(value=5000.0)
        self._lf(f, r, "N2 Temperature (°F):",   self._n2_temp_var);       r += 1
        self._lf(f, r, "Max Injection (psig):",   self._max_inj_p_var);    r += 1
        self._lf(f, r, "Max Injection (SCFM):",   self._max_inj_scfm_var); r += 1
        ttk.Label(f, text="Startup N2 pressure is auto-calculated\n"
                           "from friction + elevation to first exit.",
                  foreground="gray", font=("", 8)).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=4); r += 1

    def _build_speed_section(self, f):
        r = 0
        self._min_speed_var = tk.DoubleVar(value=0.5)
        self._max_speed_var = tk.DoubleVar(value=6.0)
        self._tgt_speed_var = tk.DoubleVar(value=3.0)
        self._throttle_var  = tk.DoubleVar(value=5.0)
        self._lf(f, r, "Min Speed (mph):",    self._min_speed_var); r += 1
        self._lf(f, r, "Max Speed (mph):",    self._max_speed_var); r += 1
        self._lf(f, r, "Target Speed (mph):", self._tgt_speed_var); r += 1
        self._lf(f, r, "Throttle Down (mi):", self._throttle_var);  r += 1

    def _build_exit_section(self, f):
        r = 0
        self._exit_run_var   = tk.DoubleVar(value=50.0)
        self._exit_end_var   = tk.DoubleVar(value=10.0)
        self._maop_var       = tk.DoubleVar(value=1200.0)
        self._max_drive_var  = tk.DoubleVar(value=1000.0)
        self._exit_bhvr_var  = tk.StringVar(value="taper_last_n_miles")
        self._lf(f, r, "Exit P run (psig):",  self._exit_run_var);  r += 1
        self._lf(f, r, "Exit P end (psig):",  self._exit_end_var);  r += 1
        ttk.Label(f, text="Exit behavior:", width=22, anchor=tk.W).grid(row=r, column=0, sticky=tk.W)
        ttk.Combobox(f, textvariable=self._exit_bhvr_var, width=20,
                     values=["taper_last_n_miles", "step_last_n_miles",
                             "linear_ramp", "constant_run", "constant_end"]
                     ).grid(row=r, column=1, sticky=tk.W); r += 1
        self._lf(f, r, "MAOP (psig):",       self._maop_var);       r += 1
        self._lf(f, r, "Max Drive (psig):",   self._max_drive_var);  r += 1

    def _build_infra_section(self, f):
        ttk.Label(f, text="Check Valves — one per line: mp, name").pack(anchor=tk.W)
        self._cv_text = tk.Text(f, height=4, width=36)
        self._cv_text.pack(fill=tk.X, pady=2)
        self._cv_text.insert(tk.END, "# mp, name\n6.86, CV-6.86\n23.65, CV-23.65")

        ttk.Label(f, text="Pump Stations — one per line: mp, name, suction_psig").pack(anchor=tk.W, pady=(6,0))
        self._ps_text = tk.Text(f, height=4, width=36)
        self._ps_text.pack(fill=tk.X, pady=2)
        self._ps_text.insert(tk.END, "# mp, name, suction_psig\n26.45, Station A, 30\n51.62, Station B, 30")

        ttk.Label(f, text="BPCV: mp, elevation_ft, name (optional)").pack(anchor=tk.W, pady=(6,0))
        self._bpcv_var = tk.StringVar(value="205.64, 137.0, St Cesaire BPCV")
        ttk.Entry(f, textvariable=self._bpcv_var, width=36).pack(fill=tk.X, pady=2)

    def _build_booster_section(self, f):
        ttk.Label(f, text="Possible booster locations — one per line:\nmp, name").pack(anchor=tk.W)
        self._boost_text = tk.Text(f, height=5, width=36)
        self._boost_text.pack(fill=tk.X, pady=2)
        self._boost_text.insert(tk.END, "# mp, name\n")

    def _build_spread_section(self, f):
        r = 0
        self._n_spreads_var          = tk.IntVar(value=1)
        self._mob_time_var           = tk.DoubleVar(value=8.0)
        self._spread_discharge_var   = tk.DoubleVar(value=1200.0)
        self._spread_suction_min_var = tk.DoubleVar(value=150.0)
        self._spread_max_scfm_var    = tk.DoubleVar(value=15_000.0)
        self._lf(f, r, "# Spreads:",         self._n_spreads_var);          r += 1
        self._lf(f, r, "Mob Time (hr):",     self._mob_time_var);           r += 1
        self._lf(f, r, "Discharge (psig):",  self._spread_discharge_var);   r += 1
        self._lf(f, r, "Suction Min (psig):", self._spread_suction_min_var); r += 1
        self._lf(f, r, "Max Flow (SCFM):",   self._spread_max_scfm_var);    r += 1

    def _build_sim_section(self, f):
        r = 0
        self._dt_var      = tk.DoubleVar(value=1.0/60.0)
        self._adaptive_var = tk.BooleanVar(value=True)
        self._lf(f, r, "Timestep (hr):", self._dt_var); r += 1
        ttk.Checkbutton(f, text="Adaptive timestep", variable=self._adaptive_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W); r += 1

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_data_source_change(self):
        src = self._data_source_var.get()
        if src == "Manual":
            self._data_file_var.set("")
            self._data_status_var.set("Using manual pipe segment entry")

    def _on_browse_data(self):
        src = self._data_source_var.get()
        if src == "ILI":
            filetypes = [("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        elif src == "KMZ":
            filetypes = [("KMZ files", "*.kmz *.kml"), ("All files", "*.*")]
        else:
            filetypes = [("Data files", "*.txt *.csv *.xlsx *.xls"), ("All files", "*.*")]

        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self._data_file_var.set(path)

    def _on_load_data(self):
        src  = self._data_source_var.get()
        path = self._data_file_var.get()
        if not path:
            messagebox.showwarning("No File", "Select a file first.")
            return
        try:
            if src == "ILI":
                self._ili_data = parse_ili(path)
                self._populate_from_ili(self._ili_data)
                self._data_status_var.set(
                    f"ILI loaded: {len(self._ili_data.mop_joints):,} joints, "
                    f"{len(self._ili_data.pump_station_records)} stations, "
                    f"BPCV={'yes' if self._ili_data.bpcv_record else 'no'}"
                )
            else:
                self._profile_data = parse_profile(path)
                self._data_status_var.set(
                    f"Profile loaded: {len(self._profile_data.mileposts):,} pts, "
                    f"{self._profile_data.total_length_mi:.1f} mi"
                )
                self._end_mp_var.set(round(float(self._profile_data.end_mp), 2))
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._data_status_var.set(f"Error: {e}")

    def _populate_from_ili(self, data: ILIData):
        """Fill in infrastructure fields from ILI data."""
        self._end_mp_var.set(round(float(data.elevation_profile[-1, 0]), 2))

        # Pipe segments
        self._seg_text.delete("1.0", tk.END)
        self._seg_text.insert(tk.END, "# OD_in, WT_in, start_mp, end_mp\n")
        for s, e, od, wt in data.pipe_geometry.segments:
            self._seg_text.insert(tk.END, f"{od}, {wt}, {s:.3f}, {e:.3f}\n")

        # Check valves
        self._cv_text.delete("1.0", tk.END)
        self._cv_text.insert(tk.END, "# mp, name\n")
        for cv in data.check_valves():
            self._cv_text.insert(tk.END, f"{cv.mp:.4f}, {cv.name}\n")

        # Pump stations
        self._ps_text.delete("1.0", tk.END)
        self._ps_text.insert(tk.END, "# mp, name, suction_psig\n")
        for ps in data.pump_station_configs():
            self._ps_text.insert(tk.END, f"{ps.mp:.4f}, {ps.name}, 30\n")

        # BPCV
        if data.bpcv_record:
            bpcv = data.bpcv_record
            self._bpcv_var.set(f"{bpcv.mp:.4f}, {bpcv.elevation_ft:.1f}, {bpcv.site} BPCV")

    # ------------------------------------------------------------------
    # Input parsing helpers
    # ------------------------------------------------------------------

    def _parse_pipe_segments(self) -> PipeGeometry:
        geom = PipeGeometry()
        text = self._seg_text.get("1.0", tk.END)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                od, wt, s, e = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                geom.segments.append((s, e, od, wt))
        return geom

    def _parse_check_valves(self) -> List[CheckValve]:
        out = []
        text = self._cv_text.get("1.0", tk.END)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 1:
                mp   = float(parts[0])
                name = parts[1] if len(parts) > 1 else f"CV-{mp:.2f}"
                out.append(CheckValve(mp=mp, name=name))
        return out

    def _parse_pump_stations(self) -> List[PumpStationConfig]:
        out = []
        text = self._ps_text.get("1.0", tk.END)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 1:
                mp       = float(parts[0])
                name     = parts[1] if len(parts) > 1 else f"Station-{mp:.2f}"
                suction  = float(parts[2]) if len(parts) > 2 else 30.0
                out.append(PumpStationConfig(mp=mp, name=name, suction_psig=suction))
        return out

    def _parse_boosters(self) -> List[BoosterConfig]:
        discharge   = float(self._spread_discharge_var.get())
        suction_min = float(self._spread_suction_min_var.get())
        max_scfm    = float(self._spread_max_scfm_var.get())
        out = []
        text = self._boost_text.get("1.0", tk.END)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 1:
                mp   = float(parts[0])
                name = parts[1] if len(parts) > 1 else f"Booster-{mp:.2f}"
                out.append(BoosterConfig(
                    mp=mp, name=name,
                    discharge_psig=discharge,
                    suction_min_psig=suction_min,
                    max_flow_scfm=max_scfm,
                ))
        return out

    def _parse_bpcv(self):
        from ..engine.bpcv import BPCVConfig
        val = self._bpcv_var.get().strip()
        if not val or val.startswith('#'):
            return None
        if self._ili_data and self._ili_data.bpcv_record:
            return self._ili_data.bpcv_config(
                downstream_sg=float(self._sg_var.get()),
                downstream_viscosity_cst=float(self._visc_var.get()),
                downstream_roughness_ft=float(self._rough_var.get()),
            )
        parts = [p.strip() for p in val.split(',')]
        if len(parts) >= 2:
            mp   = float(parts[0])
            elev = float(parts[1])
            name = parts[2] if len(parts) > 2 else "BPCV"
            return BPCVConfig(mp=mp, elevation_ft=elev, name=name)
        return None

    def _elevation_profile(self) -> np.ndarray:
        if self._ili_data:
            return self._ili_data.elevation_profile
        if self._profile_data:
            return self._profile_data.elevation_profile_array()
        raise ValueError("No elevation data loaded. Load an ILI file, KMZ, or profile file first.")

    def _mop_joints(self):
        if self._ili_data:
            return self._ili_data.mop_joints
        return []

    # ------------------------------------------------------------------
    # Public API — called by MainWindow
    # ------------------------------------------------------------------

    def build_sim_config(self) -> SimConfig:
        geom = self._parse_pipe_segments()
        elev = self._elevation_profile()
        return SimConfig(
            pipe_geometry=geom,
            elevation_profile=elev,
            purge_start_mp=float(self._start_mp_var.get()),
            purge_end_mp=float(self._end_mp_var.get()),
            fluid_sg=float(self._sg_var.get()),
            fluid_viscosity_cst=float(self._visc_var.get()),
            fluid_roughness_ft=float(self._rough_var.get()),
            n2_temperature_f=float(self._n2_temp_var.get()),
            max_injection_psig=float(self._max_inj_p_var.get()),
            max_injection_scfm=float(self._max_inj_scfm_var.get()),
            exit_pressure_run_psig=float(self._exit_run_var.get()),
            exit_pressure_end_psig=float(self._exit_end_var.get()),
            exit_pressure_behavior=self._exit_bhvr_var.get(),
            throttle_down_miles=float(self._throttle_var.get()),
            min_speed_mph=float(self._min_speed_var.get()),
            max_speed_mph=float(self._max_speed_var.get()),
            target_speed_mph=float(self._tgt_speed_var.get()),
            maop_psig=float(self._maop_var.get()),
            max_drive_psig=float(self._max_drive_var.get()),
            mop_joints=self._mop_joints(),
            mop_warning_fraction=0.95,
            check_valves=self._parse_check_valves(),
            booster_configs=self._parse_boosters(),
            pump_stations=self._parse_pump_stations(),
            bpcv=self._parse_bpcv(),
            n_spreads=int(self._n_spreads_var.get()),
            mob_time_hr=float(self._mob_time_var.get()),
            dt_hr=float(self._dt_var.get()),
            adaptive_dt=bool(self._adaptive_var.get()),
        )

    def build_backward_pass_config(self) -> BackwardPassConfig:
        geom = self._parse_pipe_segments()
        return BackwardPassConfig(
            purge_start_mp=float(self._start_mp_var.get()),
            purge_end_mp=float(self._end_mp_var.get()),
            pipe_geometry=geom,
            n2_temperature_f=float(self._n2_temp_var.get()),
            endpoint_pressure_psig=float(self._exit_end_var.get()),
            booster_configs=self._parse_boosters(),
            check_valves=self._parse_check_valves(),
        )

    def build_optimizer_config(self, backward_pass=None) -> OptimizerConfig:
        return OptimizerConfig(
            n_spreads=int(self._n_spreads_var.get()),
            mob_time_hr=float(self._mob_time_var.get()),
            booster_configs=self._parse_boosters(),
            pig_speed_mph=float(self._tgt_speed_var.get()),
            backward_pass=backward_pass,
        )

    def collect_inputs(self) -> ScenarioInputs:
        """Collect all inputs into a ScenarioInputs object for saving."""
        geom = self._parse_pipe_segments()
        segs = [{'start_mp': s, 'end_mp': e, 'od_in': od, 'wt_in': wt}
                for s, e, od, wt in geom.segments]
        try:
            elev = self._elevation_profile()
            elev_list = elev.tolist()
        except Exception:
            elev_list = []

        cvs = [{'mp': v.mp, 'name': v.name, 'is_pump_station': v.is_pump_station}
               for v in self._parse_check_valves()]
        pss = [{'mp': p.mp, 'name': p.name, 'suction_psig': p.suction_psig}
               for p in self._parse_pump_stations()]
        bss = [{'mp': b.mp, 'name': b.name} for b in self._parse_boosters()]
        bpcv_cfg = self._parse_bpcv()
        bpcv_dict = ({'mp': bpcv_cfg.mp, 'elevation_ft': bpcv_cfg.elevation_ft, 'name': bpcv_cfg.name}
                     if bpcv_cfg else None)

        return ScenarioInputs(
            purge_start_mp=float(self._start_mp_var.get()),
            purge_end_mp=float(self._end_mp_var.get()),
            pipe_segments=segs,
            elevation_profile=elev_list,
            fluid_name=self._fluid_var.get(),
            fluid_sg=float(self._sg_var.get()),
            fluid_viscosity_cst=float(self._visc_var.get()),
            fluid_roughness_ft=float(self._rough_var.get()),
            n2_temperature_f=float(self._n2_temp_var.get()),
            max_injection_psig=float(self._max_inj_p_var.get()),
            max_injection_scfm=float(self._max_inj_scfm_var.get()),
            exit_pressure_run_psig=float(self._exit_run_var.get()),
            exit_pressure_end_psig=float(self._exit_end_var.get()),
            exit_pressure_behavior=self._exit_bhvr_var.get(),
            throttle_down_miles=float(self._throttle_var.get()),
            min_speed_mph=float(self._min_speed_var.get()),
            max_speed_mph=float(self._max_speed_var.get()),
            target_speed_mph=float(self._tgt_speed_var.get()),
            maop_psig=float(self._maop_var.get()),
            max_drive_psig=float(self._max_drive_var.get()),
            check_valves=cvs,
            pump_stations=pss,
            booster_stations=bss,
            n_spreads=int(self._n_spreads_var.get()),
            mob_time_hr=float(self._mob_time_var.get()),
            spread_discharge_psig=float(self._spread_discharge_var.get()),
            spread_suction_min_psig=float(self._spread_suction_min_var.get()),
            spread_max_flow_scfm=float(self._spread_max_scfm_var.get()),
            bpcv=bpcv_dict,
            dt_hr=float(self._dt_var.get()),
            adaptive_dt=bool(self._adaptive_var.get()),
        )

    def load_from_scenario(self, inputs: ScenarioInputs):
        """Populate all inputs from a loaded ScenarioInputs."""
        self._start_mp_var.set(inputs.purge_start_mp)
        self._end_mp_var.set(inputs.purge_end_mp)
        self._sg_var.set(inputs.fluid_sg)
        self._visc_var.set(inputs.fluid_viscosity_cst)
        self._rough_var.set(inputs.fluid_roughness_ft)
        self._fluid_var.set(inputs.fluid_name)
        self._n2_temp_var.set(inputs.n2_temperature_f)
        self._max_inj_p_var.set(inputs.max_injection_psig)
        self._max_inj_scfm_var.set(inputs.max_injection_scfm)
        self._exit_run_var.set(inputs.exit_pressure_run_psig)
        self._exit_end_var.set(inputs.exit_pressure_end_psig)
        self._exit_bhvr_var.set(inputs.exit_pressure_behavior)
        self._throttle_var.set(inputs.throttle_down_miles)
        self._min_speed_var.set(inputs.min_speed_mph)
        self._max_speed_var.set(inputs.max_speed_mph)
        self._tgt_speed_var.set(inputs.target_speed_mph)
        self._maop_var.set(inputs.maop_psig)
        self._max_drive_var.set(inputs.max_drive_psig)
        self._n_spreads_var.set(inputs.n_spreads)
        self._mob_time_var.set(inputs.mob_time_hr)
        self._dt_var.set(inputs.dt_hr)
        self._adaptive_var.set(inputs.adaptive_dt)

        # Pipe segments
        self._seg_text.delete("1.0", tk.END)
        self._seg_text.insert(tk.END, "# OD_in, WT_in, start_mp, end_mp\n")
        for s in inputs.pipe_segments:
            self._seg_text.insert(tk.END,
                f"{s['od_in']}, {s['wt_in']}, {s['start_mp']}, {s['end_mp']}\n")

        # Check valves
        self._cv_text.delete("1.0", tk.END)
        self._cv_text.insert(tk.END, "# mp, name\n")
        for cv in inputs.check_valves:
            self._cv_text.insert(tk.END, f"{cv['mp']}, {cv['name']}\n")

        # Pump stations
        self._ps_text.delete("1.0", tk.END)
        self._ps_text.insert(tk.END, "# mp, name, suction_psig\n")
        for ps in inputs.pump_stations:
            self._ps_text.insert(tk.END, f"{ps['mp']}, {ps['name']}, {ps['suction_psig']}\n")

        # Boosters (location only — specs come from spread settings)
        self._boost_text.delete("1.0", tk.END)
        self._boost_text.insert(tk.END, "# mp, name\n")
        for bs in inputs.booster_stations:
            self._boost_text.insert(tk.END, f"{bs['mp']}, {bs['name']}\n")

        # Spread specs
        self._spread_discharge_var.set(inputs.spread_discharge_psig)
        self._spread_suction_min_var.set(inputs.spread_suction_min_psig)
        self._spread_max_scfm_var.set(inputs.spread_max_flow_scfm)

        # BPCV
        if inputs.bpcv:
            b = inputs.bpcv
            self._bpcv_var.set(f"{b['mp']}, {b['elevation_ft']}, {b.get('name','BPCV')}")
        else:
            self._bpcv_var.set("")

        # Elevation (stored in scenario, set as profile_data placeholder)
        if inputs.elevation_profile:
            arr = np.array(inputs.elevation_profile, dtype=float)

            class _FakeProfile:
                def elevation_profile_array(self): return arr
            self._profile_data = _FakeProfile()
            self._ili_data = None
            self._data_status_var.set(
                f"Profile from scenario: {len(arr):,} pts")

    def reset(self):
        """Reset to blank state."""
        self._ili_data = None
        self._profile_data = None
        self._data_file_var.set("")
        self._data_status_var.set("No file loaded")
