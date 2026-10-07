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
from ..data.pxp_parser import parse_pxp
from ..data.formats import detect_format
from ..data.elevation import DEFAULT_SPACING_FT, ElevationError, fetch_route_elevation
from ..data.profile_parser import ProfileData, parse_profile


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
        self._mop_joints_cache: list = []   # populated from ILI or loaded scenario

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

        self._sec_plim   = _CollapsibleSection(p, "Pressure Limits")
        self._sec_plim.pack(fill=tk.X, pady=2)
        self._build_pressure_limits_section(self._sec_plim.content)

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

        self._sg_var   = tk.DoubleVar(value=0.815)
        self._visc_var = tk.DoubleVar(value=6.0)
        self._rough_var = tk.DoubleVar(value=0.00015)
        self._lf(f, r, "Specific Gravity:", self._sg_var);    r += 1
        self._lf(f, r, "Viscosity (cSt):", self._visc_var);   r += 1
        self._lf(f, r, "Roughness (ft):",  self._rough_var);  r += 1

    def _build_n2_section(self, f):
        r = 0
        self._n2_temp_var       = tk.DoubleVar(value=45.0)
        self._max_inj_scfm_var  = tk.DoubleVar(value=5000.0)
        self._lf(f, r, "N2 Temperature (°F):",   self._n2_temp_var);       r += 1
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
        self._exit_bhvr_var  = tk.StringVar(value="taper_last_n_miles")
        self._lf(f, r, "Exit P run (psig):",  self._exit_run_var);  r += 1
        self._lf(f, r, "Exit P end (psig):",  self._exit_end_var);  r += 1
        ttk.Label(f, text="Exit behavior:", width=22, anchor=tk.W).grid(row=r, column=0, sticky=tk.W)
        ttk.Combobox(f, textvariable=self._exit_bhvr_var, width=20,
                     values=["taper_last_n_miles", "step_last_n_miles",
                             "linear_ramp", "constant_run", "constant_end"]
                     ).grid(row=r, column=1, sticky=tk.W); r += 1

    def _build_pressure_limits_section(self, f):
        ttk.Label(
            f, text="Leave blank when an MOP profile is loaded — per-joint\n"
                    "limits apply. Only needed for profile-free ('dumb') purges.",
            foreground="gray", font=("", 8),
        ).pack(anchor=tk.W, pady=(0, 4))
        inner = ttk.Frame(f)
        inner.pack(fill=tk.X)
        r = 0
        self._maop_var      = tk.StringVar(value="")
        self._max_drive_var = tk.StringVar(value="")
        self._max_inj_p_var = tk.StringVar(value="")
        self._lf(inner, r, "MAOP (psig):",            self._maop_var);      r += 1
        self._lf(inner, r, "Max Drive (psig):",        self._max_drive_var); r += 1
        self._lf(inner, r, "Max Injection (psig):",    self._max_inj_p_var); r += 1

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
            fmt = detect_format(path)
            if src == "ILI" or fmt == "pxp":
                # The ILI choice also takes a point-by-point (PxP) pressure sheet; both
                # parse to the same ILIData. A PxP file picked under another source is
                # still read as one, since a profile parser would drop its MOP and stations.
                self._ili_data = parse_pxp(path) if fmt == "pxp" else parse_ili(path)
                if fmt == "pxp":
                    self._data_source_var.set("ILI")
                self._populate_from_ili(self._ili_data)
                self._data_status_var.set(
                    f"{self._ili_data.format} loaded: {len(self._ili_data.mop_joints):,} joints, "
                    f"{len(self._ili_data.pump_station_records)} stations, "
                    f"BPCV={'yes' if self._ili_data.bpcv_record else 'no'}"
                )
            else:
                prof = parse_profile(path)
                if prof.needs_elevation:
                    prof = self._lookup_elevation(prof)
                    if prof is None:
                        self._data_status_var.set("Not loaded: the file has no elevation data")
                        return
                self._profile_data = prof
                self._data_status_var.set(
                    f"Profile loaded: {len(self._profile_data.mileposts):,} pts, "
                    f"{self._profile_data.total_length_mi:.1f} mi"
                )
                self._end_mp_var.set(round(float(self._profile_data.end_mp), 2))
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._data_status_var.set(f"Error: {e}")

    def _lookup_elevation(self, prof):
        """A KMZ/GPS file without elevations: offer the USGS 3DEP lookup instead of a flat profile."""
        if prof.lat is None or prof.lon is None:
            messagebox.showerror("No Elevation", "This file has no elevations and no coordinates to look them up from.")
            return None
        if not messagebox.askyesno(
                "No Elevation Data",
                f"This file has no elevation data. Look up its {prof.total_length_mi:.1f} mi route in USGS 3DEP "
                f"(US only, {DEFAULT_SPACING_FT:g} ft spacing)?\n\nThis sends the route's coordinates to the USGS service."):
            return None

        def progress(frac, msg):
            self._data_status_var.set(f"{msg} ({frac * 100:.0f}%)")
            self.update_idletasks()

        try:
            res = fetch_route_elevation(prof.lat, prof.lon, progress_cb=progress)
        except ElevationError as e:
            messagebox.showerror("Elevation Lookup Failed", str(e))
            return None
        if res.flags:
            messagebox.showwarning("Elevation Lookup", "\n\n".join(res.flags))
        return ProfileData(mileposts=res.mileposts, elevations_ft=res.elevations_ft, lat=res.lat, lon=res.lon,
                           source=prof.source + "+USGS")

    def _populate_from_ili(self, data: ILIData):
        """Fill in infrastructure fields from ILI data."""
        self._end_mp_var.set(round(float(data.elevation_profile[-1, 0]), 2))
        if data.format == "PxP":
            # PxP mileposts are distance from the line's origin, so a section starts past 0
            self._start_mp_var.set(round(float(data.elevation_profile[0, 0]), 2))

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

    def _parse_optional_float(self, var: tk.StringVar) -> Optional[float]:
        """Return float value or None if the entry is blank."""
        v = var.get().strip()
        return float(v) if v else None

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
            if self._mop_joints_cache:
                from ..engine.bpcv import BCPVDownstreamJoint
                from ..engine.physics import fts_to_bph, mph_to_fts, pipe_area_ft2
                ds_joints = [
                    BCPVDownstreamJoint(mp=j.mp, mop_psig=j.mop_psig, elevation_ft=j.elevation_ft)
                    for j in self._mop_joints_cache if j.mp > mp
                ]
                geom = self._parse_pipe_segments()
                ds_od, ds_wt = geom.od_wt_at(mp + 0.1)
                ds_area = pipe_area_ft2(ds_od, ds_wt)
                return BPCVConfig(
                    mp=mp, elevation_ft=elev, name=name,
                    downstream_joints=ds_joints,
                    downstream_flow_bph=fts_to_bph(mph_to_fts(float(self._tgt_speed_var.get())), ds_area),
                    downstream_od_in=ds_od,
                    downstream_wt_in=ds_wt,
                    downstream_sg=float(self._sg_var.get()),
                    downstream_viscosity_cst=float(self._visc_var.get()),
                    downstream_roughness_ft=float(self._rough_var.get()),
                )
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
        return self._mop_joints_cache

    # ------------------------------------------------------------------
    # Public API — called by MainWindow
    # ------------------------------------------------------------------

    def build_sim_config(self) -> SimConfig:
        import math
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
            max_injection_psig=self._parse_optional_float(self._max_inj_p_var) or math.inf,
            max_injection_scfm=float(self._max_inj_scfm_var.get()),
            exit_pressure_run_psig=float(self._exit_run_var.get()),
            exit_pressure_end_psig=float(self._exit_end_var.get()),
            exit_pressure_behavior=self._exit_bhvr_var.get(),
            throttle_down_miles=float(self._throttle_var.get()),
            min_speed_mph=float(self._min_speed_var.get()),
            max_speed_mph=float(self._max_speed_var.get()),
            target_speed_mph=float(self._tgt_speed_var.get()),
            maop_psig=self._parse_optional_float(self._maop_var) or math.inf,
            max_drive_psig=self._parse_optional_float(self._max_drive_var) or math.inf,
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

        mop_joint_dicts = [
            {'mp': j.mp, 'mop_psig': j.mop_psig, 'elevation_ft': j.elevation_ft,
             'od_in': j.od_in, 'wt_in': j.wt_in}
            for j in self._mop_joints()
        ]
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
            max_injection_psig=self._parse_optional_float(self._max_inj_p_var),
            max_injection_scfm=float(self._max_inj_scfm_var.get()),
            exit_pressure_run_psig=float(self._exit_run_var.get()),
            exit_pressure_end_psig=float(self._exit_end_var.get()),
            exit_pressure_behavior=self._exit_bhvr_var.get(),
            throttle_down_miles=float(self._throttle_var.get()),
            min_speed_mph=float(self._min_speed_var.get()),
            max_speed_mph=float(self._max_speed_var.get()),
            target_speed_mph=float(self._tgt_speed_var.get()),
            maop_psig=self._parse_optional_float(self._maop_var),
            max_drive_psig=self._parse_optional_float(self._max_drive_var),
            check_valves=cvs,
            pump_stations=pss,
            booster_stations=bss,
            n_spreads=int(self._n_spreads_var.get()),
            mob_time_hr=float(self._mob_time_var.get()),
            spread_discharge_psig=float(self._spread_discharge_var.get()),
            spread_suction_min_psig=float(self._spread_suction_min_var.get()),
            spread_max_flow_scfm=float(self._spread_max_scfm_var.get()),
            bpcv=bpcv_dict,
            mop_joints=mop_joint_dicts,
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
        self._max_inj_scfm_var.set(inputs.max_injection_scfm)
        self._exit_run_var.set(inputs.exit_pressure_run_psig)
        self._exit_end_var.set(inputs.exit_pressure_end_psig)
        self._exit_bhvr_var.set(inputs.exit_pressure_behavior)
        self._throttle_var.set(inputs.throttle_down_miles)
        self._min_speed_var.set(inputs.min_speed_mph)
        self._max_speed_var.set(inputs.max_speed_mph)
        self._tgt_speed_var.set(inputs.target_speed_mph)
        self._maop_var.set(str(inputs.maop_psig) if inputs.maop_psig is not None else "")
        self._max_drive_var.set(str(inputs.max_drive_psig) if inputs.max_drive_psig is not None else "")
        self._max_inj_p_var.set(str(inputs.max_injection_psig) if inputs.max_injection_psig is not None else "")
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

        # MOP joints (from ILI, stored in scenario so headless runs work)
        if inputs.mop_joints:
            from ..engine.mop_check import MOPJoint
            self._mop_joints_cache = [
                MOPJoint(
                    mp=j['mp'], mop_psig=j['mop_psig'],
                    elevation_ft=j['elevation_ft'],
                    od_in=j.get('od_in', 24.0), wt_in=j.get('wt_in', 0.313),
                )
                for j in inputs.mop_joints
            ]
        else:
            self._mop_joints_cache = []

    def reset(self):
        """Reset to blank state."""
        self._ili_data = None
        self._profile_data = None
        self._mop_joints_cache = []
        self._data_file_var.set("")
        self._data_status_var.set("No file loaded")
