# Pipeline Pigging Simulation Program with Smart Enhancements (v29 - multi-booster, ILI parser, variable pipe)
import numpy as np
import pandas as pd
import zipfile
from lxml import etree as ET
from scipy.interpolate import CubicSpline
from geopy.distance import geodesic
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, simpledialog
import matplotlib.pyplot as plt
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except Exception:
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    Figure = None
import time
import logging
import unittest
import os
import sys
import math
import re

# --- Import-safe runtime setup (DO NOT execute UI/tests on import) ---
_LOGGING_CONFIGURED = False

def configure_logging(log_path: str = "purge_simulation.log", level: int = logging.INFO) -> None:
    """Configure logging once. Safe to call multiple times."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _LOGGING_CONFIGURED = True

def verify_tkinter() -> None:
    """Verify tkinter can initialize. Raises RuntimeError on failure."""
    try:
        tk.Tk().destroy()
        logging.info("Tkinter initialized successfully")
    except Exception as e:
        logging.error(f"Tkinter initialization failed: {str(e)}")
        raise RuntimeError(f"Tkinter initialization failed: {str(e)}") from e


# NPS data (Outer Diameter in inches)
nps_data = {
    "1/8": {"OD_in": 0.405}, "1/4": {"OD_in": 0.540}, "3/8": {"OD_in": 0.675}, "1/2": {"OD_in": 0.840},
    "3/4": {"OD_in": 1.050}, "1": {"OD_in": 1.315}, "1 1/4": {"OD_in": 1.660}, "1 1/2": {"OD_in": 1.900},
    "2": {"OD_in": 2.375}, "2 1/2": {"OD_in": 2.875}, "3": {"OD_in": 3.500}, "3 1/2": {"OD_in": 4.000},
    "4": {"OD_in": 4.500}, "5": {"OD_in": 5.563}, "6": {"OD_in": 6.625}, "8": {"OD_in": 8.625},
    "10": {"OD_in": 10.750}, "12": {"OD_in": 12.750}, "14": {"OD_in": 14.000}, "16": {"OD_in": 16.000},
    "18": {"OD_in": 18.000}, "20": {"OD_in": 20.000}, "24": {"OD_in": 24.000}, "26": {"OD_in": 26.000},
    "28": {"OD_in": 28.000}, "30": {"OD_in": 30.000}, "32": {"OD_in": 32.000}, "34": {"OD_in": 34.000},
    "36": {"OD_in": 36.000}, "40": {"OD_in": 40.000}, "42": {"OD_in": 42.000}, "44": {"OD_in": 44.000},
    "48": {"OD_in": 48.000}, "52": {"OD_in": 52.000}, "56": {"OD_in": 56.000}, "60": {"OD_in": 60.000},
    "64": {"OD_in": 64.000}, "68": {"OD_in": 68.000}, "72": {"OD_in": 72.000}, "80": {"OD_in": 80.000},}

# Roughness data (in feet)
roughness_data = {
    1: {"material": "New Welded Steel", "roughness_ft": 0.00015},
    2: {"material": "Rusted/Corroded Welded Steel", "roughness_ft": 0.0005},
    3: {"material": "Welded HDPE", "roughness_ft": 0.000005}
}

# Fluid data
fluid_data = {
    1: {"name": "Diesel", "sg": 0.84, "viscosity_cst": 2.7},
    2: {"name": "Gasoline", "sg": 0.74, "viscosity_cst": 0.6},
    3: {"name": "Crude Oil", "sg": None, "viscosity_cst": None},
    4: {"name": "Water", "sg": 1.0, "viscosity_cst": 1.0},
    5: {"name": "NGL (Y1-grade)", "sg": 0.6, "viscosity_cst": 0.3}
}

# Nitrogen properties
N2_MW_KG_PER_MOL = 0.0280134
N2_CRIT_T_K = 126.192
N2_CRIT_P_PA = 3.3958e6  # ~492.3 psia
N2_ACENTRIC = 0.0372
N2_TEMP_F_DEFAULT = 45.0
N2_VISCOSITY_PA_S = 1.75e-5
SCF_TO_FT3 = 1.0
ATM_PSI = 14.7

# Simple dicts — no dataclass import needed
# BoosterStation keys: milepost, discharge_psig, suction_min_psig, max_flow_scfm, name
# PipeSegment keys: start_mp, end_mp, nps, wall_thickness_in, roughness_num

# Column detection aliases for flexible TXT/CSV/ILI parser (case-insensitive)
_COL_LAT   = ['latitude', 'lat', 'gps_lat', 'gps latitude', 'gps lat']
_COL_LON   = ['longitude', 'lon', 'long', 'gps_lon', 'gps longitude', 'gps lon']
_COL_ELEV  = ['altitude (ft)', 'altitude', 'elevation', 'elev', 'alt', 'height',
               'altitude_ft', 'elevation_ft', 'elev_ft', 'altitude (m)', 'elevation (m)',
               'altitude(ft)', 'elevation(ft)', 'z', 'gps_alt', 'gps altitude']
_COL_MP    = ['milepost', 'log distance', 'log_distance', 'chainage', 'distance',
               'mp', 'station', 'odometer', 'dist_mi', 'dist_ft', 'km', 'meters',
               'log dist', 'log_dist']
_COL_OD    = ['od', 'outside diameter', 'pipe od', 'outer diameter', 'nominal od',
               'od_in', 'od (in)', 'diameter (in)', 'diameter_in', 'o.d.', 'o.d. (in)']
_COL_WT    = ['wall thickness', 'wt', 'wall_thickness', 'nominal wt', 'wt_in',
               'wt (in)', 'thickness', 'spec wt', 'wall thickness (in)', 'wall_thickness_in',
               'nom. wt', 'nom wt', 'spec. wt']


# ------------------------ Small utility helpers ------------------------
def resolve_nps_key(nps_val):
    """Map GUI 'nps' (str like '16' or float 16.0) to a key present in nps_data."""
    if nps_val in nps_data:
        return nps_val
    try:
        as_float = float(nps_val)
        cand = str(int(round(as_float)))
        if cand in nps_data:
            return cand
        for k in nps_data.keys():
            try:
                if abs(float(k) - as_float) < 1e-9:
                    return k
            except Exception:
                continue
    except Exception:
        pass
    k = str(nps_val)
    if k in nps_data:
        return k
    raise KeyError(f"NPS '{nps_val}' not found in nps_data; available keys: {list(nps_data.keys())}")

def print_inputs(inputs):
    print("\n=== Simulation Inputs ===")
    for key, value in sorted(inputs.items()):
        print(f"{key}: {value}")
    print("=== End Inputs ===\n")

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# ------------------------ KML/KMZ coordinate extraction helpers ------------------------
def _lname(tag):
    """Return localname for an XML tag (strip namespace)."""
    try:
        if isinstance(tag, str) and '}' in tag:
            return tag.split('}', 1)[1]
    except Exception:
        pass
    return tag

def _get_secure_lxml_parser():
    """Create an lxml XMLParser with safer defaults."""
    try:
        return ET.XMLParser(resolve_entities=False, no_network=True, recover=True, huge_tree=True)
    except TypeError:
        return ET.XMLParser(resolve_entities=False, recover=True)

def _parse_kml_coordinates_text(coord_text):
    """Parse standard KML <coordinates> text tokens 'lon,lat,alt'."""
    out = []
    for token in (coord_text or '').strip().split():
        try:
            parts = token.split(',')
            if len(parts) < 2:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            elev = float(parts[2]) if len(parts) > 2 and parts[2] != '' else 0.0
            out.append((lat, lon, elev))
        except Exception:
            continue
    return out

def _parse_gx_coord_text(coord_text):
    """Parse Google Earth gx:coord text 'lon lat alt'."""
    try:
        parts = (coord_text or '').strip().split()
        if len(parts) < 2:
            return None
        lon = float(parts[0])
        lat = float(parts[1])
        elev = float(parts[2]) if len(parts) > 2 else 0.0
        return (lat, lon, elev)
    except Exception:
        return None

def extract_pipeline_coords_from_kml_root(root, file_path=None, multiline_mode='longest', max_points=200000):
    """Extract a best-effort pipeline centerline coordinate sequence from a KML root."""
    linestring_candidates = []
    for ls in root.iter():
        if _lname(ls.tag) == 'LineString':
            for ch in ls.iter():
                if _lname(ch.tag) == 'coordinates' and getattr(ch, 'text', None) and ch.text.strip():
                    pts = _parse_kml_coordinates_text(ch.text)
                    if pts:
                        linestring_candidates.append(pts)
                    break

    if linestring_candidates:
        if multiline_mode == 'all':
            coords = [p for seg in linestring_candidates for p in seg]
        else:
            coords = max(linestring_candidates, key=len)
        try:
            logging.info(f"KML parse: found {len(linestring_candidates)} LineString(s); using {len(coords)} points (mode={multiline_mode})")
        except Exception:
            pass
        if max_points and len(coords) > max_points:
            step = max(1, int(len(coords) / max_points))
            coords = coords[::step]
        return coords

    track_candidates = []
    for tr in root.iter():
        if _lname(tr.tag) == 'Track':
            pts = []
            for ch in tr.iter():
                if _lname(ch.tag) == 'coord' and getattr(ch, 'text', None) and ch.text.strip():
                    p = _parse_gx_coord_text(ch.text)
                    if p is not None:
                        pts.append(p)
            if pts:
                track_candidates.append(pts)

    if track_candidates:
        coords = max(track_candidates, key=len)
        if max_points and len(coords) > max_points:
            step = max(1, int(len(coords) / max_points))
            coords = coords[::step]
        return coords

    fallback_candidates = []
    for ch in root.iter():
        if _lname(ch.tag) == 'coordinates' and getattr(ch, 'text', None) and ch.text.strip():
            skip = False
            try:
                parent = ch.getparent()
                while parent is not None:
                    ln = _lname(parent.tag)
                    if ln in ('Polygon', 'LinearRing'):
                        skip = True
                        break
                    parent = parent.getparent()
            except Exception:
                pass
            if skip:
                continue
            pts = _parse_kml_coordinates_text(ch.text)
            if pts:
                fallback_candidates.append(pts)

    if fallback_candidates:
        coords = max(fallback_candidates, key=len)
        if max_points and len(coords) > max_points:
            step = max(1, int(len(coords) / max_points))
            coords = coords[::step]
        return coords

    return []

def calculate_trendline_slope(mileposts, elevations, start_mp, end_mp):
    logging.debug(f"Calculating trendline slope from {start_mp} to {end_mp}")
    mask = (mileposts >= start_mp) & (mileposts <= end_mp)
    selected_mileposts = mileposts[mask]
    selected_elevations = elevations[mask]
    if len(selected_mileposts) < 2:
        logging.warning("Insufficient points for trendline slope calculation")
        return 0
    slope = (selected_elevations[-1] - selected_elevations[0]) / (selected_mileposts[-1] - selected_mileposts[0])
    logging.debug(f"Calculated slope: {slope}")
    return slope

def show_help():
    logging.info("Displaying help guide")
    help_text = """
Pipeline Pigging Simulation Program
==================================
- This version auto-adjusts N2 injection rate to maximize pig speed (<= Max) at a fixed exit pressure.
- Exit pressure tapers from 'run' to 'end' over the last N miles (Throttle Down).
- Friction uses Darcy-Weisbach (Swamee-Jain/Haaland), applied to remaining liquid slug length.
- Export fixes: NPS key is resolved robustly, avoiding the 16.0 KeyError.
- v29: Gas Booster Stations, ILI/CSV flexible parser, variable pipe segments.
    """
    try:
        help_window = tk.Toplevel()
        help_window.title("User Help")
        help_window.update()
        text = tk.Text(help_window, height=20, width=80)
        text.insert(tk.END, help_text)
        text.pack(padx=10, pady=10)
        tk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=5)
        logging.info("Help guide displayed")
    except Exception as e:
        logging.error(f"Failed to display help guide: {str(e)}")
        print(f"Error: Failed to display help guide: {str(e)}")
        raise


def get_user_inputs(dialog_root, profile_start_mp, profile_end_mp):
    logging.info("Entering get_user_inputs")
    inputs = {}
    booster_stations_list = []
    pipe_segments_list = []

    def validate_and_submit():
        try:
            nonlocal inputs
            logging.debug("Validating user inputs")
            purge_name = purge_name_entry.get().strip() or "Pipeline Purge"
            nps = (nps_var.get() or "").strip()
            if not nps:
                raise ValueError("Nominal Pipe Size must be selected.")
            pipe_wt = wt_entry.get().strip()
            if not pipe_wt:
                raise ValueError("Wall thickness must be provided.")
            pipe_wt = float(pipe_wt)
            pipe_od = nps_data[resolve_nps_key(nps)]["OD_in"]
            if pipe_wt <= 0 or pipe_wt >= pipe_od / 2:
                raise ValueError("Wall thickness must be positive and less than half the OD.")
            roughness_num = int(material_var.get().split(':')[0])
            if roughness_num not in roughness_data:
                raise ValueError("Invalid pipe material selected.")
            fluid_num = int(fluid_var.get().split(':')[0])
            if fluid_num not in fluid_data:
                raise ValueError("Invalid fluid selected.")
            api_gravity = float(api_entry.get()) if fluid_num == 3 else None
            viscosity_cst = float(viscosity_entry.get()) if fluid_num == 3 else None
            if fluid_num == 3:
                if api_gravity <= 0:
                    raise ValueError("API Gravity must be positive.")
                if viscosity_cst <= 0:
                    raise ValueError("Viscosity must be positive.")
            max_n2_rate_scfm = float(max_rate_entry.get()) if max_rate_entry.get().strip() else None
            n2_cutoff_scf = float(n2_cutoff_entry.get()) if n2_cutoff_entry.get().strip() else None
            exit_run = float(exit_run_entry.get())
            if exit_run < 0:
                raise ValueError("Minimum Exit Pressure (Run) must be non-negative.")
            exit_end = float(exit_end_entry.get())
            if exit_end < 0:
                raise ValueError("Exit Pressure (End) must be non-negative.")

            exit_behavior = exit_behavior_var.get().strip()
            if not exit_behavior:
                exit_behavior = "taper_last_n_miles"

            endpoint_constraint_mode = endpoint_constraint_mode_var.get().strip().lower()
            if not endpoint_constraint_mode:
                endpoint_constraint_mode = "none"
            if endpoint_constraint_mode not in ("none", "clamp_to_monitors"):
                endpoint_constraint_mode = "none"

            exit_pressure_min_clamp = float(exit_min_clamp_entry.get()) if exit_min_clamp_entry.get().strip() else None
            if exit_pressure_min_clamp is not None and exit_pressure_min_clamp < 0:
                raise ValueError("Exit Pressure Min Clamp must be non-negative if provided.")

            exit_pressure_max_clamp = float(exit_max_clamp_entry.get()) if exit_max_clamp_entry.get().strip() else None
            if exit_pressure_max_clamp is not None and exit_pressure_max_clamp < 0:
                raise ValueError("Exit Pressure Max Clamp must be non-negative if provided.")

            if (exit_pressure_min_clamp is not None) and (exit_pressure_max_clamp is not None) and (exit_pressure_min_clamp > exit_pressure_max_clamp):
                raise ValueError("Exit Pressure Min Clamp cannot exceed Exit Pressure Max Clamp.")

            exit_pressure_ramp_psi_per_hr = float(exit_ramp_entry.get()) if exit_ramp_entry.get().strip() else None
            if exit_pressure_ramp_psi_per_hr is not None and exit_pressure_ramp_psi_per_hr <= 0:
                raise ValueError("Exit Pressure Ramp Limit must be positive if provided.")

            max_outlet_flow_bph = float(max_outlet_flow_entry.get()) if max_outlet_flow_entry.get().strip() else None
            if max_outlet_flow_bph is not None and max_outlet_flow_bph <= 0:
                raise ValueError("Max Outlet Flow to Tankage must be positive if provided.")

            max_speed_from_flow_mph = None
            if max_outlet_flow_bph is not None:
                pipe_id_in = pipe_od - 2 * pipe_wt
                pipe_diameter_ft = pipe_id_in / 12.0
                area_ft2 = math.pi * (pipe_diameter_ft / 2.0) ** 2
                q_ft3_s = max_outlet_flow_bph * 5.614583333333333 / 3600.0
                v_ft_s = q_ft3_s / area_ft2
                max_speed_from_flow_mph = v_ft_s * 3600.0 / 5280.0

            slack_pressure_psig = float(slack_entry.get()) if slack_entry.get().strip() else 50.0
            if slack_pressure_psig < 0:
                raise ValueError("Slack Threshold must be non-negative.")

            maop_psig = float(maop_entry.get()) if maop_entry.get().strip() else None
            if maop_psig is not None and maop_psig <= 0:
                raise ValueError("MAOP must be positive if provided.")

            try:
                safe_drive_envelope_enabled = bool(safe_drive_env_var.get())
            except Exception:
                safe_drive_envelope_enabled = True

            try:
                static_block_in_assume_max_drive = bool(static_maxdrive_var.get())
            except Exception:
                static_block_in_assume_max_drive = True

            try:
                static_assume_respects_safe_envelope = not bool(static_ignore_env_var.get())
            except Exception:
                static_assume_respects_safe_envelope = True

            if maop_psig is None:
                safe_drive_envelope_enabled = False

            n2_end = float(n2_end_entry.get())
            if n2_end < 0:
                raise ValueError("Estimated nitrogen pressure must be non-negative.")
            max_speed = float(max_speed_entry.get())
            if max_speed <= 0:
                raise ValueError("Max Pig Speed must be positive.")
            min_speed = float(min_speed_entry.get())
            if min_speed < 0:
                raise ValueError("Min Pig Speed must be non-negative.")
            if max_speed <= min_speed / 1.25:
                raise ValueError("Max Pig Speed must be greater than Min Pig Speed / 1.25.")
            target_speed = float(target_speed_entry.get())
            if target_speed <= 0 or target_speed > max_speed:
                raise ValueError("Target Pig Speed must be positive and <= Max Pig Speed.")
            purge_start = float(purge_start_entry.get())
            purge_end = float(purge_end_entry.get())
            system_end = float(system_end_entry.get())
            if not (purge_start < purge_end <= system_end):
                raise ValueError("Mileposts must satisfy: Start < End <= System End.")
            if purge_start < profile_start_mp:
                logging.warning(f"Purge Start Milepost {purge_start:.1f} is below profile range {profile_start_mp:.1f}. Adjusting to {profile_start_mp:.1f}.")
                purge_start = profile_start_mp
            if system_end > profile_end_mp + 1e-6:
                logging.warning(f"System Endpoint Milepost {system_end:.3f} exceeds profile range {profile_end_mp:.3f}. Adjusting to {profile_end_mp:.3f}.")
                system_end = profile_end_mp
            if purge_end > system_end:
                logging.warning(f"Purge End Milepost {purge_end:.1f} exceeds System Endpoint {system_end:.1f}. Adjusting to {system_end:.1f}.")
                purge_end = system_end
            throttle_down = float(throttle_down_entry.get())
            if throttle_down < 0 or throttle_down > (purge_end - purge_start):
                raise ValueError(f"Throttle Down Point must be between 0 and {purge_end - purge_start:.1f} miles.")

            has_ips = ips_var.get()
            ips_mp = None
            ips_mps = []
            ips_shutdown_dist = None
            min_pump_suction_pressure = None
            min_pump_flow_bph = None
            ips_check_dp_psi = 5.0
            if has_ips:
                ips_mp = ips_mp_entry.get().strip()
                if not ips_mp:
                    raise ValueError("IPS milepost(s) must be provided.")
                ips_mps = parse_milepost_list(ips_mp)
                if not ips_mps:
                    raise ValueError("Could not parse IPS milepost(s). Use comma-separated mileposts.")
                for _mp in ips_mps:
                    if not (purge_start < _mp < system_end):
                        raise ValueError("Each IPS milepost must be between Purge Start and System End.")
                ips_shutdown_dist = ips_shutdown_entry.get().strip()
                if not ips_shutdown_dist:
                    raise ValueError("IPS shutdown distance must be provided.")
                ips_shutdown_dist = float(ips_shutdown_dist)
                if ips_shutdown_dist <= 0:
                    raise ValueError("IPS shutdown distance must be positive.")
                min_pump_suction_pressure = min_pump_suction_entry.get().strip()
                if not min_pump_suction_pressure:
                    raise ValueError("Minimum Pump Suction Pressure must be provided.")
                min_pump_suction_pressure = float(min_pump_suction_pressure)
                if min_pump_suction_pressure < 0:
                    raise ValueError("Minimum Pump Suction Pressure must be non-negative.")

                min_pump_flow_bph = None
                _mpf = min_pump_flow_entry.get().strip()
                if _mpf:
                    min_pump_flow_bph = float(_mpf)
                    if min_pump_flow_bph <= 0:
                        raise ValueError("Minimum Pump Flow must be positive if provided.")

                _cdp = ips_check_dp_entry.get().strip()
                ips_check_dp_psi = float(_cdp) if _cdp else 5.0
                if ips_check_dp_psi < 0:
                    raise ValueError("IPS check valve seal DP must be non-negative.")

            resolution = int(resolution_entry.get()) if resolution_entry.get().strip() else 500
            if resolution < 50 or resolution > 2000:
                raise ValueError("Resolution must be between 50 and 2000 points.")

            resample_profile_to_resolution = bool(resample_profile_var.get())

            max_nitrogen_pressure = float(max_nitrogen_pressure_entry.get())

            inputs.update({
                'nps': nps, 'pipe_wt': pipe_wt, 'roughness_num': roughness_num, 'fluid_num': fluid_num,
                'api_gravity': api_gravity, 'viscosity_cst': viscosity_cst, 'max_n2_rate_scfm': max_n2_rate_scfm,
                'exit_pressure_run': exit_run, 'exit_pressure_end': exit_end, 'exit_pressure_behavior': exit_behavior,
                'endpoint_constraint_mode': endpoint_constraint_mode, 'exit_pressure_min_clamp': exit_pressure_min_clamp,
                'exit_pressure_max_clamp': exit_pressure_max_clamp, 'exit_pressure_ramp_psi_per_hr': exit_pressure_ramp_psi_per_hr,
                'slack_pressure_psig': slack_pressure_psig, 'maop_psig': maop_psig,
                'safe_drive_envelope_enabled': safe_drive_envelope_enabled,
                'static_block_in_assume_max_drive': static_block_in_assume_max_drive,
                'static_assume_respects_safe_envelope': static_assume_respects_safe_envelope,
                'n2_end_pressure': n2_end,
                'max_pig_speed': max_speed, 'min_pig_speed': min_speed, 'target_pig_speed': target_speed,
                'purge_start_mp': purge_start, 'purge_end_mp': purge_end, 'system_end_mp': system_end,
                'throttle_down_miles': throttle_down,
                'elevation_format': elev_format_var.get(),
                'elevation_units': elev_units_var.get(), 'has_ips': has_ips, 'ips_mp': ips_mp, 'ips_mps': ips_mps,
                'ips_shutdown_dist': ips_shutdown_dist, 'min_pump_suction_pressure': min_pump_suction_pressure,
                'min_pump_flow_bph': min_pump_flow_bph, 'ips_check_dp_psi': ips_check_dp_psi,
                'max_outlet_flow_bph': max_outlet_flow_bph,
                'max_pig_speed_from_max_outlet_flow_mph': max_speed_from_flow_mph,
                'resolution': resolution,
                'resample_profile_to_resolution': resample_profile_to_resolution,
                'n2_cutoff_scf': n2_cutoff_scf,
                'max_nitrogen_pressure': max_nitrogen_pressure,
                'max_drive_pressure': max_nitrogen_pressure,
                'booster_stations': list(booster_stations_list),
                'pipe_segments': list(pipe_segments_list),
                'purge_name': purge_name,
            })
            inputs.setdefault('temperature_f', float(N2_TEMP_F_DEFAULT))
            logging.info(f"Validated inputs: NPS={nps}, Target Speed={target_speed} mph, Purge Start={purge_start}, Purge End={purge_end}, Resolution={resolution}, Boosters={len(booster_stations_list)}")
            win.destroy()
        except ValueError as e:
            logging.error(f"Input validation failed: {str(e)}")
            messagebox.showerror("Input Error", str(e), parent=win)

    try:
        logging.info("Initializing Tkinter root for input GUI")
        win = tk.Toplevel(dialog_root)
        logging.info("Tkinter input GUI created")
        win.title("Pipeline Pigging Simulation Inputs")
        try:
            win.geometry("600x800")
        except Exception:
            pass
        win.minsize(520, 600)

        win.lift()
        win.attributes('-topmost', True)
        win.after(250, lambda: win.attributes('-topmost', False))
        win.update_idletasks()

        container = ttk.Frame(win)
        container.grid(row=0, column=0, sticky="nsew")
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
        vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        root = ttk.Frame(canvas)
        _form_window = canvas.create_window((0, 0), window=root, anchor="nw")

        def _sync_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event):
            canvas.itemconfigure(_form_window, width=event.width)

        root.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_width)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        def _on_mousewheel_linux(event):
            try:
                if event.num == 4:
                    canvas.yview_scroll(-3, "units")
                elif event.num == 5:
                    canvas.yview_scroll(3, "units")
            except Exception:
                pass

        win.bind("<MouseWheel>", _on_mousewheel)
        win.bind("<Button-4>", _on_mousewheel_linux)
        win.bind("<Button-5>", _on_mousewheel_linux)

        messagebox.showinfo(
            "Tip",
            "This input window is scrollable. Use the mouse wheel or the scrollbar on the right to reach Submit.",
            parent=win
        )
    except Exception as e:
        logging.error(f"Failed to initialize Tkinter input GUI: {str(e)}")
        print(f"Error: Failed to initialize Tkinter input GUI: {str(e)}")
        raise

    row = 0
    tk.Label(root, text="Purge Description / Title:").grid(row=row, column=0, sticky='e')
    purge_name_entry = tk.Entry(root, width=35)
    purge_name_entry.insert(0, "Pipeline Purge")
    purge_name_entry.grid(row=row, column=1, sticky='w')
    row += 1

    tk.Label(root, text="Nominal Pipe Size (NPS):").grid(row=row, column=0, sticky='e')
    nps_var = tk.StringVar(value="")
    nps_values = [""] + list(nps_data.keys())
    nps_combo = ttk.Combobox(root, textvariable=nps_var, values=nps_values, state="readonly")
    nps_combo.grid(row=row, column=1, sticky='w')
    row += 1

    tk.Label(root, text="Wall Thickness (in):").grid(row=row, column=0, sticky='e')
    wt_entry = tk.Entry(root)
    wt_entry.insert(0, "0.280")
    wt_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Pipe Material:").grid(row=row, column=0, sticky='e')
    material_var = tk.StringVar(value="2: Rusted/Corroded Welded Steel")
    material_options = [f"{k}: {v['material']}" for k, v in roughness_data.items()]
    ttk.Combobox(root, textvariable=material_var, values=material_options).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Fluid Type:").grid(row=row, column=0, sticky='e')
    fluid_var = tk.StringVar(value="3: Crude Oil")
    fluid_options = [f"{k}: {v['name']}" for k, v in fluid_data.items()]
    ttk.Combobox(root, textvariable=fluid_var, values=fluid_options).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="API Gravity (if Crude Oil):").grid(row=row, column=0, sticky='e')
    api_entry = tk.Entry(root)
    api_entry.insert(0, "25")
    api_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Viscosity (cSt) (if Crude Oil):").grid(row=row, column=0, sticky='e')
    viscosity_entry = tk.Entry(root)
    viscosity_entry.insert(0, "200")
    viscosity_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Min Exit Pressure (Run) (psi):").grid(row=row, column=0, sticky='e')
    exit_run_entry = tk.Entry(root)
    exit_run_entry.insert(0, "100")
    exit_run_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Exit Pressure (End) (psi):").grid(row=row, column=0, sticky='e')
    exit_end_entry = tk.Entry(root)
    exit_end_entry.insert(0, "100")
    exit_end_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Exit Pressure Behavior:").grid(row=row, column=0, sticky='e')
    exit_behavior_var = tk.StringVar(value="taper_last_n_miles")
    exit_behavior_combo = ttk.Combobox(
        root,
        textvariable=exit_behavior_var,
        state="readonly",
        values=["taper_last_n_miles", "step_last_n_miles", "linear_ramp", "constant_run", "constant_end"]
    )
    exit_behavior_combo.grid(row=row, column=1, sticky='w')
    row += 1

    tk.Label(root, text="Endpoint Constraint Mode:").grid(row=row, column=0, sticky='e')
    endpoint_constraint_mode_var = tk.StringVar(value="none")
    endpoint_constraint_combo = ttk.Combobox(
        root,
        textvariable=endpoint_constraint_mode_var,
        state="readonly",
        values=["none", "clamp_to_monitors"]
    )
    endpoint_constraint_combo.grid(row=row, column=1, sticky='w')
    row += 1

    tk.Label(root, text="Exit Pressure Min Clamp (psig) [optional]:").grid(row=row, column=0, sticky='e')
    exit_min_clamp_entry = tk.Entry(root)
    exit_min_clamp_entry.insert(0, "")
    exit_min_clamp_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Exit Pressure Max Clamp (psig) [optional]:").grid(row=row, column=0, sticky='e')
    exit_max_clamp_entry = tk.Entry(root)
    exit_max_clamp_entry.insert(0, "")
    exit_max_clamp_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Exit Pressure Ramp Limit (psi/hr) [optional]:").grid(row=row, column=0, sticky='e')
    exit_ramp_entry = tk.Entry(root)
    exit_ramp_entry.insert(0, "")
    exit_ramp_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Max Outlet Flow to Tankage (BPH) [optional]:").grid(row=row, column=0, sticky='e')
    max_outlet_flow_entry = tk.Entry(root)
    max_outlet_flow_entry.insert(0, "")
    max_outlet_flow_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Slack Threshold (psig):").grid(row=row, column=0, sticky='e')
    slack_entry = tk.Entry(root)
    slack_entry.insert(0, "50")
    slack_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="MAOP (psig) (optional):").grid(row=row, column=0, sticky='e')
    maop_entry = tk.Entry(root)
    maop_entry.insert(0, "")
    maop_entry.grid(row=row, column=1)
    row += 1

    safe_drive_env_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        root,
        text="Safe Drive Envelope (limit drive pressure to avoid static MAOP exceed)",
        variable=safe_drive_env_var
    ).grid(row=row, column=0, columnspan=2, sticky='w')
    row += 1

    static_maxdrive_var = tk.BooleanVar(value=True)
    static_maxdrive_cb = tk.Checkbutton(
        root,
        text="Static Assume Max Drive (conservative block-in MAOP check)",
        variable=static_maxdrive_var
    )
    static_maxdrive_cb.grid(row=row, column=0, columnspan=2, sticky='w')
    row += 1

    static_ignore_env_var = tk.BooleanVar(value=False)
    static_ignore_env_cb = tk.Checkbutton(
        root,
        text="Static WORST-CASE: ignore Safe Drive Envelope (stress-test operator scenarios)",
        variable=static_ignore_env_var
    )
    static_ignore_env_cb.grid(row=row, column=0, columnspan=2, sticky='w')
    row += 1

    def _update_static_ignore_state(*_args):
        if static_maxdrive_var.get():
            static_ignore_env_cb.configure(state="normal")
        else:
            static_ignore_env_var.set(False)
            static_ignore_env_cb.configure(state="disabled")

    try:
        static_maxdrive_var.trace_add("write", _update_static_ignore_state)
    except Exception:
        static_maxdrive_var.trace("w", lambda *_a: _update_static_ignore_state())

    _update_static_ignore_state()

    tk.Label(root, text="Est. N2 Pressure at End (psi):").grid(row=row, column=0, sticky='e')
    n2_end_entry = tk.Entry(root)
    n2_end_entry.insert(0, "300")
    n2_end_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Max Nitrogen Pressure (psi):").grid(row=row, column=0, sticky='e')
    max_nitrogen_pressure_entry = tk.Entry(root)
    max_nitrogen_pressure_entry.insert(0, "400")
    max_nitrogen_pressure_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Max N2 Rate (SCFM) [optional]:").grid(row=row, column=0, sticky='e')
    max_rate_entry = tk.Entry(root)
    max_rate_entry.insert(0, "")
    max_rate_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="N2 Cutoff Volume (SCF) [optional]:").grid(row=row, column=0, sticky='e')
    n2_cutoff_entry = tk.Entry(root)
    n2_cutoff_entry.insert(0, "")
    n2_cutoff_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Max Pig Speed (mph):").grid(row=row, column=0, sticky='e')
    max_speed_entry = tk.Entry(root)
    max_speed_entry.insert(0, "4")
    max_speed_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Min Pig Speed (mph):").grid(row=row, column=0, sticky='e')
    min_speed_entry = tk.Entry(root)
    min_speed_entry.insert(0, "1")
    min_speed_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Target Pig Speed (mph):").grid(row=row, column=0, sticky='e')
    target_speed_entry = tk.Entry(root)
    target_speed_entry.insert(0, "2")
    target_speed_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text=f"Purge Start Milepost (min {profile_start_mp:.1f}):").grid(row=row, column=0, sticky='e')
    purge_start_entry = tk.Entry(root)
    purge_start_entry.insert(0, f"{profile_start_mp:.1f}")
    purge_start_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text=f"Purge End Milepost (max {profile_end_mp:.1f}):").grid(row=row, column=0, sticky='e')
    purge_end_entry = tk.Entry(root)
    purge_end_entry.insert(0, f"{profile_end_mp:.1f}")
    purge_end_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text=f"System Endpoint Milepost (max {profile_end_mp:.1f}):").grid(row=row, column=0, sticky='e')
    system_end_entry = tk.Entry(root)
    system_end_entry.insert(0, f"{profile_end_mp:.1f}")
    system_end_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Throttle Down Point (miles from end):").grid(row=row, column=0, sticky='e')
    throttle_down_entry = tk.Entry(root)
    throttle_down_entry.insert(0, "9.6")
    throttle_down_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Is there an Intermediate Pump Station?").grid(row=row, column=0, sticky='e')
    ips_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, variable=ips_var).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="IPS Milepost(s) (comma-separated):").grid(row=row, column=0, sticky='e')
    ips_mp_entry = tk.Entry(root)
    ips_mp_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="IPS Shutdown Distance (miles):").grid(row=row, column=0, sticky='e')
    ips_shutdown_entry = tk.Entry(root)
    ips_shutdown_entry.insert(0, "1")
    ips_shutdown_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Minimum Pump Suction Pressure (psi):").grid(row=row, column=0, sticky='e')
    min_pump_suction_entry = tk.Entry(root)
    min_pump_suction_entry.insert(0, "50")
    min_pump_suction_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Minimum Pump Flow (BPH) [optional]:").grid(row=row, column=0, sticky='e')
    min_pump_flow_entry = tk.Entry(root)
    min_pump_flow_entry.grid(row=row, column=1)
    row += 1

    tk.Label(root, text="IPS Check Valve Seal DP (psi) [optional]:").grid(row=row, column=0, sticky='e')
    ips_check_dp_entry = tk.Entry(root)
    ips_check_dp_entry.insert(0, "5")
    ips_check_dp_entry.grid(row=row, column=1)
    row += 1

    # --- Gas Booster Stations section ---
    tk.Label(root, text="Gas Booster Stations (N2 compressors in gas column):").grid(row=row, column=0, columnspan=2, sticky='w')
    row += 1

    booster_listbox = tk.Listbox(root, height=4, width=55)
    booster_listbox.grid(row=row, column=0, columnspan=2, sticky='w', padx=4)
    row += 1

    def _refresh_booster_list():
        booster_listbox.delete(0, tk.END)
        for b in booster_stations_list:
            nm = b.get('name', '')
            booster_listbox.insert(tk.END, f"MP {b['milepost']:.2f}: {b['discharge_psig']:.0f} psig disch, {b.get('suction_min_psig',0):.0f} psig suct_min, {b.get('max_flow_scfm','unlim')} scfm max" + (f" [{nm}]" if nm else ""))

    def _add_booster():
        dlg = tk.Toplevel(win)
        dlg.title("Add Gas Booster Station")
        dlg.grab_set()
        fields = {}
        defs = [
            ("Milepost", "0.0"),
            ("Discharge Pressure (psig)", "350"),
            ("Suction Min Pressure (psig)", "50"),
            ("Max Flow (SCFM, blank=unlimited)", ""),
            ("Name (optional)", ""),
        ]
        for r, (lbl, default) in enumerate(defs):
            tk.Label(dlg, text=lbl + ":").grid(row=r, column=0, sticky='e', padx=4, pady=2)
            e = tk.Entry(dlg)
            e.insert(0, default)
            e.grid(row=r, column=1, padx=4, pady=2)
            fields[lbl] = e

        def _ok():
            try:
                mp = float(fields["Milepost"].get())
                disch = float(fields["Discharge Pressure (psig)"].get())
                suct_min = float(fields["Suction Min Pressure (psig)"].get())
                mf_str = fields["Max Flow (SCFM, blank=unlimited)"].get().strip()
                mf = float(mf_str) if mf_str else None
                nm = fields["Name (optional)"].get().strip()
                booster_stations_list.append({
                    'milepost': mp,
                    'discharge_psig': disch,
                    'suction_min_psig': suct_min,
                    'max_flow_scfm': mf,
                    'name': nm,
                })
                booster_stations_list.sort(key=lambda b: float(b['milepost']))
                _refresh_booster_list()
                dlg.destroy()
            except ValueError as ex:
                messagebox.showerror("Error", str(ex), parent=dlg)

        tk.Button(dlg, text="OK", command=_ok).grid(row=len(defs), column=0, pady=6)
        tk.Button(dlg, text="Cancel", command=dlg.destroy).grid(row=len(defs), column=1, pady=6)
        dlg.wait_window()

    def _clear_boosters():
        booster_stations_list.clear()
        _refresh_booster_list()

    btn_frame_bst = ttk.Frame(root)
    btn_frame_bst.grid(row=row, column=0, columnspan=2, sticky='w')
    tk.Button(btn_frame_bst, text="Add Booster", command=_add_booster).pack(side='left', padx=2)
    tk.Button(btn_frame_bst, text="Clear Boosters", command=_clear_boosters).pack(side='left', padx=2)
    row += 1

    # --- Pipe Segments (variable NPS/WT) section ---
    tk.Label(root, text="Pipe Segments - Variable NPS/WT (optional):").grid(row=row, column=0, columnspan=2, sticky='w')
    row += 1

    seg_listbox = tk.Listbox(root, height=4, width=55)
    seg_listbox.grid(row=row, column=0, columnspan=2, sticky='w', padx=4)
    row += 1

    def _refresh_seg_list():
        seg_listbox.delete(0, tk.END)
        for s in pipe_segments_list:
            seg_listbox.insert(tk.END, f"MP {s['start_mp']:.2f}-{s['end_mp']:.2f}: NPS {s['nps']}, WT {s['wall_thickness_in']}, Rgh #{s.get('roughness_num',1)}")

    def _add_pipe_seg():
        dlg = tk.Toplevel(win)
        dlg.title("Add Pipe Segment")
        dlg.grab_set()
        fields = {}
        defs = [
            ("Start Milepost", "0.0"),
            ("End Milepost", "10.0"),
            ("NPS", "16"),
            ("Wall Thickness (in)", "0.312"),
            ("Roughness # (1=new steel, 2=corroded, 3=HDPE)", "1"),
        ]
        for r2, (lbl, default) in enumerate(defs):
            tk.Label(dlg, text=lbl + ":").grid(row=r2, column=0, sticky='e', padx=4, pady=2)
            e = tk.Entry(dlg)
            e.insert(0, default)
            e.grid(row=r2, column=1, padx=4, pady=2)
            fields[lbl] = e

        def _ok_seg():
            try:
                smp = float(fields["Start Milepost"].get())
                emp = float(fields["End Milepost"].get())
                nps_s = fields["NPS"].get().strip()
                wt_s = float(fields["Wall Thickness (in)"].get())
                rnum = int(fields["Roughness # (1=new steel, 2=corroded, 3=HDPE)"].get())
                if smp >= emp:
                    raise ValueError("Start MP must be less than End MP.")
                resolve_nps_key(nps_s)  # validate
                pipe_segments_list.append({
                    'start_mp': smp,
                    'end_mp': emp,
                    'nps': nps_s,
                    'wall_thickness_in': wt_s,
                    'roughness_num': rnum,
                })
                pipe_segments_list.sort(key=lambda s: float(s['start_mp']))
                _refresh_seg_list()
                dlg.destroy()
            except (ValueError, KeyError) as ex:
                messagebox.showerror("Error", str(ex), parent=dlg)

        tk.Button(dlg, text="OK", command=_ok_seg).grid(row=len(defs), column=0, pady=6)
        tk.Button(dlg, text="Cancel", command=dlg.destroy).grid(row=len(defs), column=1, pady=6)
        dlg.wait_window()

    def _clear_segs():
        pipe_segments_list.clear()
        _refresh_seg_list()

    btn_frame_seg = ttk.Frame(root)
    btn_frame_seg.grid(row=row, column=0, columnspan=2, sticky='w')
    tk.Button(btn_frame_seg, text="Add Segment", command=_add_pipe_seg).pack(side='left', padx=2)
    tk.Button(btn_frame_seg, text="Clear Segments", command=_clear_segs).pack(side='left', padx=2)
    row += 1

    tk.Label(root, text="Elevation Profile Format:").grid(row=row, column=0, sticky='e')
    elev_format_var = tk.StringVar(value="TXT")
    ttk.Combobox(root, textvariable=elev_format_var, values=["KMZ/KML", "Excel", "TXT"]).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Elevation Units:").grid(row=row, column=0, sticky='e')
    elev_units_var = tk.StringVar(value="Feet")
    ttk.Combobox(root, textvariable=elev_units_var, values=["Feet", "Meters"]).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Resolution (points):").grid(row=row, column=0, sticky='e')
    resolution_entry = tk.Entry(root)
    resolution_entry.insert(0, "500")
    resolution_entry.grid(row=row, column=1)
    row += 1

    resample_profile_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        root,
        text="Resample purge segment to Resolution (recommended)",
        variable=resample_profile_var
    ).grid(row=row, column=0, columnspan=2, sticky='w')
    row += 1

    tk.Button(root, text="Submit", command=validate_and_submit).grid(row=row, column=0, columnspan=2)
    tk.Button(root, text="Help", command=show_help).grid(row=row, column=1, sticky='e')

    logging.info("Starting Tkinter wait for input GUI")
    win.grab_set()
    win.wait_window()

    logging.info("Exiting get_user_inputs")
    return inputs


def calculate_friction_loss(v, L, D, fluid_density, viscosity, roughness, purged_fraction=0):
    rho_m = fluid_density / 32.174
    mu_eff = viscosity
    try:
        Re = rho_m * v * D / max(mu_eff, 1e-12)
        haaland = -1.8 * np.log10((roughness / D / 3.7)**1.11 + 6.9 / max(Re, 1e-12))
        f = (1.0 / haaland)**2
        dp_psf = f * (L / D) * 0.5 * rho_m * v * v
        friction_loss = dp_psf / 144.0
        return friction_loss
    except Exception as e:
        logging.error(f"Friction loss calculation failed: {str(e)}")
        raise ValueError(f"Error in friction loss calculation: {str(e)}")

def parse_milepost_list(val):
    """Parse a milepost list from a GUI string or pass-through list/tuple."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        out = []
        for x in val:
            try:
                out.append(float(x))
            except Exception:
                continue
        return out
    s = str(val).strip()
    if not s:
        return []
    parts = re.split(r"[;,\s]+", s)
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


# ------------------------ Conservative MAOP drive envelope & N2 mass helpers ------------------------
SCF_PER_LBMOL_STD_60F = 379.482
N2_MW_LBM_PER_LBMOL = 28.0134
N2_LBM_PER_SCF_STD = N2_MW_LBM_PER_LBMOL / SCF_PER_LBMOL_STD_60F
N2_KG_PER_SCF_STD = N2_LBM_PER_SCF_STD * 0.45359237


def n2_mass_lbm_from_scf(scf: float) -> float:
    try:
        return float(scf) * float(N2_LBM_PER_SCF_STD)
    except Exception:
        return 0.0


def n2_mass_kg_from_scf(scf: float) -> float:
    try:
        return float(scf) * float(N2_KG_PER_SCF_STD)
    except Exception:
        return 0.0


def compute_safe_max_drive_envelope(
    pig_mileposts,
    pig_elevations_ft,
    system_mileposts,
    system_elevations_ft,
    fluid_gamma_lbft3,
    maop_psig,
    max_drive_psig,
):
    """Compute conservative safe max behind-pig gas pressure vs pig position (psig)."""
    try:
        maop = float(maop_psig)
    except Exception:
        return np.full(len(pig_mileposts), float(max_drive_psig), dtype=float)

    if (not np.isfinite(maop)) or maop <= 0:
        return np.full(len(pig_mileposts), float(max_drive_psig), dtype=float)

    try:
        max_drive = float(max_drive_psig)
    except Exception:
        max_drive = maop

    grad = float(fluid_gamma_lbft3) / 144.0

    sys_mps = np.asarray(system_mileposts, dtype=float)
    sys_elev = np.asarray(system_elevations_ft, dtype=float)
    if sys_mps.size == 0:
        return np.full(len(pig_mileposts), min(max_drive, maop), dtype=float)

    min_elev_from = np.minimum.accumulate(sys_elev[::-1])[::-1]

    pig_mps = np.asarray(pig_mileposts, dtype=float)
    pig_elev = np.asarray(pig_elevations_ft, dtype=float)

    out = np.full_like(pig_mps, np.nan, dtype=float)
    for i, mp in enumerate(pig_mps):
        j = int(np.searchsorted(sys_mps, mp, side="left"))
        if j < 0:
            j = 0
        if j >= len(min_elev_from):
            j = len(min_elev_from) - 1

        min_ahead = float(min_elev_from[j])
        head_to_min = grad * (float(pig_elev[i]) - min_ahead)
        safe = maop - max(0.0, head_to_min)
        safe = max(0.0, min(float(safe), float(maop), float(max_drive)))
        out[i] = safe

    return out

def target_exit_pressure(mp, inputs, purge_start, purge_end, throttle_down, t_hours=None):
    """Return outlet/endpoint pressure (psig) based on a selectable behavior program."""
    run_p = float(inputs.get('exit_pressure_run', 0.0))
    end_p = float(inputs.get('exit_pressure_end', run_p))
    behavior = str(inputs.get('exit_pressure_behavior', 'taper_last_n_miles')).strip().lower()

    aliases = {
        'taper': 'taper_last_n_miles',
        'taper_last': 'taper_last_n_miles',
        'linear': 'linear_ramp',
        'ramp': 'linear_ramp',
        'hold_run': 'constant_run',
        'hold_end': 'constant_end',
        'step': 'step_last_n_miles',
    }
    behavior = aliases.get(behavior, behavior)

    purge_len = max(1e-12, float(purge_end) - float(purge_start))
    dist_from_end = float(purge_end) - float(mp)

    if behavior == 'constant_end':
        return end_p
    if behavior == 'constant_run':
        return run_p

    if behavior == 'linear_ramp':
        frac = (float(mp) - float(purge_start)) / purge_len
        frac = max(0.0, min(1.0, frac))
        return run_p * (1.0 - frac) + end_p * frac

    if behavior == 'step_last_n_miles':
        td = max(0.0, float(throttle_down or 0.0))
        if td > 0.0 and dist_from_end <= td:
            return end_p
        return run_p

    td = max(0.0, float(throttle_down or 0.0))
    if td > 0.0 and dist_from_end <= td:
        taper_factor = max(0.0, min(1.0, dist_from_end / td))
        return run_p * taper_factor + end_p * (1.0 - taper_factor)
    return run_p


def _local_extrema_indices(y: np.ndarray):
    """Return (max_idx_list, min_idx_list) for local extrema in a 1D array."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return [], []
    dy = np.diff(y)
    s = np.sign(dy)
    if np.all(s == 0):
        return [], []
    for i in range(1, s.size):
        if s[i] == 0:
            s[i] = s[i-1]
    for i in range(s.size - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i+1]
    max_idx = []
    min_idx = []
    for i in range(1, s.size):
        if s[i-1] > 0 and s[i] < 0:
            max_idx.append(i)
        elif s[i-1] < 0 and s[i] > 0:
            min_idx.append(i)
    return max_idx, min_idx


def _pick_with_spacing(cand_idx, mps, elevs, n_pick, min_spacing_mi, highest=True):
    """Pick up to n_pick candidate indices with minimum spacing in mileposts."""
    if not cand_idx:
        return []
    mps = np.asarray(mps, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    cand_idx = list(dict.fromkeys([int(i) for i in cand_idx if 0 <= int(i) < mps.size]))
    cand_idx.sort(key=lambda i: elevs[i], reverse=highest)
    chosen = []
    for i in cand_idx:
        mp = mps[i]
        if all(abs(mp - mps[j]) >= min_spacing_mi for j in chosen):
            chosen.append(i)
        if len(chosen) >= n_pick:
            break
    return chosen


def _binned_extrema_indices(mps: np.ndarray, elevs: np.ndarray, n_bins: int = 10):
    """Return (hi_idx_list, lo_idx_list) of per-bin extrema along milepost."""
    mps = np.asarray(mps, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    if mps.size == 0:
        return [], []
    n_bins = int(max(1, n_bins))
    mp0 = float(mps[0])
    mp1 = float(mps[-1])
    if not np.isfinite(mp0) or not np.isfinite(mp1) or mp1 <= mp0:
        return [], []
    edges = np.linspace(mp0, mp1, n_bins + 1)
    hi_idx = []
    lo_idx = []
    for b in range(n_bins):
        lo = edges[b]
        hi = edges[b + 1]
        if b == n_bins - 1:
            mask = (mps >= lo) & (mps <= hi)
        else:
            mask = (mps >= lo) & (mps < hi)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        e = elevs[idx]
        hi_i = int(idx[int(np.nanargmax(e))]) if np.any(np.isfinite(e)) else int(idx[0])
        lo_i = int(idx[int(np.nanargmin(e))]) if np.any(np.isfinite(e)) else int(idx[0])
        hi_idx.append(hi_i)
        lo_idx.append(lo_i)
    hi_idx = list(dict.fromkeys(hi_idx))
    lo_idx = list(dict.fromkeys(lo_idx))
    return hi_idx, lo_idx


def _collapse_nearby(idx_list, mps, elevs, min_spacing_mi, highest=True, protect_idx=None):
    """Collapse picks closer than min_spacing_mi, keeping the more extreme point."""
    if not idx_list:
        return []
    protect_idx = set(protect_idx or [])
    mps = np.asarray(mps, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    idx = sorted(set(int(i) for i in idx_list if 0 <= int(i) < mps.size), key=lambda i: float(mps[i]))
    if min_spacing_mi is None:
        return idx
    try:
        min_spacing_mi = float(min_spacing_mi)
    except Exception:
        return idx
    if min_spacing_mi <= 0:
        return idx

    out = []
    for i in idx:
        if not out:
            out.append(i)
            continue
        prev = out[-1]
        if abs(float(mps[i]) - float(mps[prev])) >= min_spacing_mi:
            out.append(i)
            continue

        if prev in protect_idx and i in protect_idx:
            out.append(i)
            continue
        if prev in protect_idx:
            continue
        if i in protect_idx:
            out[-1] = i
            continue

        if highest:
            keep_i = i if float(elevs[i]) >= float(elevs[prev]) else prev
        else:
            keep_i = i if float(elevs[i]) <= float(elevs[prev]) else prev
        out[-1] = keep_i

    out2 = []
    seen = set()
    for i in out:
        if i not in seen:
            out2.append(i); seen.add(i)
    return out2


def build_auto_monitor_points(mps, elevs, n_high=10, n_low=10, min_spacing_mi=0.5, name_prefix="", n_bins=10, mode="binned"):
    """Build monitor points at representative elevation highs/lows along a profile."""
    mps = np.asarray(mps, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    points = []
    if mps.size < 2:
        return points

    scope = (name_prefix or '').strip()

    def _nm(kind, label, mp):
        if name_prefix:
            return f"{name_prefix}{kind} {label} @ MP {mp:.3f}"
        return f"{kind} {label} @ MP {mp:.3f}"

    mode = (mode or "binned").strip().lower()

    try:
        g_hi = int(np.nanargmax(elevs))
    except Exception:
        g_hi = int(np.argmax(elevs))
    try:
        g_lo = int(np.nanargmin(elevs))
    except Exception:
        g_lo = int(np.argmin(elevs))

    if mode in ("local", "local_extrema", "peaks"):
        max_idx, min_idx = _local_extrema_indices(elevs)
        max_idx = sorted(max_idx, key=lambda i: elevs[i], reverse=True)
        min_idx = sorted(min_idx, key=lambda i: elevs[i])
        high_sel = _pick_with_spacing(max_idx, mps, elevs, int(n_high), float(min_spacing_mi), highest=True)
        low_sel  = _pick_with_spacing(min_idx, mps, elevs, int(n_low),  float(min_spacing_mi), highest=False)
        if g_hi not in high_sel:
            high_sel = [g_hi] + high_sel
        if g_lo not in low_sel:
            low_sel = [g_lo] + low_sel
        high_sel = _collapse_nearby(high_sel, mps, elevs, min_spacing_mi, highest=True, protect_idx={g_hi})
        low_sel  = _collapse_nearby(low_sel,  mps, elevs, min_spacing_mi, highest=False, protect_idx={g_lo})
        if len(high_sel) > int(n_high):
            rest = [i for i in high_sel if i != g_hi]
            rest.sort(key=lambda i: float(elevs[i]), reverse=True)
            high_sel = [g_hi] + rest[:max(0, int(n_high)-1)]
        if len(low_sel) > int(n_low):
            rest = [i for i in low_sel if i != g_lo]
            rest.sort(key=lambda i: float(elevs[i]))
            low_sel = [g_lo] + rest[:max(0, int(n_low)-1)]
    else:
        hi_bins, lo_bins = _binned_extrema_indices(mps, elevs, n_bins=int(n_bins))
        high_sel = list(dict.fromkeys([g_hi] + hi_bins))
        low_sel  = list(dict.fromkeys([g_lo] + lo_bins))

        high_sel = _collapse_nearby(high_sel, mps, elevs, min_spacing_mi, highest=True, protect_idx={g_hi})
        low_sel  = _collapse_nearby(low_sel,  mps, elevs, min_spacing_mi, highest=False, protect_idx={g_lo})

        try:
            n_high = int(n_high)
        except Exception:
            n_high = 10
        try:
            n_low = int(n_low)
        except Exception:
            n_low = 10

        if n_high > 0 and len(high_sel) > n_high:
            rest = [i for i in high_sel if i != g_hi]
            if n_high == 1:
                high_sel = [g_hi]
            else:
                k = max(1, int(math.ceil(len(rest) / float(n_high - 1))))
                high_sel = [g_hi] + rest[::k][:n_high-1]

        if n_low > 0 and len(low_sel) > n_low:
            rest = [i for i in low_sel if i != g_lo]
            if n_low == 1:
                low_sel = [g_lo]
            else:
                k = max(1, int(math.ceil(len(rest) / float(n_low - 1))))
                low_sel = [g_lo] + rest[::k][:n_low-1]

    def _emit(idx, kind, label, category):
        mp = float(mps[idx])
        points.append({
            'name': _nm(kind, label, mp),
            'mp': mp,
            'elev': float(elevs[idx]),
            'category': category,
            'scope': scope,
            'min_psig': None,
            'max_psig': None
        })

    _emit(g_hi, 'High', 'GLOBAL', 'auto_high')
    _emit(g_lo, 'Low', 'GLOBAL', 'auto_low')

    hi_rest = [i for i in high_sel if i != g_hi]
    lo_rest = [i for i in low_sel if i != g_lo]
    hi_rest.sort(key=lambda i: float(mps[i]))
    lo_rest.sort(key=lambda i: float(mps[i]))

    for k, i in enumerate(hi_rest, start=1):
        _emit(i, 'High', f'BIN{k:02d}', 'auto_high')

    for k, i in enumerate(lo_rest, start=1):
        _emit(i, 'Low', f'BIN{k:02d}', 'auto_low')

    return points


# === SI-based slug friction helper (Darcy-Weisbach over remaining liquid) ===
def slug_friction_psi_SI(L_ft, D_ft, v_fts, api_gravity, viscosity_cst, eps_ft, sg_fallback=0.85):
    """Darcy-Weisbach dp across the liquid slug ahead of the pig. Returns psi."""
    try:
        if L_ft <= 0 or v_fts <= 0 or D_ft <= 0:
            return 0.0
        ft_to_m = 0.3048
        L = L_ft * ft_to_m
        D = D_ft * ft_to_m
        v = v_fts * ft_to_m
        eps = eps_ft * ft_to_m

        if api_gravity is None and sg_fallback is not None:
            SG = sg_fallback
        elif api_gravity is None:
            SG = 0.85
        else:
            SG = 141.5 / (api_gravity + 131.5)
        rho = 999.0 * SG
        nu  = (viscosity_cst if viscosity_cst else 1.0) * 1.0e-6

        Re = (v * D) / max(1e-12, nu)
        if Re < 2300.0:
            f = 64.0 / max(1.0, Re)
        else:
            f = 0.25 / (math.log10((eps/(3.7*D)) + (5.74/(Re**0.9))))**2

        dp_Pa = f * (L / D) * 0.5 * rho * v * v
        dp_psi = dp_Pa / 6894.757
        return float(dp_psi)
    except Exception:
        return 0.0


# === Real-gas nitrogen helpers (Peng-Robinson) ===
def _f_to_k(t_f: float) -> float:
    return (t_f - 32.0) * (5.0 / 9.0) + 273.15

def _psia_to_pa(p_psia: float) -> float:
    return float(p_psia) * 6894.757

def _pa_to_psia(p_pa: float) -> float:
    return float(p_pa) / 6894.757

def z_factor_n2_pr(p_psia: float, t_f: float) -> float:
    """Return nitrogen compressibility factor Z using Peng-Robinson EOS."""
    p_pa = max(1.0, _psia_to_pa(p_psia))
    t_k = max(1.0, _f_to_k(t_f))
    R = 8.314462618

    kappa = 0.37464 + 1.54226 * N2_ACENTRIC - 0.26992 * (N2_ACENTRIC ** 2)
    alpha = (1.0 + kappa * (1.0 - math.sqrt(t_k / N2_CRIT_T_K))) ** 2
    a = 0.45724 * (R ** 2) * (N2_CRIT_T_K ** 2) / N2_CRIT_P_PA
    b = 0.07780 * R * N2_CRIT_T_K / N2_CRIT_P_PA
    A = a * alpha * p_pa / (R ** 2 * t_k ** 2)
    B = b * p_pa / (R * t_k)

    c3 = 1.0
    c2 = -(1.0 - B)
    c1 = A - 3.0 * B * B - 2.0 * B
    c0 = -(A * B - B * B - B ** 3)

    roots = np.roots([c3, c2, c1, c0])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
    if not real_roots:
        return 1.0
    Z = max(real_roots)
    if not np.isfinite(Z) or Z <= 0:
        return 1.0
    return float(Z)


def n2_density_kg_m3(p_psia: float, t_f: float) -> float:
    """Nitrogen density from PR EOS: rho = P*MW/(Z*R*T)."""
    p_pa = _psia_to_pa(p_psia)
    t_k = _f_to_k(t_f)
    Z = z_factor_n2_pr(p_psia, t_f)
    R = 8.314462618
    rho = p_pa * N2_MW_KG_PER_MOL / (max(1e-12, Z) * R * t_k)
    return float(rho)


def n2_moles_from_scf(scf: float, t_std_f: float = 60.0, z_std: float = 1.0) -> float:
    """Convert SCF at standard conditions into moles."""
    p_std_pa = _psia_to_pa(ATM_PSI)
    t_std_k = _f_to_k(t_std_f)
    V_m3 = float(scf) * 0.028316846592
    R = 8.314462618
    n = p_std_pa * V_m3 / (max(1e-12, z_std) * R * t_std_k)
    return float(n)


def n2_pressure_psia_from_moles(n_mol: float, V_ft3: float, t_f: float) -> float:
    """Solve for nitrogen pressure (psia) given moles, volume, temperature using PR EOS."""
    V_m3 = max(1e-12, float(V_ft3) * 0.028316846592)
    t_k = _f_to_k(t_f)
    R = 8.314462618
    p_pa = max(1.0, n_mol * R * t_k / V_m3)
    for _ in range(20):
        p_psia = _pa_to_psia(p_pa)
        Z = z_factor_n2_pr(p_psia, t_f)
        p_new = n_mol * R * t_k * Z / V_m3
        if abs(p_new - p_pa) / max(1.0, p_pa) < 1e-8:
            p_pa = p_new
            break
        p_pa = p_new
    return float(_pa_to_psia(p_pa))


def n2_moles_from_pressure_volume(p_psia: float, V_ft3: float, t_f: float) -> float:
    """Estimate nitrogen moles from pressure, volume, and temperature using PR Z."""
    p_psia = max(0.01, float(p_psia))
    V_m3 = max(1e-12, float(V_ft3) * 0.028316846592)
    t_k = _f_to_k(t_f)
    R = 8.314462618
    Z = max(1e-6, float(z_factor_n2_pr(p_psia, t_f)))
    p_pa = _psia_to_pa(p_psia)
    return float((p_pa * V_m3) / (Z * R * t_k))


def pressure_psig_from_moles(n_moles: float, V_ft3: float, T_F: float) -> float:
    """Convenience wrapper: nitrogen pressure (psig) from moles & volume."""
    try:
        psia = n2_pressure_psia_from_moles(n_moles, V_ft3, T_F)
        return float(psia) - ATM_PSI
    except Exception:
        R = 10.7316
        T_R = float(T_F) + 459.67
        psia = (float(n_moles) * R * T_R) / max(1e-12, float(V_ft3))
        return float(psia) - ATM_PSI

def gas_friction_loss_psi_SI(L_ft: float, D_ft: float, v_fts: float, p_avg_psia: float, t_f: float, eps_ft: float) -> float:
    """Darcy-Weisbach dp (psi) for nitrogen over length L at average pressure p_avg."""
    if L_ft <= 0 or v_fts <= 0 or D_ft <= 0:
        return 0.0
    ft_to_m = 0.3048
    L = L_ft * ft_to_m
    D = D_ft * ft_to_m
    v = v_fts * ft_to_m
    eps = eps_ft * ft_to_m

    rho = n2_density_kg_m3(p_avg_psia, t_f)
    mu = N2_VISCOSITY_PA_S
    Re = rho * v * D / max(1e-12, mu)
    if Re < 2300.0:
        f = 64.0 / max(1.0, Re)
    else:
        f = 0.25 / (math.log10((eps/(3.7*D)) + (5.74/(Re**0.9))))**2

    dp_pa = f * (L / D) * 0.5 * rho * v * v
    return float(dp_pa / 6894.757)


def required_injection_pressure_psig(p_behind_pig_psig: float, L_gas_ft: float, D_ft: float, v_fts: float, eps_ft: float, t_f: float) -> tuple:
    """Return (p_inj_psig, dp_gas_psi) required to supply the pressure immediately behind the pig."""
    p_behind_pig_psia = float(p_behind_pig_psig) + ATM_PSI
    if L_gas_ft <= 0 or v_fts <= 0:
        return float(p_behind_pig_psig), 0.0
    p_inj_psia = p_behind_pig_psia
    for _ in range(20):
        p_avg = 0.5 * (p_behind_pig_psia + p_inj_psia)
        dp = gas_friction_loss_psi_SI(L_gas_ft, D_ft, v_fts, p_avg, t_f, eps_ft)
        p_new = p_behind_pig_psia + dp
        if abs(p_new - p_inj_psia) / max(1.0, p_inj_psia) < 1e-8:
            p_inj_psia = p_new
            break
        p_inj_psia = p_new
    dp_gas = max(0.0, p_inj_psia - p_behind_pig_psia)
    return float(p_inj_psia - ATM_PSI), float(dp_gas)


def scfm_required_from_velocity(v_fts: float, area_ft2: float, p_inj_psig: float, p_behind_pig_psig: float, t_f: float) -> float:
    """Compute nitrogen SCFM required to sustain volumetric displacement at pig speed."""
    if v_fts <= 0 or area_ft2 <= 0:
        return 0.0
    p_inj_psia = float(p_inj_psig) + ATM_PSI
    p_behind_pig_psia = float(p_behind_pig_psig) + ATM_PSI
    p_avg = 0.5 * (p_inj_psia + p_behind_pig_psia)
    rho_avg = n2_density_kg_m3(p_avg, t_f)
    ft_to_m = 0.3048
    area_m2 = float(area_ft2) * (ft_to_m ** 2)
    v_m_s = float(v_fts) * ft_to_m
    m_dot = rho_avg * v_m_s * area_m2

    rho_std = n2_density_kg_m3(ATM_PSI, 60.0)
    q_std_m3_s = m_dot / max(1e-12, rho_std)
    scfm = q_std_m3_s * 35.3146667 * 60.0
    return float(scfm)


# ======================== NEW v29 FUNCTIONS ========================

def _detect_col(df_cols_lower, aliases):
    """Return first matching column name from df_cols_lower dict {lowercase: original}."""
    for alias in aliases:
        if alias in df_cols_lower:
            return df_cols_lower[alias]
    return None


def _parse_flexible_profile(file_path, elevation_units='Feet'):
    """
    Parse a TXT/CSV/ILI file with flexible column detection.
    Supports: lat/lon/elev GPS-based, milepost+elev, or milepost+lat/lon only.
    Returns: (mileposts_relative, elevations_ft, raw_df_with_pipe_props)
    """
    df = None
    for sep in [None, ',', '\t', ';', '|']:
        try:
            if sep is None:
                df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
            else:
                df = pd.read_csv(file_path, sep=sep, on_bad_lines='skip')
            if df is not None and len(df.columns) >= 2:
                break
        except Exception:
            continue
    if df is None or len(df.columns) < 2:
        raise ValueError(f"Could not parse TXT/CSV file: {file_path}. Check delimiter and format.")

    cols_lower = {str(c).strip().lower(): str(c) for c in df.columns}

    lat_col  = _detect_col(cols_lower, [a.lower() for a in _COL_LAT])
    lon_col  = _detect_col(cols_lower, [a.lower() for a in _COL_LON])
    elev_col = _detect_col(cols_lower, [a.lower() for a in _COL_ELEV])
    mp_col   = _detect_col(cols_lower, [a.lower() for a in _COL_MP])

    has_gps  = (lat_col is not None and lon_col is not None)
    has_elev = (elev_col is not None)
    has_mp   = (mp_col is not None)

    if not has_gps and not has_mp:
        raise ValueError(
            f"File {os.path.basename(file_path)}: Cannot find lat/lon or milepost columns.\n"
            f"Detected columns: {list(df.columns)}"
        )
    if not has_elev and not has_gps:
        raise ValueError(
            f"File {os.path.basename(file_path)}: Cannot find elevation or GPS altitude column.\n"
            f"Detected columns: {list(df.columns)}"
        )

    if has_gps:
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df = df.dropna(subset=[lat_col, lon_col])
    if has_elev:
        df[elev_col] = pd.to_numeric(df[elev_col], errors='coerce')
    if has_mp:
        df[mp_col] = pd.to_numeric(df[mp_col], errors='coerce')
        df = df.dropna(subset=[mp_col])

    if len(df) < 2:
        raise ValueError(f"File {os.path.basename(file_path)}: fewer than 2 valid rows after parsing.")

    # Compute mileposts
    if has_gps:
        lats = df[lat_col].to_numpy()
        lons = df[lon_col].to_numpy()
        mps_rel = [0.0]
        for j in range(1, len(lats)):
            try:
                d = geodesic((lats[j-1], lons[j-1]), (lats[j], lons[j])).miles
            except Exception:
                d = haversine_distance(lats[j-1], lons[j-1], lats[j], lons[j])
            mps_rel.append(mps_rel[-1] + d)
        mps_rel = np.array(mps_rel)
    else:
        mps_raw = df[mp_col].to_numpy(dtype=float)
        mp_range = float(np.nanmax(mps_raw) - np.nanmin(mps_raw))
        if mp_range > 500:
            mps_rel = (mps_raw - mps_raw[0]) / 5280.0
        elif mp_range > 50 and mp_range < 500:
            mps_rel = (mps_raw - mps_raw[0]) * 0.621371
        else:
            mps_rel = mps_raw - mps_raw[0]

    # Elevations
    if has_elev:
        elevs = df[elev_col].to_numpy(dtype=float)
        valid = np.isfinite(elevs)
        if valid.sum() >= 2:
            elevs = np.where(valid, elevs, np.interp(np.arange(len(elevs)), np.where(valid)[0], elevs[valid]))
        col_lower = elev_col.lower()
        if '(m)' in col_lower or '_m' in col_lower or col_lower.endswith(' m'):
            elevs = elevs * 3.28084
        elif elevation_units == 'Meters':
            elevs = elevs * 3.28084
    elif has_gps:
        logging.warning(f"No elevation column found in {file_path}; setting elevations to 0.")
        elevs = np.zeros(len(mps_rel))
    else:
        raise ValueError(f"No elevation data available in {file_path}.")

    if len(mps_rel) != len(elevs):
        min_len = min(len(mps_rel), len(elevs))
        mps_rel = mps_rel[:min_len]
        elevs = elevs[:min_len]

    mps_rel = np.array(mps_rel, dtype=float)
    elevs = np.array(elevs, dtype=float)

    if has_gps:
        lats_out = df[lat_col].to_numpy(dtype=float)[:len(mps_rel)]
        lons_out = df[lon_col].to_numpy(dtype=float)[:len(mps_rel)]
        return {
            'mode': 'latlon',
            'lat': lats_out,
            'lon': lons_out,
            'elevation_ft': elevs,
        }
    else:
        return {
            'mode': 'milepost',
            'milepost': mps_rel,
            'elevation_ft': elevs,
        }


def solve_gas_column_with_boosters(
    P_behind_pig_psig, pig_mp, purge_start_mp,
    pipe_diameter_ft, area_ft2, roughness_ft, t_f, v_fts,
    active_boosters
):
    """
    Solve pressures and injection flow for a gas column segmented by booster stations.

    Physics:
    - Build sub-columns: [purge_start, b1.mp, b2.mp, ..., pig_mp]
    - Work backwards from pig: each sub-column has gas friction dp
    - Each booster delivers at its discharge_psig (caps if needed)
    - Injection point supplies to the first sub-column at the required suction of booster 1
    - Feasibility: injection_psig must be <= caller's max; each booster suction >= suction_min_psig

    Returns dict:
      injection_psig, injection_scfm, gas_dp_total, booster_states (list), feasible (bool),
      seg_pressures (list of (P_left_psig, P_right_psig) per sub-column from upstream to pig)
    """
    boundaries = [purge_start_mp] + [float(b['milepost']) for b in active_boosters] + [float(pig_mp)]
    n_segs = len(boundaries) - 1
    feasible = True

    P_right = float(P_behind_pig_psig)
    seg_pressures = [(0.0, 0.0)] * n_segs

    booster_states = []

    for k in range(n_segs - 1, -1, -1):
        L_seg_ft = max(1e-6, (boundaries[k + 1] - boundaries[k]) * 5280.0)
        p_avg_psia = max(ATM_PSI, P_right + ATM_PSI)
        dp_seg = gas_friction_loss_psi_SI(L_seg_ft, pipe_diameter_ft, v_fts, p_avg_psia, t_f, roughness_ft)
        P_left = P_right + dp_seg
        seg_pressures[k] = (float(P_left), float(P_right))

        if k > 0:
            booster = active_boosters[k - 1]
            discharge_needed = P_left
            discharge_actual = min(float(booster['discharge_psig']), discharge_needed + 50.0)
            if discharge_needed > float(booster['discharge_psig']) + 1e-6:
                feasible = False
                discharge_actual = float(booster['discharge_psig'])

            booster_suction = discharge_actual
            if booster_suction < float(booster.get('suction_min_psig', 0.0)) - 1e-6:
                feasible = False

            booster_states.insert(0, {
                'mp': float(booster['milepost']),
                'name': booster.get('name', ''),
                'suction_psig': None,
                'discharge_psig': discharge_actual,
                'discharge_needed_psig': discharge_needed,
                'feasible': discharge_needed <= float(booster['discharge_psig']) + 1e-6,
            })
            P_right = float(P_left)

    # Fill in booster suction pressures from computed seg_pressures
    for idx, bs in enumerate(booster_states):
        bs['suction_psig'] = float(seg_pressures[idx][1])

    injection_psig = float(seg_pressures[0][0])

    injection_scfm = scfm_required_from_velocity(
        v_fts, area_ft2, injection_psig, float(P_behind_pig_psig), t_f
    )
    if active_boosters and injection_scfm > 0:
        for b in active_boosters:
            mf = float(b.get('max_flow_scfm', math.inf) or math.inf)
            if mf < math.inf and injection_scfm > mf + 1e-6:
                feasible = False

    gas_dp_total = max(0.0, injection_psig - float(P_behind_pig_psig))

    return {
        'injection_psig': injection_psig,
        'injection_scfm': injection_scfm,
        'gas_dp_total': gas_dp_total,
        'booster_states': booster_states,
        'seg_pressures': seg_pressures,
        'feasible': feasible,
    }


def compute_n2_inventory_scf(pig_mp, purge_start_mp, area_ft2, active_boosters, seg_pressures, t_f):
    """Total SCF of N2 in the gas column behind the pig at the current pig position."""
    boundaries = [purge_start_mp] + [float(b['milepost']) for b in active_boosters] + [float(pig_mp)]
    total = 0.0
    for k, (P_left, P_right) in enumerate(seg_pressures):
        L_ft = max(0.0, (boundaries[k + 1] - boundaries[k]) * 5280.0)
        V_ft3 = area_ft2 * L_ft
        P_avg_psia = 0.5 * (float(P_left) + ATM_PSI + float(P_right) + ATM_PSI)
        Z_avg = max(1e-6, z_factor_n2_pr(P_avg_psia, t_f))
        Z_std = max(1e-6, z_factor_n2_pr(ATM_PSI, 60.0))
        scf = V_ft3 * P_avg_psia / ATM_PSI * Z_std / Z_avg
        total += max(0.0, scf)
    return float(total)


def get_pipe_props_at_mp(mp, pipe_segments, default_od_in, default_wt_in, default_roughness_num):
    """Look up pipe OD, WT, roughness at a given milepost from segment list."""
    for seg in pipe_segments:
        if float(seg['start_mp']) <= float(mp) <= float(seg['end_mp']):
            try:
                od = nps_data[resolve_nps_key(seg['nps'])]['OD_in']
                wt = float(seg['wall_thickness_in'])
                rnum = int(seg.get('roughness_num', default_roughness_num))
                return od, wt, rnum
            except Exception:
                pass
    return default_od_in, default_wt_in, default_roughness_num


def run_simulation(dialog_root, inputs, purge_mileposts, elevations, system_mileposts, system_elevations):
    cfg = dict(inputs)
    cfg.setdefault('temperature_f', float(N2_TEMP_F_DEFAULT))

    sim_t0 = time.monotonic()
    try:
        max_wall_time_s = float(cfg.get('max_wall_time_s', 180.0) or 180.0)
    except Exception:
        max_wall_time_s = 180.0
    try:
        max_elapsed_time_hr = float(cfg.get('max_elapsed_time_hr', 1.0e6) or 1.0e6)
    except Exception:
        max_elapsed_time_hr = 1.0e6
    try:
        max_sim_steps = int(cfg.get('max_sim_steps', 0) or 0)
    except Exception:
        max_sim_steps = 0
    stop_reason = None

    def _to_float(val, default=None):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    _numeric_fields = [
        'pipe_wt','viscosity_cst','api_gravity','max_nitrogen_pressure','max_drive_pressure',
        'exit_pressure_run','exit_pressure_end','purge_end_mp','purge_start_mp',
        'roughness_num','max_n2_rate_scfm','max_pig_speed','min_pig_speed',
        'n2_end_pressure','resolution','system_end_mp','target_pig_speed','slack_pressure_psig','maop_psig','max_outlet_flow_bph'
    ]
    for _k in _numeric_fields:
        if _k in cfg:
            _v = _to_float(cfg[_k], None)
            if _v is not None:
                cfg[_k] = _v

    try:
        nps_key = resolve_nps_key(cfg['nps'])
    except KeyError as e:
        raise ValueError(str(e))
    pipe_od = nps_data[nps_key]["OD_in"]
    pipe_id = pipe_od - 2 * cfg['pipe_wt']
    if pipe_id <= 0:
        raise ValueError(f"Computed ID <= 0. Check NPS {nps_key} and wall {cfg['pipe_wt']}.")
    pipe_diameter = pipe_id / 12.0
    area = np.pi * (pipe_diameter / 2) ** 2

    max_outlet_flow_bph = cfg.get('max_outlet_flow_bph', None)
    v_flow_cap = math.inf
    if max_outlet_flow_bph is not None:
        try:
            max_outlet_flow_bph = float(max_outlet_flow_bph)
        except Exception:
            max_outlet_flow_bph = None
    if max_outlet_flow_bph is not None and max_outlet_flow_bph > 0:
        q_cap_ft3_s = max_outlet_flow_bph * 5.614583333333333 / 3600.0
        v_flow_cap = q_cap_ft3_s / area
        cfg['max_pig_speed_from_max_outlet_flow_mph'] = v_flow_cap / 1.46667

    roughness = roughness_data[int(cfg['roughness_num'])]['roughness_ft']

    if int(cfg['fluid_num']) == 3:
        specific_gravity = 141.5 / (131.5 + cfg['api_gravity'])
        viscosity_cst = cfg['viscosity_cst']
    else:
        specific_gravity = fluid_data[int(cfg['fluid_num'])]['sg']
        viscosity_cst = fluid_data[int(cfg['fluid_num'])]['viscosity_cst']

    api_for_dp = cfg['api_gravity'] if int(cfg['fluid_num']) == 3 else None
    gamma = specific_gravity * 62.4
    fluid_density = gamma
    viscosity = (gamma / 32.174) * (viscosity_cst * 1.076e-5)

    purge_length = (cfg['purge_end_mp'] - cfg['purge_start_mp']) * 5280.0
    total_volume_ft3 = area * purge_length
    total_volume_scf = total_volume_ft3 / SCF_TO_FT3

    n2_temp_f = float(cfg.get('n2_temp_f', N2_TEMP_F_DEFAULT) or N2_TEMP_F_DEFAULT)

    P_end_abs = float(cfg['n2_end_pressure']) + ATM_PSI
    Z_end = z_factor_n2_pr(P_end_abs, n2_temp_f)
    Z_std = z_factor_n2_pr(ATM_PSI, 60.0)
    T_std_R = 60.0 + 459.67
    T_pipe_R = n2_temp_f + 459.67
    cutoff_volume = total_volume_ft3 * (P_end_abs / ATM_PSI) * (T_std_R / max(1e-12, T_pipe_R)) * (Z_std / max(1e-12, Z_end))
    logging.info(
        f"Total pipe volume: {total_volume_ft3:.0f} ft3, {total_volume_scf:.0f} SCF, "
        f"Cutoff (N2) volume: {cutoff_volume:.0f} SCF"
    )

    n_points = len(purge_mileposts)
    distances = purge_mileposts * 5280.0

    cs = CubicSpline(system_mileposts, system_elevations)
    h_exit = cs(cfg['system_end_mp'])
    head_losses = fluid_density * (h_exit - elevations) / 144.0

    slack_threshold_psig = float(cfg.get('slack_pressure_psig', 50.0))
    maop_psig = cfg.get('maop_psig', None)
    try:
        maop_psig = float(maop_psig) if maop_psig is not None else None
    except Exception:
        maop_psig = None

    max_drive_user_psig = float(cfg.get('max_nitrogen_pressure', cfg.get('max_drive_pressure', 400.0)) or 400.0)
    safe_max_drive_envelope = None
    if (maop_psig is not None) and bool(cfg.get('safe_drive_envelope_enabled', True)):
        try:
            safe_max_drive_envelope = compute_safe_max_drive_envelope(
                purge_mileposts, elevations, system_mileposts, system_elevations,
                fluid_density, maop_psig, max_drive_user_psig,
            )
        except Exception as _e:
            logging.warning(f"Safe drive envelope failed; disabling. Reason: {_e}")
            safe_max_drive_envelope = None

    monitor_points = []
    if bool(cfg.get('auto_monitor_enabled', True)):
        try:
            n_hi = int(cfg.get('auto_monitor_n_high', 10))
        except Exception:
            n_hi = 10
        try:
            n_lo = int(cfg.get('auto_monitor_n_low', 10))
        except Exception:
            n_lo = 10
        try:
            spacing_mi = float(cfg.get('auto_monitor_min_spacing_mi', 0.5))
        except Exception:
            spacing_mi = 0.5

        mode = str(cfg.get('auto_monitor_mode', 'binned')).strip().lower()
        if mode not in ('binned', 'local_extrema', 'local', 'peaks'):
            mode = 'binned'
        try:
            n_bins = int(cfg.get('auto_monitor_bins', 10))
        except Exception:
            n_bins = 10

        scope = str(cfg.get('auto_monitor_scope', 'system')).strip().lower()
        if scope not in ('system', 'purge', 'both'):
            scope = 'system'

        if scope in ('system', 'both'):
            monitor_points.extend(build_auto_monitor_points(
                system_mileposts, system_elevations,
                n_high=n_hi, n_low=n_lo, min_spacing_mi=spacing_mi,
                name_prefix="SYS ", n_bins=n_bins, mode=mode
            ))
        if scope in ('purge', 'both'):
            monitor_points.extend(build_auto_monitor_points(
                purge_mileposts, elevations,
                n_high=n_hi, n_low=n_lo, min_spacing_mi=spacing_mi,
                name_prefix="PURGE ", n_bins=n_bins, mode=mode
            ))

    ips_mps = []
    if bool(cfg.get('has_ips', False)):
        ips_mps = parse_milepost_list(cfg.get('ips_mps', None) or cfg.get('ips_mp', None))
        ips_mps = [float(m) for m in ips_mps if cfg['purge_start_mp'] < float(m) < cfg['system_end_mp']]
        ips_mps = sorted(list(dict.fromkeys(ips_mps)))
        for k, mp in enumerate(ips_mps, start=1):
            try:
                elev_mp = float(cs(mp))
            except Exception:
                elev_mp = float(np.interp(mp, purge_mileposts, elevations))
            monitor_points.append({
                'name': f'IPS#{k} @ MP {mp:.3f}',
                'mp': float(mp),
                'elev': elev_mp,
                'category': 'pump_station'
            })
    cfg['ips_mps'] = ips_mps

    min_pump_suction_psig = cfg.get('min_pump_suction_pressure', None)
    try:
        min_pump_suction_psig = float(min_pump_suction_psig) if min_pump_suction_psig is not None else None
    except Exception:
        min_pump_suction_psig = None

    for p in monitor_points:
        p.setdefault('category', 'user')
        p['min_psig'] = None
        p['max_psig'] = None
        if maop_psig is not None:
            p['max_psig'] = maop_psig

        if p.get('category') in ('auto_high', 'pump_station'):
            p['min_psig'] = slack_threshold_psig
            if p.get('category') == 'pump_station' and min_pump_suction_psig is not None:
                p['min_psig'] = max(p['min_psig'], min_pump_suction_psig)

        if p.get('category') == 'auto_low':
            p['min_psig'] = None

    _seen = set()
    for p in monitor_points:
        base = p['name']
        name = base
        n = 2
        while name in _seen:
            name = f"{base} ({n})"
            n += 1
        p['name'] = name
        _seen.add(name)

    monitor_pressures = {p['name']: np.full(n_points, np.nan, dtype=np.float64) for p in monitor_points}
    monitor_pressures_static = {p['name']: np.full(n_points, np.nan, dtype=np.float64) for p in monitor_points}
    monitor_pressures_static_maxdrive = {p['name']: np.full(n_points, np.nan, dtype=np.float64) for p in monitor_points}

    static_hgl_ft = np.full(n_points, np.nan, dtype=np.float64)
    static_max_any_psig = np.full(n_points, np.nan, dtype=np.float64)
    static_max_any_mp = np.full(n_points, np.nan, dtype=np.float64)
    static_max_any_elev_ft = np.full(n_points, np.nan, dtype=np.float64)
    static_max_any_phase = np.array([''] * n_points, dtype=object)

    static_hgl_ft_maxdrive = np.full(n_points, np.nan, dtype=np.float64)
    static_max_any_psig_maxdrive = np.full(n_points, np.nan, dtype=np.float64)
    static_max_any_mp_maxdrive = np.full(n_points, np.nan, dtype=np.float64)
    static_max_any_elev_ft_maxdrive = np.full(n_points, np.nan, dtype=np.float64)
    static_max_any_phase_maxdrive = np.array([''] * n_points, dtype=object)

    static_violations = []
    static_violations_maxdrive = []
    monitor_violations = []

    elapsed_times = np.zeros(n_points, dtype=np.float64)
    behind_pig_pressures = np.zeros(n_points, dtype=np.float64)
    injection_pressures = np.zeros(n_points, dtype=np.float64)
    gas_dp_losses = np.zeros(n_points, dtype=np.float64)
    differential_pressures = np.zeros(n_points, dtype=np.float64)
    miles_to_outlet = np.zeros(n_points, dtype=np.float64)
    friction_losses = np.zeros(n_points, dtype=np.float64)
    exit_pressures = np.zeros(n_points, dtype=np.float64)
    exit_pressures_programmed = np.full(n_points, np.nan, dtype=np.float64)
    exit_pressures_lb = np.full(n_points, np.nan, dtype=np.float64)
    exit_pressures_ub = np.full(n_points, np.nan, dtype=np.float64)
    injection_rates = np.zeros(n_points, dtype=np.float64)
    cumulative_n2 = np.zeros(n_points, dtype=np.float64)
    pig_speeds = np.zeros(n_points, dtype=np.float64)
    exceed_points = []

    hydraulic_boundary_mps = np.full(n_points, np.nan, dtype=np.float64)
    hydraulic_boundary_pressures = np.full(n_points, np.nan, dtype=np.float64)
    head_to_boundary = np.full(n_points, np.nan, dtype=np.float64)
    ips_active_mp = np.full(n_points, np.nan, dtype=np.float64)
    ips_suction_psig = np.full(n_points, np.nan, dtype=np.float64)
    ips_discharge_req_psig = np.full(n_points, np.nan, dtype=np.float64)
    ips_pump_dp_req_psi = np.full(n_points, np.nan, dtype=np.float64)
    ips_flow_bph = np.full(n_points, np.nan, dtype=np.float64)
    ips_state = np.array([""] * n_points, dtype=object)

    # --- v29: Booster stations and pipe segments ---
    booster_stations_raw = cfg.get('booster_stations', []) or []
    booster_stations = sorted(
        [b for b in booster_stations_raw
         if cfg['purge_start_mp'] < float(b.get('milepost', -1)) < cfg['system_end_mp']],
        key=lambda b: float(b['milepost'])
    )

    n2_inventory_scf = np.full(n_points, np.nan, dtype=np.float64)
    booster_diagnostics = [{} for _ in range(n_points)]

    pipe_segments_cfg = cfg.get('pipe_segments', []) or []
    _default_od_in = pipe_od
    _default_wt_in = cfg['pipe_wt']
    _default_rnum = int(cfg['roughness_num'])

    cumulative_n2[0] = 0.0
    cumulative_distance = 0.0
    last_valid_i = n_points - 1
    v_max = cfg['max_pig_speed'] * 1.46667
    v_target = float(cfg.get('target_pig_speed', cfg['max_pig_speed']) or cfg['max_pig_speed']) * 1.46667

    if v_flow_cap != math.inf:
        v_max = min(v_max, v_flow_cap)
        v_target = min(v_target, v_flow_cap)
    max_inj_psig = float(cfg.get('max_nitrogen_pressure', cfg.get('max_drive_pressure', 400.0)) or 400.0)

    max_rate_in = cfg.get('max_n2_rate_scfm', None)
    try:
        max_rate = float(max_rate_in) if (max_rate_in is not None and float(max_rate_in) > 0) else math.inf
    except Exception:
        max_rate = math.inf

    n2_cutoff = cfg.get('n2_cutoff_scf', None)
    if n2_cutoff is not None:
        try:
            n2_cutoff = float(n2_cutoff)
            if n2_cutoff <= 0:
                n2_cutoff = None
        except Exception:
            n2_cutoff = None
    cutoff_value = n2_cutoff if n2_cutoff is not None else cutoff_volume
    coast_mode = False
    n2_moles_total = None
    V_gas_ft3 = None
    coast_start_p_psig = None
    coast_start_V_ft3 = None

    try:
        ips_shutdown_dist = float(cfg.get('ips_shutdown_dist', 0.0) or 0.0)
    except Exception:
        ips_shutdown_dist = 0.0
    try:
        min_pump_suction_pressure = float(cfg.get('min_pump_suction_pressure', 0.0) or 0.0)
    except Exception:
        min_pump_suction_pressure = 0.0
    min_pump_flow_bph = cfg.get('min_pump_flow_bph', None)
    try:
        min_pump_flow_bph = float(min_pump_flow_bph) if (min_pump_flow_bph is not None and float(min_pump_flow_bph) > 0) else None
    except Exception:
        min_pump_flow_bph = None
    try:
        ips_check_dp_psi = float(cfg.get('ips_check_dp_psi', cfg.get('ips_check_dp', 5.0)) or 5.0)
    except Exception:
        ips_check_dp_psi = 5.0

    def _bph_from_v(v_fts, area_ft2):
        return (max(0.0, float(v_fts)) * float(area_ft2) * 3600.0) / 5.614583

    def _v_from_bph(bph, area_ft2):
        if area_ft2 <= 0:
            return 0.0
        return (float(bph) * 5.614583) / (3600.0 * float(area_ft2))

    P_end_prev = None

    max_steps = n_points - 1
    if max_sim_steps and max_sim_steps > 0:
        max_steps = min(max_steps, int(max_sim_steps))

    for i in range(max_steps):
        segment_length_ft = max(1e-6, distances[i + 1] - distances[i])
        pig_mp_now = float(purge_mileposts[i])
        elev_pig = float(elevations[i])

        # Variable pipe properties for this step
        if pipe_segments_cfg:
            _od_i, _wt_i, _rnum_i = get_pipe_props_at_mp(
                pig_mp_now, pipe_segments_cfg, _default_od_in, _default_wt_in, _default_rnum
            )
            _pid_i = _od_i - 2.0 * _wt_i
            _pdiam_i = _pid_i / 12.0
            _area_i = math.pi * (_pdiam_i / 2.0) ** 2
            _rough_i = roughness_data[_rnum_i]['roughness_ft']
        else:
            _pdiam_i = pipe_diameter
            _area_i = area
            _rough_i = roughness

        max_behind_cap_psig_step = None
        if safe_max_drive_envelope is not None:
            try:
                _cap = float(safe_max_drive_envelope[i])
                if np.isfinite(_cap) and _cap > 0:
                    max_behind_cap_psig_step = _cap
            except Exception:
                max_behind_cap_psig_step = None

        if (time.monotonic() - sim_t0) > max_wall_time_s:
            stop_reason = f"TIMEOUT>{max_wall_time_s:.0f}s"
            last_valid_i = max(0, i - 1)
            logging.warning(f"Simulation timeout reached at i={i}, MP={pig_mp_now:.3f}. Stopping early.")
            break

        P_end_prog = float(target_exit_pressure(pig_mp_now, cfg, cfg['purge_start_mp'], cfg['purge_end_mp'], cfg['throttle_down_miles']))
        P_end = float(P_end_prog)
        exit_pressures_programmed[i] = float(P_end_prog)

        L_gas_ft = max(0.0, (pig_mp_now - cfg['purge_start_mp']) * 5280.0)

        P_trap_psig = 0.0
        if coast_mode and cutoff_value is not None:
            if n2_moles_total is None:
                n2_moles_total = n2_moles_from_scf(cutoff_value)
            if V_gas_ft3 is None:
                V_gas_ft3 = max(1e-6, area * max(1e-6, L_gas_ft))
            P_trap_psig = pressure_psig_from_moles(n2_moles_total, V_gas_ft3, cfg.get('temperature_f', 60.0))
        else:
            P_trap_psig = 0.0

        def solve_boundary(boundary_mp, P_boundary, elev_boundary, label=""):
            slug_length_ft_local = max(0.0, (float(boundary_mp) - pig_mp_now) * 5280.0)
            head_local = float(fluid_density) * (float(elev_boundary) - elev_pig) / 144.0

            def evaluate(v_try, enforce_caps=True):
                v_try = float(v_try)
                fr_liq = slug_friction_psi_SI(slug_length_ft_local, _pdiam_i, v_try, api_for_dp, viscosity_cst, _rough_i, sg_fallback=specific_gravity)
                P_behind_pig_req = max(float(P_boundary), float(P_boundary) + head_local + fr_liq)

                if coast_mode and cutoff_value is not None:
                    cap = float(P_trap_psig)
                    ok = (P_behind_pig_req <= cap + 1e-6) if enforce_caps else True
                    P_behind_pig_out = min(cap, P_behind_pig_req) if enforce_caps else P_behind_pig_req
                    P_inj_out = P_behind_pig_out
                    dp_gas = 0.0
                    q_use = 0.0
                    return ok, P_behind_pig_req, P_behind_pig_out, P_inj_out, dp_gas, q_use, fr_liq

                # Gas column: with or without boosters
                _active_bst = [b for b in booster_stations if float(b['milepost']) < pig_mp_now]
                if _active_bst:
                    bst_sol = solve_gas_column_with_boosters(
                        P_behind_pig_req, pig_mp_now, cfg['purge_start_mp'],
                        _pdiam_i, _area_i, _rough_i, cfg.get('temperature_f', 60.0),
                        v_try, _active_bst
                    )
                    P_inj_req = bst_sol['injection_psig']
                    dp_gas_val = bst_sol['gas_dp_total']
                    q_required = bst_sol['injection_scfm']
                    if not bst_sol['feasible']:
                        ok = False
                    else:
                        ok = True
                    if enforce_caps:
                        ok = ok and (P_inj_req <= max_inj_psig + 1e-6) and (q_required <= max_rate + 1e-6)
                        if max_behind_cap_psig_step is not None:
                            ok = ok and (P_behind_pig_req <= float(max_behind_cap_psig_step) + 1e-6)
                    P_behind_pig_out = P_behind_pig_req
                    P_inj_out = P_inj_req
                    dp_gas = float(dp_gas_val)
                    q_use = q_required
                    return ok, P_behind_pig_req, P_behind_pig_out, P_inj_out, dp_gas, q_use, fr_liq
                else:
                    P_inj_req, dp_gas_val = required_injection_pressure_psig(
                        P_behind_pig_req, L_gas_ft, _pdiam_i, v_try, _rough_i,
                        cfg.get('temperature_f', 60.0)
                    )
                    q_required = scfm_required_from_velocity(
                        v_try, _area_i, P_inj_req, P_behind_pig_req, cfg.get('temperature_f', 60.0)
                    )

                    ok = True
                    if enforce_caps:
                        ok = (P_inj_req <= max_inj_psig + 1e-6) and (q_required <= max_rate + 1e-6)
                        if max_behind_cap_psig_step is not None:
                            ok = ok and (P_behind_pig_req <= float(max_behind_cap_psig_step) + 1e-6)

                    P_behind_pig_out = P_behind_pig_req
                    P_inj_out = P_inj_req
                    dp_gas = float(dp_gas_val)
                    q_use = q_required
                    return ok, P_behind_pig_req, P_behind_pig_out, P_inj_out, dp_gas, q_use, fr_liq

            ok0, P_req0, P_out0, P_inj0, dp_gas0, q0, fr0 = evaluate(1e-6, enforce_caps=True)
            if not ok0:
                return {
                    'feasible': False, 'v': 0.0, 'P_behind_req': P_req0, 'P_behind_out': P_out0,
                    'P_inj': P_inj0, 'dp_gas': dp_gas0, 'q_use': q0, 'fr_liq': fr0,
                    'head': head_local, 'slug_length_ft': slug_length_ft_local,
                    'booster_states': [], 'seg_pressures': []
                }

            v_hi = min(v_max, v_target)
            ok_hi, P_req_hi, P_out_hi, P_inj_hi, dp_gas_hi, q_hi, fr_hi = evaluate(v_hi, enforce_caps=True)

            if ok_hi:
                _active_bst2 = [b for b in booster_stations if float(b['milepost']) < pig_mp_now]
                _bst_states2 = []
                _seg_p2 = []
                if _active_bst2 and not (coast_mode and cutoff_value is not None):
                    _bsol2 = solve_gas_column_with_boosters(
                        P_out_hi, pig_mp_now, cfg['purge_start_mp'],
                        _pdiam_i, _area_i, _rough_i, cfg.get('temperature_f', 60.0),
                        v_hi, _active_bst2
                    )
                    _bst_states2 = _bsol2.get('booster_states', [])
                    _seg_p2 = _bsol2.get('seg_pressures', [])
                return {
                    'feasible': True, 'v': float(v_hi), 'P_behind_req': P_req_hi, 'P_behind_out': P_out_hi,
                    'P_inj': P_inj_hi, 'dp_gas': dp_gas_hi, 'q_use': q_hi, 'fr_liq': fr_hi,
                    'head': head_local, 'slug_length_ft': slug_length_ft_local,
                    'booster_states': _bst_states2, 'seg_pressures': _seg_p2
                }

            lo, hi = 0.0, float(v_hi)
            best = 0.0
            best_pack = (P_req0, P_out0, P_inj0, dp_gas0, q0, fr0)
            for _ in range(40):
                if (hi - lo) < 1e-6:
                    break
                mid = 0.5 * (lo + hi)
                ok_mid, P_req_mid, P_out_mid, P_inj_mid, dp_gas_mid, q_mid, fr_mid = evaluate(mid, enforce_caps=True)
                if ok_mid:
                    lo = mid
                    best = mid
                    best_pack = (P_req_mid, P_out_mid, P_inj_mid, dp_gas_mid, q_mid, fr_mid)
                else:
                    hi = mid

            P_req_b, P_out_b, P_inj_b, dp_gas_b, q_b, fr_b = best_pack
            _active_bst3 = [b for b in booster_stations if float(b['milepost']) < pig_mp_now]
            _bst_states3 = []
            _seg_p3 = []
            if _active_bst3 and not (coast_mode and cutoff_value is not None):
                _bsol3 = solve_gas_column_with_boosters(
                    P_out_b, pig_mp_now, cfg['purge_start_mp'],
                    _pdiam_i, _area_i, _rough_i, cfg.get('temperature_f', 60.0),
                    best, _active_bst3
                )
                _bst_states3 = _bsol3.get('booster_states', [])
                _seg_p3 = _bsol3.get('seg_pressures', [])
            return {
                'feasible': True, 'v': float(best), 'P_behind_req': P_req_b, 'P_behind_out': P_out_b,
                'P_inj': P_inj_b, 'dp_gas': dp_gas_b, 'q_use': q_b, 'fr_liq': fr_b,
                'head': head_local, 'slug_length_ft': slug_length_ft_local,
                'booster_states': _bst_states3, 'seg_pressures': _seg_p3
            }

        endpoint_mode = str(cfg.get('endpoint_constraint_mode', 'none')).strip().lower()
        exit_min_clamp = cfg.get('exit_pressure_min_clamp', None)
        exit_max_clamp = cfg.get('exit_pressure_max_clamp', None)
        exit_ramp = cfg.get('exit_pressure_ramp_psi_per_hr', None)
        try:
            exit_min_clamp = float(exit_min_clamp) if exit_min_clamp is not None else None
        except Exception:
            exit_min_clamp = None
        try:
            exit_max_clamp = float(exit_max_clamp) if exit_max_clamp is not None else None
        except Exception:
            exit_max_clamp = None
        try:
            exit_ramp = float(exit_ramp) if exit_ramp is not None else None
        except Exception:
            exit_ramp = None

        def _exit_bounds_for_monitors(v_fts: float):
            lb = -1.0e9
            ub =  1.0e9
            v_fts = float(max(0.0, v_fts))

            for p in monitor_points:
                try:
                    mp_p = float(p.get('mp'))
                except Exception:
                    continue
                if mp_p < pig_mp_now - 1e-9:
                    continue

                try:
                    elev_p = float(p.get('elev', cs(mp_p)))
                except Exception:
                    elev_p = float(np.interp(mp_p, purge_mileposts, elevations))

                L_seg_ft = max(0.0, (float(cfg['system_end_mp']) - mp_p) * 5280.0)
                fr_seg = slug_friction_psi_SI(L_seg_ft, _pdiam_i, v_fts, api_for_dp, viscosity_cst, _rough_i, sg_fallback=specific_gravity) if v_fts > 1e-10 else 0.0
                head_seg = float(fluid_density) * (float(h_exit) - elev_p) / 144.0
                coeff = float(head_seg) + float(fr_seg)

                max_lim = p.get('max_psig', None)
                if max_lim is not None:
                    try:
                        ub = min(ub, float(max_lim) - coeff)
                    except Exception:
                        pass

                min_lim = p.get('min_psig', None)
                if min_lim is not None:
                    try:
                        lb = max(lb, float(min_lim) - coeff)
                    except Exception:
                        pass

                lb = max(lb, float(slack_threshold_psig) - coeff)

            if exit_min_clamp is not None:
                lb = max(lb, float(exit_min_clamp))
            if exit_max_clamp is not None:
                ub = min(ub, float(exit_max_clamp))

            return lb, ub

        P_end_use = float(P_end)

        if endpoint_mode == "clamp_to_monitors":
            P_try = float(P_end_use)
            sol_tmp = None
            for _ in range(4):
                sol_tmp = solve_boundary(cfg['system_end_mp'], P_try, h_exit, label="OFF_CLAMP")
                if not sol_tmp.get('feasible', False):
                    break
                v_tmp = float(sol_tmp.get('v', 0.0))
                lb, ub = _exit_bounds_for_monitors(v_tmp)
                exit_pressures_lb[i] = lb
                exit_pressures_ub[i] = ub

                if lb > ub:
                    logging.warning(
                        f"Endpoint constraint infeasible at pig MP {pig_mp_now:.3f}: "
                        f"LB={lb:.2f} psig > UB={ub:.2f} psig. Clamping to UB to protect MAOP."
                    )
                    P_new = min(float(P_end_prog), float(ub))
                else:
                    P_new = min(max(float(P_end_prog), float(lb)), float(ub))

                if abs(P_new - P_try) < 0.05:
                    P_try = float(P_new)
                    break
                P_try = float(P_new)

            P_end_use = float(P_try)

        if (exit_ramp is not None) and (P_end_prev is not None):
            sol_dt = solve_boundary(cfg['system_end_mp'], P_end_use, h_exit, label="OFF_DT")
            if sol_dt.get('feasible', False):
                v_dt = max(1e-9, float(sol_dt.get('v', 0.0)))
                dt_est_hr = float(segment_length_ft) / v_dt / 3600.0
                max_delta = float(exit_ramp) * max(0.0, dt_est_hr)
                P_end_use = min(max(P_end_use, float(P_end_prev) - max_delta), float(P_end_prev) + max_delta)

        P_end = float(P_end_use)
        sol_off = solve_boundary(cfg['system_end_mp'], P_end, h_exit, label="OFF")
        v_off = float(sol_off['v']) if sol_off.get('feasible', False) else 0.0

        P_end_prev = float(P_end)

        use_ips = False
        active_ips = None
        ips_reason = "OFF"
        sol_use = sol_off
        P_boundary_use = P_end
        boundary_mp_use = float(cfg['system_end_mp'])
        elev_boundary_use = float(h_exit)

        next_ips = None
        if cfg.get('has_ips', False) and isinstance(ips_mps, (list, tuple)) and len(ips_mps) > 0:
            for _mp in ips_mps:
                if float(_mp) > pig_mp_now:
                    next_ips = float(_mp)
                    break

        in_shutdown_zone = False
        if next_ips is not None and ips_shutdown_dist > 0.0:
            in_shutdown_zone = (pig_mp_now >= (next_ips - ips_shutdown_dist))

        if next_ips is not None and (not in_shutdown_zone):
            if (not sol_off['feasible']) or (v_off < (v_target * 0.995)):
                h_ips = float(cs(next_ips))
                P_suction_set = max(float(min_pump_suction_pressure), float(slack_threshold_psig))
                sol_on = solve_boundary(next_ips, P_suction_set, h_ips, label="ON")
                v_on = float(sol_on['v']) if sol_on['feasible'] else 0.0

                if sol_on['feasible'] and (next_ips < float(cfg['system_end_mp']) - 1e-9):
                    L_down_ft = max(0.0, (float(cfg['system_end_mp']) - next_ips) * 5280.0)
                    head_down = float(fluid_density) * (float(h_exit) - h_ips) / 144.0
                    fr_down = slug_friction_psi_SI(L_down_ft, _pdiam_i, v_on, api_for_dp, viscosity_cst, _rough_i, sg_fallback=specific_gravity)
                    P_discharge_req = max(float(P_end), float(P_end) + head_down + fr_down)
                    dp_pump_req = float(P_discharge_req) - float(P_suction_set)
                else:
                    P_discharge_req = float(P_end)
                    dp_pump_req = 0.0

                bph_on = _bph_from_v(v_on, _area_i)
                minflow_ok = True if (min_pump_flow_bph is None) else (bph_on >= float(min_pump_flow_bph) - 1e-9)
                seals_check = (dp_pump_req > float(ips_check_dp_psi) + 1e-9)

                if sol_on['feasible'] and minflow_ok and seals_check and (v_on > v_off + 1e-9):
                    use_ips = True
                    active_ips = next_ips
                    sol_use = sol_on
                    P_boundary_use = float(P_suction_set)
                    boundary_mp_use = float(next_ips)
                    elev_boundary_use = float(h_ips)
                    ips_reason = "ON" if (v_on >= v_target * 0.995) else "ON_FLOW_SHORTFALL"

                    ips_active_mp[i] = float(next_ips)
                    ips_suction_psig[i] = float(P_suction_set)
                    ips_discharge_req_psig[i] = float(P_discharge_req)
                    ips_pump_dp_req_psi[i] = float(dp_pump_req)
                    ips_flow_bph[i] = float(bph_on)
                    ips_state[i] = ips_reason
                else:
                    if in_shutdown_zone:
                        ips_state[i] = "SHUTDOWN_ZONE"
                    elif not sol_on['feasible']:
                        ips_state[i] = "ON_STALL"
                    elif not minflow_ok:
                        ips_state[i] = "ON_BELOW_MINFLOW"
                    elif not seals_check:
                        ips_state[i] = "CHECK_NOT_SEALED"
                    else:
                        ips_state[i] = "NO_BENEFIT"
            else:
                ips_state[i] = "DRIVE_SUFFICIENT"
        elif next_ips is not None and in_shutdown_zone:
            ips_state[i] = "SHUTDOWN_ZONE"
        else:
            ips_state[i] = "NO_STATION_AHEAD"

        v = float(sol_use['v']) if sol_use['feasible'] else 0.0
        P_behind_pig_req = float(sol_use['P_behind_req'])
        P_behind_pig_out = float(sol_use['P_behind_out'])
        P_inj_out = float(sol_use['P_inj'])
        dp_gas = float(sol_use['dp_gas'])
        q_use = float(sol_use['q_use'])
        fr_use = float(sol_use['fr_liq'])
        head_use = float(sol_use['head'])

        injection_pressures[i] = P_inj_out
        behind_pig_pressures[i] = P_behind_pig_out
        differential_pressures[i] = P_inj_out - P_end
        injection_rates[i] = 0.0 if (coast_mode and cutoff_value is not None) else q_use
        pig_speeds[i] = v / 1.46667
        friction_losses[i] = fr_use
        exit_pressures[i] = float(P_end)
        miles_to_outlet[i] = float(cfg['system_end_mp'] - pig_mp_now)
        gas_dp_losses[i] = dp_gas

        hydraulic_boundary_mps[i] = float(boundary_mp_use)
        hydraulic_boundary_pressures[i] = float(P_boundary_use)
        head_to_boundary[i] = float(head_use)
        ips_flow_bph[i] = _bph_from_v(v, _area_i) if not use_ips else ips_flow_bph[i]
        if not use_ips:
            ips_flow_bph[i] = _bph_from_v(v, _area_i)

        # N2 inventory (total SCF in gas column at this step)
        _active_bst_now = [b for b in booster_stations if float(b['milepost']) < pig_mp_now]
        if _active_bst_now and sol_use.get('booster_states') and sol_use.get('seg_pressures'):
            n2_inventory_scf[i] = compute_n2_inventory_scf(
                pig_mp_now, cfg['purge_start_mp'], _area_i, _active_bst_now,
                sol_use.get('seg_pressures', [(P_inj_out, P_behind_pig_out)]),
                n2_temp_f
            )
            booster_diagnostics[i] = {'booster_states': sol_use.get('booster_states', [])}
        else:
            L_gas_now = max(0.0, (pig_mp_now - cfg['purge_start_mp']) * 5280.0)
            V_gas_now = _area_i * L_gas_now
            P_avg_psia = 0.5 * (P_inj_out + P_behind_pig_out) + ATM_PSI
            Z_avg = max(1e-6, z_factor_n2_pr(P_avg_psia, n2_temp_f))
            Z_std2 = max(1e-6, z_factor_n2_pr(ATM_PSI, 60.0))
            n2_inventory_scf[i] = V_gas_now * P_avg_psia / ATM_PSI * Z_std2 / Z_avg


        # --- Hypothetical blocked-in static check ---
        hgl_static_liq_ft = np.nan
        gas_static_psig = np.nan
        if bool(cfg.get('static_block_in_enabled', True)):
            try:
                elev_pig = float(cs(pig_mp_now))
            except Exception:
                elev_pig = float(np.interp(pig_mp_now, system_mileposts, system_elevations))

            try:
                n2_scf_now = float(cumulative_n2[i])
            except Exception:
                n2_scf_now = 0.0
            n2_moles_now = n2_moles_from_scf(max(0.0, n2_scf_now))
            L_gas_ft_static = max(1e-6, (pig_mp_now - float(cfg['purge_start_mp'])) * 5280.0)
            V_gas_ft3_static = max(1e-6, float(_area_i) * float(L_gas_ft_static))
            try:
                P_trap_now = float(pressure_psig_from_moles(n2_moles_now, V_gas_ft3_static, cfg.get('temperature_f', 60.0)))
            except Exception:
                P_trap_now = float(P_behind_pig_out)
            P_gas_static = float(max(float(P_behind_pig_out), float(P_trap_now)))
            P_gas_static = float(max(0.0, P_gas_static))

            gas_static_psig = P_gas_static
            hgl_static_liq_ft = float(elev_pig + (P_gas_static * 144.0) / float(fluid_density))
            static_hgl_ft[i] = hgl_static_liq_ft

            gas_static_psig_maxdrive = gas_static_psig
            hgl_static_liq_ft_maxdrive = hgl_static_liq_ft
            if bool(cfg.get('static_block_in_assume_max_drive', True)):
                assumed_drive = float(cfg.get('max_nitrogen_pressure', cfg.get('max_drive_pressure', 0.0)) or 0.0)
                if bool(cfg.get('static_assume_respects_safe_envelope', True)) and (max_behind_cap_psig_step is not None):
                    assumed_drive = min(float(assumed_drive), float(max_behind_cap_psig_step))
                P_gas_static_maxdrive = float(max(float(P_gas_static), float(assumed_drive)))
                gas_static_psig_maxdrive = P_gas_static_maxdrive
                hgl_static_liq_ft_maxdrive = float(elev_pig + (P_gas_static_maxdrive * 144.0) / float(fluid_density))
            static_hgl_ft_maxdrive[i] = hgl_static_liq_ft_maxdrive

            try:
                sys_mps_arr = np.asarray(system_mileposts, dtype=float)
                sys_elev_arr = np.asarray(system_elevations, dtype=float)
                P_static_liq_all = (hgl_static_liq_ft - sys_elev_arr) * float(fluid_density) / 144.0
                liq_mask = sys_mps_arr >= pig_mp_now - 1e-9
                max_liq = np.nan
                max_liq_mp = np.nan
                max_liq_elev = np.nan
                if np.any(liq_mask):
                    masked = np.where(liq_mask, P_static_liq_all, -np.inf)
                    j = int(np.nanargmax(masked))
                    max_liq = float(masked[j])
                    max_liq_mp = float(sys_mps_arr[j])
                    max_liq_elev = float(sys_elev_arr[j])
            except Exception:
                max_liq = np.nan
                max_liq_mp = np.nan
                max_liq_elev = np.nan

            try:
                elev_inj = float(cs(float(cfg['purge_start_mp'])))
            except Exception:
                elev_inj = float(np.interp(float(cfg['purge_start_mp']), system_mileposts, system_elevations))

            gas_max = float(P_gas_static)
            gas_max_mp = float(cfg['purge_start_mp'])
            gas_max_elev = float(elev_inj)

            if np.isfinite(max_liq) and (float(max_liq) >= float(gas_max)):
                static_max_any_psig[i] = float(max_liq)
                static_max_any_mp[i] = float(max_liq_mp)
                static_max_any_elev_ft[i] = float(max_liq_elev)
                static_max_any_phase[i] = 'liquid'
            else:
                static_max_any_psig[i] = float(gas_max)
                static_max_any_mp[i] = float(gas_max_mp)
                static_max_any_elev_ft[i] = float(gas_max_elev)
                static_max_any_phase[i] = 'gas'

            if (maop_psig is not None) and np.isfinite(static_max_any_psig[i]) and (float(static_max_any_psig[i]) > float(maop_psig) + 1e-6):
                static_violations.append({
                    'time_hr': float(elapsed_times[i]),
                    'pig_mp': float(pig_mp_now),
                    'phase': str(static_max_any_phase[i]),
                    'max_static_psig': float(static_max_any_psig[i]),
                    'max_static_mp': float(static_max_any_mp[i]),
                    'max_static_elev_ft': float(static_max_any_elev_ft[i]),
                    'maop_psig': float(maop_psig),
                    'margin_psig': float(static_max_any_psig[i]) - float(maop_psig),
                })

            if np.isfinite(hgl_static_liq_ft_maxdrive):
                try:
                    sys_mps_arr_md = np.asarray(system_mileposts, dtype=float)
                    sys_elev_arr_md = np.asarray(system_elevations, dtype=float)
                    P_static_liq_all_md = (hgl_static_liq_ft_maxdrive - sys_elev_arr_md) * float(fluid_density) / 144.0
                    liq_mask_md = (sys_mps_arr_md >= (pig_mp_now - 1e-9))
                    max_liq_md = np.nan
                    max_liq_mp_md = np.nan
                    max_liq_elev_md = np.nan
                    if np.any(liq_mask_md):
                        masked_md = np.where(liq_mask_md, P_static_liq_all_md, -np.inf)
                        j_md = int(np.nanargmax(masked_md))
                        max_liq_md = float(masked_md[j_md])
                        max_liq_mp_md = float(sys_mps_arr_md[j_md])
                        max_liq_elev_md = float(sys_elev_arr_md[j_md])
                except Exception:
                    max_liq_md = np.nan
                    max_liq_mp_md = np.nan
                    max_liq_elev_md = np.nan

                try:
                    elev_inj_md = float(cs(float(cfg['purge_start_mp'])))
                except Exception:
                    elev_inj_md = float(np.interp(float(cfg['purge_start_mp']), system_mileposts, system_elevations))

                gas_max_md = float(gas_static_psig_maxdrive)
                gas_max_mp_md = float(cfg['purge_start_mp'])
                gas_max_elev_md = float(elev_inj_md)

                if np.isfinite(max_liq_md) and (float(max_liq_md) >= float(gas_max_md)):
                    static_max_any_psig_maxdrive[i] = float(max_liq_md)
                    static_max_any_mp_maxdrive[i] = float(max_liq_mp_md)
                    static_max_any_elev_ft_maxdrive[i] = float(max_liq_elev_md)
                    static_max_any_phase_maxdrive[i] = 'liquid'
                else:
                    static_max_any_psig_maxdrive[i] = float(gas_max_md)
                    static_max_any_mp_maxdrive[i] = float(gas_max_mp_md)
                    static_max_any_elev_ft_maxdrive[i] = float(gas_max_elev_md)
                    static_max_any_phase_maxdrive[i] = 'gas'

                if (maop_psig is not None) and np.isfinite(static_max_any_psig_maxdrive[i]) and (float(static_max_any_psig_maxdrive[i]) > float(maop_psig) + 1e-6):
                    static_violations_maxdrive.append({
                        'time_hr': float(elapsed_times[i]),
                        'pig_mp': float(pig_mp_now),
                        'phase': str(static_max_any_phase_maxdrive[i]),
                        'max_static_psig': float(static_max_any_psig_maxdrive[i]),
                        'max_static_mp': float(static_max_any_mp_maxdrive[i]),
                        'max_static_elev_ft': float(static_max_any_elev_ft_maxdrive[i]),
                        'maop_psig': float(maop_psig),
                        'margin_psig': float(static_max_any_psig_maxdrive[i]) - float(maop_psig),
                        'mode': 'ASSUME_MAX_DRIVE'
                    })

        # --- Evaluate monitor point pressures at this step ---
        for p in monitor_points:
            mp_p = float(p['mp'])
            elev_p = float(p.get('elev', 0.0))
            label_p = p['name']

            is_liquid = (mp_p >= pig_mp_now - 1e-9)

            if is_liquid:
                if use_ips and (active_ips is not None) and (mp_p >= float(active_ips) - 1e-9):
                    mp_ref = float(cfg['system_end_mp'])
                    elev_ref = float(h_exit)
                    P_ref = float(P_end)
                elif use_ips and (active_ips is not None):
                    mp_ref = float(active_ips)
                    elev_ref = float(elev_boundary_use)
                    P_ref = float(P_boundary_use)
                else:
                    mp_ref = float(cfg['system_end_mp'])
                    elev_ref = float(h_exit)
                    P_ref = float(P_end)

                L_seg_ft = max(0.0, (mp_ref - mp_p) * 5280.0)
                fr_seg = slug_friction_psi_SI(L_seg_ft, _pdiam_i, v, api_for_dp, viscosity_cst, _rough_i, sg_fallback=specific_gravity) if v > 1e-12 else 0.0
                head_seg = float(fluid_density) * (elev_ref - elev_p) / 144.0
                P_p = float(P_ref) + head_seg + fr_seg
            else:
                P_p = float(P_behind_pig_out)

            monitor_pressures[p['name']][i] = P_p

            if np.isfinite(hgl_static_liq_ft):
                if mp_p >= pig_mp_now - 1e-9:
                    monitor_pressures_static[p['name']][i] = (hgl_static_liq_ft - elev_p) * float(fluid_density) / 144.0
                else:
                    monitor_pressures_static[p['name']][i] = float(gas_static_psig)

            if np.isfinite(hgl_static_liq_ft_maxdrive):
                if mp_p >= pig_mp_now - 1e-9:
                    monitor_pressures_static_maxdrive[p['name']][i] = (hgl_static_liq_ft_maxdrive - elev_p) * float(fluid_density) / 144.0
                else:
                    monitor_pressures_static_maxdrive[p['name']][i] = float(gas_static_psig_maxdrive)

            max_lim = p.get('max_psig', None)
            if max_lim is not None and P_p > max_lim + 1e-6:
                exceed_points.append({
                    'mp': mp_p, 'label': label_p, 'rule': 'MAOP',
                    'limit_psig': max_lim, 'pressure_psig': P_p, 'margin_psig': P_p - max_lim
                })
                monitor_violations.append({
                    'time_hr': float(elapsed_times[i]), 'pig_mp': pig_mp_now,
                    'point_name': p['name'], 'point_mp': mp_p, 'point_elev': elev_p,
                    'phase': 'liquid' if is_liquid else 'gas', 'rule': 'MAX_PRESSURE',
                    'limit_psig': float(max_lim), 'pressure_psig': float(P_p),
                    'margin_psig': float(P_p) - float(max_lim)
                })
            min_lim = p.get('min_psig', None)
            if (min_lim is not None) and is_liquid and (P_p < min_lim - 1e-6):
                exceed_points.append({
                    'mp': mp_p, 'label': label_p, 'rule': 'MIN',
                    'limit_psig': min_lim, 'pressure_psig': P_p, 'margin_psig': P_p - min_lim
                })
                monitor_violations.append({
                    'time_hr': float(elapsed_times[i]), 'pig_mp': pig_mp_now,
                    'point_name': p['name'], 'point_mp': mp_p, 'point_elev': elev_p,
                    'phase': 'liquid' if is_liquid else 'gas', 'rule': 'MIN_PRESSURE',
                    'limit_psig': float(min_lim), 'pressure_psig': float(P_p),
                    'margin_psig': float(P_p) - float(min_lim)
                })

            if is_liquid and (P_p < float(slack_threshold_psig) - 1e-6):
                exceed_points.append({
                    'mp': mp_p, 'label': label_p, 'rule': 'SLACK',
                    'limit_psig': float(slack_threshold_psig), 'pressure_psig': P_p,
                    'margin_psig': P_p - float(slack_threshold_psig)
                })
                monitor_violations.append({
                    'time_hr': float(elapsed_times[i]), 'pig_mp': pig_mp_now,
                    'point_name': p['name'], 'point_mp': mp_p, 'point_elev': elev_p,
                    'phase': 'liquid', 'rule': 'SLACK',
                    'limit_psig': float(slack_threshold_psig), 'pressure_psig': float(P_p),
                    'margin_psig': float(P_p) - float(slack_threshold_psig)
                })

        if v <= 1e-8:
            last_valid_i = i
            logging.warning(f"Near-zero velocity at MP {purge_mileposts[i]:.2f}; stopping.")
            break
        time_step_hours = segment_length_ft / v / 3600.0
        elapsed_times[i + 1] = elapsed_times[i] + time_step_hours

        if elapsed_times[i + 1] > max_elapsed_time_hr:
            stop_reason = f"MAX_ELAPSED_TIME>{max_elapsed_time_hr:.0f}hr"
            last_valid_i = min(i + 1, n_points - 1)
            logging.warning(
                f"Max elapsed time cap reached ({max_elapsed_time_hr:.0f} hr) at MP={purge_mileposts[last_valid_i]:.3f}. "
                "Stopping early."
            )
            break

        if coast_mode and cutoff_value is not None:
            cumulative_n2[i] = cutoff_value
        n2_added_scf = 0.0 if coast_mode else (q_use * time_step_hours * 60.0)
        cumulative_n2_next_raw = cumulative_n2[i] + n2_added_scf
        cumulative_n2[i + 1] = cumulative_n2_next_raw

        cumulative_distance += segment_length_ft
        if (not coast_mode) and (cutoff_value is not None) and (cumulative_n2_next_raw >= cutoff_value):
            cumulative_n2[i + 1] = cutoff_value
            coast_mode = True

            frac_to_cutoff = 1.0
            if n2_added_scf > 1e-12:
                frac_to_cutoff = max(0.0, min(1.0, (float(cutoff_value) - float(cumulative_n2[i])) / float(n2_added_scf)))
            mp_cutoff = float(purge_mileposts[i]) + frac_to_cutoff * float(purge_mileposts[i + 1] - purge_mileposts[i])
            L_gas_cutoff_ft = max(1e-6, (mp_cutoff - cfg['purge_start_mp']) * 5280.0)
            V_cutoff_ft3 = max(1e-6, _area_i * L_gas_cutoff_ft)
            p_cutoff_psia = max(ATM_PSI + 1e-6, float(P_behind_pig_out) + ATM_PSI)

            coast_start_p_psig = float(P_behind_pig_out)
            coast_start_V_ft3 = float(V_cutoff_ft3)
            n2_moles_total = n2_moles_from_pressure_volume(p_cutoff_psia, V_cutoff_ft3, cfg.get('temperature_f', 60.0))

            L_gas_next_ft = max(1e-6, (purge_mileposts[i + 1] - cfg['purge_start_mp']) * 5280.0)
            V_gas_ft3 = max(V_cutoff_ft3, _area_i * L_gas_next_ft)

        elif coast_mode and V_gas_ft3 is not None:
            V_gas_ft3 += _area_i * segment_length_ft

    if stop_reason is None and max_steps < (n_points - 1):
        stop_reason = f"MAX_STEPS({max_steps})"
        last_valid_i = min(max_steps, n_points - 1)
        logging.warning(f"Simulation stopped due to max_sim_steps cap: {stop_reason}")

    j = min(last_valid_i, n_points - 1)
    try:
        mp_j = purge_mileposts[j]
        P_exit_j = target_exit_pressure(mp_j, cfg, cfg['purge_start_mp'], cfg['purge_end_mp'], cfg['throttle_down_miles'])
        head_j = head_losses[j]
        P_behind_pig_static = max(P_exit_j, P_exit_j + head_j)

        if coast_mode and (n2_moles_total is not None) and (V_gas_ft3 is not None):
            p_psia = n2_pressure_psia_from_moles(n2_moles_total, V_gas_ft3, n2_temp_f)
            P_trap_psig = max(0.0, p_psia - ATM_PSI)
            behind_pig_pressures[j] = P_trap_psig
            injection_pressures[j] = P_trap_psig
            gas_dp_losses[j] = 0.0
        else:
            behind_pig_pressures[j] = P_behind_pig_static
            injection_pressures[j] = P_behind_pig_static
            gas_dp_losses[j] = 0.0

        exit_pressures[j] = P_exit_j
        friction_losses[j] = 0.0
        pig_speeds[j] = 0.0
        injection_rates[j] = 0.0
        if coast_mode and cutoff_value is not None:
            cumulative_n2[j] = cutoff_value
    except Exception:
        pass

    try:
        if monitor_points:
            pig_mp_now = float(purge_mileposts[j])
            for p in monitor_points:
                mp_p = float(p['mp'])
                elev_p = float(p.get('elev', cs(mp_p)))
                phase = 'liquid' if (mp_p >= pig_mp_now - 1e-9) else 'gas'
                if phase == 'liquid':
                    head_p = fluid_density * (float(h_exit) - elev_p) / 144.0
                    P_p = float(exit_pressures[j]) + float(head_p)
                else:
                    P_p = float(behind_pig_pressures[j])
                monitor_pressures[p['name']][j] = float(P_p)
    except Exception:
        pass

    alarms = []
    alarm_events = []

    if last_valid_i < (n_points - 1):
        alarms.append('STALL')
        alarm_events.append({
            'alarm': 'STALL',
            'time_hr': float(elapsed_times[last_valid_i]),
            'pig_mp': float(purge_mileposts[last_valid_i]),
            'detail': 'Near-zero velocity; simulation stopped early.'
        })

    if stop_reason and 'TIMEOUT' in str(stop_reason):
        alarms.append('TIMEOUT')
        alarm_events.append({
            'alarm': 'TIMEOUT',
            'time_hr': float(elapsed_times[last_valid_i]) if last_valid_i >= 0 else 0.0,
            'pig_mp': float(purge_mileposts[last_valid_i]) if last_valid_i >= 0 else float(purge_mileposts[0]),
            'detail': f"Stopped due to wall-clock timeout ({stop_reason})."
        })
    if stop_reason and 'MAX_ELAPSED_TIME' in str(stop_reason):
        alarms.append('MAX_ELAPSED_TIME')
        alarm_events.append({
            'alarm': 'MAX_ELAPSED_TIME',
            'time_hr': float(elapsed_times[last_valid_i]) if last_valid_i >= 0 else 0.0,
            'pig_mp': float(purge_mileposts[last_valid_i]) if last_valid_i >= 0 else float(purge_mileposts[0]),
            'detail': f"Stopped due to max elapsed-time cap ({stop_reason})."
        })
    if stop_reason and 'MAX_STEPS' in str(stop_reason):
        alarms.append('MAX_STEPS')
        alarm_events.append({
            'alarm': 'MAX_STEPS',
            'time_hr': float(elapsed_times[last_valid_i]) if last_valid_i >= 0 else 0.0,
            'pig_mp': float(purge_mileposts[last_valid_i]) if last_valid_i >= 0 else float(purge_mileposts[0]),
            'detail': f"Stopped due to explicit step cap ({stop_reason})."
        })

    try:
        min_speed_alarm_mph = float(cfg.get('min_pig_speed')) if cfg.get('min_pig_speed') is not None else None
    except Exception:
        min_speed_alarm_mph = None

    if min_speed_alarm_mph is not None and last_valid_i >= 0:
        speeds = np.asarray(pig_speeds[:last_valid_i + 1], dtype=float)
        if np.any(np.isfinite(speeds)):
            jmin = int(np.nanargmin(speeds))
            vmin = float(speeds[jmin])
            if vmin < (min_speed_alarm_mph - 1e-9):
                alarms.append('BELOW_MIN_SPEED')
                alarm_events.append({
                    'alarm': 'BELOW_MIN_SPEED',
                    'time_hr': float(elapsed_times[jmin]),
                    'pig_mp': float(purge_mileposts[jmin]),
                    'detail': f"Pig speed fell below min_pig_speed alarm threshold ({min_speed_alarm_mph:.3g} mph).",
                    'value': vmin,
                    'limit': min_speed_alarm_mph
                })

    try:
        mv = monitor_violations or []
        if any(v.get('rule') == 'SLACK' for v in mv):
            alarms.append('SLACK')
        if any(v.get('rule') == 'MAX_PRESSURE' for v in mv):
            alarms.append('MAOP_EXCEED')
        if any(v.get('rule') == 'MIN_PRESSURE' for v in mv):
            alarms.append('MIN_PRESSURE')
    except Exception:
        pass

    _seenA = set()
    alarms = [a for a in alarms if not (a in _seenA or _seenA.add(a))]
    logging.info("Exiting run_simulation")
    return {
        'purge_mileposts': purge_mileposts[:last_valid_i + 1],
        'elevations': elevations[:last_valid_i + 1],
        'elapsed_times': elapsed_times[:last_valid_i + 1],
        'behind_pig_pressures': behind_pig_pressures[:last_valid_i + 1],
        'drive_pressures': behind_pig_pressures[:last_valid_i + 1],
        'injection_pressures': injection_pressures[:last_valid_i + 1],
        'gas_dp_losses': gas_dp_losses[:last_valid_i + 1],
        'friction_losses': friction_losses[:last_valid_i + 1],
        'head_losses': head_losses[:last_valid_i + 1],
        'exit_pressures': exit_pressures[:last_valid_i + 1],
        'exit_pressures_programmed': exit_pressures_programmed[:last_valid_i + 1],
        'exit_pressures_lb': exit_pressures_lb[:last_valid_i + 1],
        'exit_pressures_ub': exit_pressures_ub[:last_valid_i + 1],
        'injection_rates': injection_rates[:last_valid_i + 1],
        'cumulative_n2': cumulative_n2[:last_valid_i + 1],
        'differential_pressures': differential_pressures[:last_valid_i + 1],
        'miles_to_outlet': miles_to_outlet[:last_valid_i + 1],
        'pig_speeds': pig_speeds[:last_valid_i + 1],
        'n2_inventory_scf': n2_inventory_scf[:last_valid_i + 1],
        'booster_stations': booster_stations,
        'booster_diagnostics': booster_diagnostics[:last_valid_i + 1],
        'monitor_points': monitor_points,
        'monitor_pressures': {k: v[:last_valid_i + 1] for k, v in monitor_pressures.items()},
        'monitor_violations': monitor_violations,
        'monitor_pressures_static': {k: v[:last_valid_i + 1] for k, v in monitor_pressures_static.items()},
        'monitor_pressures_static_maxdrive': ({k: v[:last_valid_i + 1] for k, v in monitor_pressures_static_maxdrive.items()} if bool(cfg.get('static_block_in_assume_max_drive', True)) else {}),
        'static_hgl_ft_maxdrive': (static_hgl_ft_maxdrive[:last_valid_i + 1] if bool(cfg.get('static_block_in_assume_max_drive', True)) else np.array([])),
        'static_max_any_psig_maxdrive': (static_max_any_psig_maxdrive[:last_valid_i + 1] if bool(cfg.get('static_block_in_assume_max_drive', True)) else np.array([])),
        'static_max_any_mp_maxdrive': (static_max_any_mp_maxdrive[:last_valid_i + 1] if bool(cfg.get('static_block_in_assume_max_drive', True)) else np.array([])),
        'static_max_any_elev_ft_maxdrive': (static_max_any_elev_ft_maxdrive[:last_valid_i + 1] if bool(cfg.get('static_block_in_assume_max_drive', True)) else np.array([])),
        'static_max_any_phase_maxdrive': (static_max_any_phase_maxdrive[:last_valid_i + 1] if bool(cfg.get('static_block_in_assume_max_drive', True)) else np.array([], dtype=object)),
        'static_violations_maxdrive': (static_violations_maxdrive if bool(cfg.get('static_block_in_assume_max_drive', True)) else []),
        'safe_max_drive_psig': (safe_max_drive_envelope[:last_valid_i + 1] if safe_max_drive_envelope is not None else None),
        'static_hgl_ft': static_hgl_ft[:last_valid_i + 1],
        'static_max_any_psig': static_max_any_psig[:last_valid_i + 1],
        'static_max_any_mp': static_max_any_mp[:last_valid_i + 1],
        'static_max_any_elev_ft': static_max_any_elev_ft[:last_valid_i + 1],
        'static_max_any_phase': static_max_any_phase[:last_valid_i + 1],
        'static_violations': static_violations,
        'slack_threshold_psig': slack_threshold_psig,
        'maop_psig': maop_psig,
        'static_maxdrive_mode': (
            "OFF" if not bool(cfg.get('static_block_in_assume_max_drive', True))
            else ("MaxDrive_WorstCaseIgnoreEnvelope" if not bool(cfg.get('static_assume_respects_safe_envelope', True))
                  else "MaxDrive_RespectSafeEnvelope")
        ),
        'exceed_points': exceed_points,
        'hydraulic_boundary_mps': hydraulic_boundary_mps[:last_valid_i + 1],
        'hydraulic_boundary_pressures': hydraulic_boundary_pressures[:last_valid_i + 1],
        'head_to_boundary': head_to_boundary[:last_valid_i + 1],
        'ips_active_mp': ips_active_mp[:last_valid_i + 1],
        'ips_suction_psig': ips_suction_psig[:last_valid_i + 1],
        'ips_discharge_req_psig': ips_discharge_req_psig[:last_valid_i + 1],
        'ips_pump_dp_req_psi': ips_pump_dp_req_psi[:last_valid_i + 1],
        'ips_flow_bph': ips_flow_bph[:last_valid_i + 1],
        'ips_state': ips_state[:last_valid_i + 1],
        'alarms': alarms,
        'alarm_events': alarm_events,
        'last_valid_i': last_valid_i,
        'system_mileposts': system_mileposts,
        'system_elevations': system_elevations,
    }


# ======================== HYDRAULIC ANIMATION ========================

def _ft_per_psi_liq(fluid_density_lbft3: float) -> float:
    """Feet of liquid head per psig: 144 / density_lb_ft3"""
    return 144.0 / max(1.0, float(fluid_density_lbft3))


def compute_liquid_hgl(
    pig_mp, end_mp, P_end_psig, v_fts,
    sys_mps, sys_elevs,
    pipe_diam_ft, roughness_ft,
    fluid_density_lbft3, api_gravity, viscosity_cSt,
    ips_active_mp=None, ips_suction_psig=None, ips_discharge_psig=None,
):
    """
    Compute hydraulic grade line (total head, ft) for the liquid section.

    When an IPS is active the HGL has two disjoint segments:
      • pig → IPS suction  (upstream of pump)
      • IPS discharge → delivery end  (downstream of pump)
    These are stitched together with a NaN separator so that matplotlib renders
    them as two separate lines, and the caller draws the vertical pump-head jump
    independently.

    Returns (mps, head_ft, P_psig) arrays. NaN entries act as line-break markers.
    """
    fppsi = _ft_per_psi_liq(fluid_density_lbft3)
    sg_fb = fluid_density_lbft3 / 62.4
    sys_mps   = np.asarray(sys_mps,   dtype=float)
    sys_elevs = np.asarray(sys_elevs, dtype=float)

    def _integrate_upstream(mp_start, mp_end, P_at_downstream):
        """Return (mps, head_ft) for one segment, integrating upstream."""
        mask = (sys_mps >= mp_start - 0.01) & (sys_mps <= mp_end + 0.01)
        lm_s = sys_mps[mask]
        le_s = sys_elevs[mask]
        n_s  = len(lm_s)
        if n_s < 2:
            h = np.array([
                float(np.interp(mp_start, sys_mps, sys_elevs)) + P_at_downstream * fppsi,
                float(np.interp(mp_end,   sys_mps, sys_elevs)) + P_at_downstream * fppsi,
            ])
            return np.array([mp_start, mp_end]), h
        head_s = np.full(n_s, np.nan)
        head_s[-1] = le_s[-1] + P_at_downstream * fppsi
        for k in range(n_s - 2, -1, -1):
            L_ft   = max(0.0, (lm_s[k + 1] - lm_s[k]) * 5280.0)
            fr_psi = (slug_friction_psi_SI(L_ft, pipe_diam_ft, v_fts,
                                            api_gravity, viscosity_cSt, roughness_ft,
                                            sg_fallback=sg_fb)
                      if (v_fts > 1e-9 and L_ft > 0.0) else 0.0)
            head_s[k] = head_s[k + 1] + fr_psi * fppsi
        return lm_s, head_s

    has_ips = (ips_active_mp is not None and ips_suction_psig is not None
               and ips_discharge_psig is not None
               and pig_mp < float(ips_active_mp) < end_mp)

    if has_ips:
        ips_mp = float(ips_active_mp)
        # Upstream segment: pig → IPS suction (P at IPS = ips_suction_psig)
        lm_up, hd_up = _integrate_upstream(pig_mp, ips_mp, float(ips_suction_psig))
        # Downstream segment: IPS discharge → delivery end
        lm_dn, hd_dn = _integrate_upstream(ips_mp, end_mp, float(P_end_psig))
        # Override the IPS endpoint in the downstream array to discharge pressure
        ips_elev_dn = float(np.interp(ips_mp, sys_mps, sys_elevs))
        hd_dn[0]    = ips_elev_dn + float(ips_discharge_psig) * fppsi

        # Stitch with NaN separator so matplotlib draws two distinct lines
        nan_row = np.array([np.nan])
        lm_out = np.concatenate([lm_up, nan_row, lm_dn])
        hd_out = np.concatenate([hd_up, nan_row, hd_dn])
        le_out = np.concatenate([
            sys_elevs[(sys_mps >= pig_mp - 0.01) & (sys_mps <= ips_mp + 0.01)],
            nan_row,
            sys_elevs[(sys_mps >= ips_mp - 0.01) & (sys_mps <= end_mp + 0.01)],
        ])
    else:
        # Single segment: pig → delivery end
        lm_out, hd_out = _integrate_upstream(pig_mp, end_mp, float(P_end_psig))
        mask_all = (sys_mps >= pig_mp - 0.01) & (sys_mps <= end_mp + 0.01)
        le_out   = sys_elevs[mask_all]

    P_arr = np.where(np.isfinite(hd_out) & np.isfinite(le_out[:len(hd_out)]),
                     (hd_out - le_out[:len(hd_out)]) * fluid_density_lbft3 / 144.0,
                     np.nan)
    return lm_out, hd_out, P_arr


def generate_purge_animation(
    results, inputs,
    output_path=None,
    fps: int = 12,
    max_frames: int = 400,
    parent_widget=None,
):
    """
    Generate an MP4 (or GIF) hydraulic animation of the purge simulation.

    Shows per time-step:
      • Ground elevation fill and MAOP head envelope (static)
      • N2 gas pressure as total-head line (injection → pig face)
      • Liquid hydraulic grade line (pig face → delivery end)
      • IPS pump-head vertical jump
      • Pig position marker
      • Phase fill regions (N2 / liquid)
      • Monitor-point pressure annotations
      • Status box (time, speed, SCFM, N2 cumulative)
    """
    try:
        import imageio
    except ImportError:
        messagebox.showwarning(
            "Missing Package",
            "imageio is required for animation.\n"
            "Install with:  pip install imageio[pyav]\n"
            "For GIF only:  pip install imageio[pillow]",
            parent=parent_widget,
        )
        return None

    # ── Output path ──────────────────────────────────────────────────────────
    if output_path is None:
        _rt = tk.Tk(); _rt.withdraw()
        output_path = filedialog.asksaveasfilename(
            title="Save Animation As",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("GIF Animation", "*.gif"),
                       ("All Files", "*.*")],
        )
        _rt.destroy()
        if not output_path:
            return None

    # ── Extract results ───────────────────────────────────────────────────────
    pig_mps   = np.asarray(results["purge_mileposts"],       dtype=float)
    pig_elevs = np.asarray(results["elevations"],            dtype=float)
    speeds    = np.asarray(results["pig_speeds"],            dtype=float)   # mph
    P_behind  = np.asarray(results["behind_pig_pressures"],  dtype=float)
    P_inj     = np.asarray(results["injection_pressures"],   dtype=float)
    P_exit    = np.asarray(results["exit_pressures"],        dtype=float)
    inj_rates = np.asarray(results["injection_rates"],       dtype=float)
    cum_n2    = np.asarray(results["cumulative_n2"],         dtype=float)
    times_hr  = np.asarray(results["elapsed_times"],         dtype=float)
    last_i    = int(results["last_valid_i"])
    n_steps   = last_i + 1

    def _safe_arr(key, n):
        a = results.get(key)
        if a is None:
            return np.zeros(n)
        return np.asarray(a, dtype=float)[:n]

    ips_act   = _safe_arr("ips_active_mp",         n_steps)
    ips_suc   = _safe_arr("ips_suction_psig",      n_steps)
    ips_disc  = _safe_arr("ips_discharge_req_psig", n_steps)
    mon_pts   = results.get("monitor_points",   []) or []
    mon_press = results.get("monitor_pressures", {}) or {}
    booster_st= results.get("booster_stations", []) or []
    maop      = results.get("maop_psig",  None)
    slack_p   = results.get("slack_threshold_psig", None)

    sys_mps   = np.asarray(results.get("system_mileposts",  pig_mps), dtype=float)
    sys_elevs = np.asarray(results.get("system_elevations", pig_elevs), dtype=float)

    purge_start = float(inputs["purge_start_mp"])
    end_mp      = float(inputs.get("system_end_mp", float(pig_mps[-1])))

    # ── Pipe / fluid properties ───────────────────────────────────────────────
    try:
        nps_key = resolve_nps_key(inputs["nps"])
        pipe_od = nps_data[nps_key]["OD_in"]
    except Exception:
        pipe_od = float(str(inputs.get("nps", 12))) + 0.5
    pipe_wt   = float(inputs.get("pipe_wt", 0.375))
    pipe_diam = (pipe_od - 2.0 * pipe_wt) / 12.0
    rnum      = int(inputs.get("roughness_num", 1))
    roughness = roughness_data.get(rnum, roughness_data[1])["roughness_ft"]

    fluid_num = int(inputs.get("fluid_num", 1))
    api_g     = inputs.get("api_gravity", None)
    visc      = inputs.get("viscosity_cst", None)
    if fluid_num == 3 and api_g is not None:
        sg   = 141.5 / (131.5 + float(api_g))
        visc = float(visc) if visc is not None else 5.0
    else:
        fdata = fluid_data.get(fluid_num, fluid_data[1])
        sg    = fdata.get("sg") or 0.85
        visc  = float(visc) if visc is not None else (fdata.get("viscosity_cst") or 2.0)
    density    = sg * 62.4
    fppsi      = _ft_per_psi_liq(density)
    fluid_name = fluid_data.get(fluid_num, {}).get("name", "Fluid")

    # ── IPS list for station labels ──────────────────────────────────────────
    ips_raw = inputs.get("ips_mps", []) or []
    if isinstance(ips_raw, str):
        try:   ips_raw = [float(x.strip()) for x in ips_raw.split(",") if x.strip()]
        except Exception: ips_raw = []
    ips_list = [float(m) for m in ips_raw]

    # ── Animation frame selection ─────────────────────────────────────────────
    step   = max(1, n_steps // max_frames)
    fidx   = list(range(0, n_steps, step))
    if (n_steps - 1) not in fidx:
        fidx.append(n_steps - 1)
    n_frames = len(fidx)

    # ── Y-axis limits ─────────────────────────────────────────────────────────
    y_lo = float(np.nanmin(sys_elevs)) - 400.0
    maop_hi = (float(np.nanmax(sys_elevs)) + float(maop) * fppsi + 400.0) if maop else 0.0
    inj_hi  = float(np.nanmin(sys_elevs)) + float(np.nanmax(P_inj[:n_steps])) * fppsi * 1.1
    y_hi    = max(float(np.nanmax(sys_elevs)) + 800.0, maop_hi, inj_hi)
    y_span  = y_hi - y_lo
    x_lo    = float(sys_mps[0])
    x_hi    = float(sys_mps[-1])

    maop_head  = (sys_elevs + float(maop)  * fppsi) if maop    else None
    slack_head = (sys_elevs + float(slack_p) * fppsi) if slack_p else None

    # ── Station labels ────────────────────────────────────────────────────────
    stn_pts = [(float(p["mp"]), p.get("name", "")) for p in mon_pts]
    for m in ips_list:
        if not any(abs(m - s[0]) < 0.3 for s in stn_pts):
            stn_pts.append((m, "IPS"))
    stn_pts.sort(key=lambda x: x[0])

    # ── Build figure with Agg backend (no GUI window) ─────────────────────────
    from matplotlib.figure import Figure as _MFig
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _MCAgg
    import matplotlib.patches as _mpatch

    fig    = _MFig(figsize=(16, 9), dpi=110, facecolor="white")
    canvas = _MCAgg(fig)
    ax     = fig.add_subplot(111)
    ax2    = ax.twinx()

    ax.set_facecolor("#f0f2f5")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax2.set_ylim(y_lo / fppsi, y_hi / fppsi)
    ax2.set_ylabel("Pressure (psig)", fontsize=9, color="#555")
    ax2.tick_params(axis="y", colors="#777", labelsize=8)
    ax.set_xlabel("Milepost", fontsize=11)
    ax.set_ylabel("Elevation / Total Head (ft)", fontsize=11)

    purge_nm = str(inputs.get("purge_name", "Pipeline Purge"))
    nps_str  = str(inputs.get("nps", ""))
    ax.set_title(
        f"{nps_str}\"-NPS {fluid_name} Displacement  —  {purge_nm}",
        fontsize=13, fontweight="bold", pad=8,
    )

    # Static: ground elevation fill
    ax.fill_between(sys_mps, y_lo, sys_elevs, color="#8d9c8e", alpha=0.5, zorder=1)
    ax.plot(sys_mps, sys_elevs, color="#3a3a3a", lw=1.0, zorder=2, label="Ground Elevation")

    # Static: MAOP total-head envelope
    if maop_head is not None:
        ax.plot(sys_mps, maop_head, color="#c0392b", lw=2.0, ls="--", zorder=3,
                label=f"MAOP ({maop:.0f} psig) Head")

    # Static: slack threshold
    if slack_head is not None:
        ax.plot(sys_mps, slack_head, color="#e67e22", lw=1.2, ls=":", zorder=3,
                alpha=0.8, label=f"Slack ({slack_p:.0f} psig)")

    # Static: station verticals + labels
    y_lbl = y_lo + 0.01 * y_span
    for mp_s, nm_s in stn_pts:
        if x_lo <= mp_s <= x_hi:
            ax.axvline(x=mp_s, color="#bbb", lw=0.6, ls=":", zorder=1)
            ax.text(mp_s, y_lbl, nm_s, ha="center", va="bottom",
                    fontsize=7, color="#333", clip_on=True)

    # Static: booster station markers
    for b in booster_st:
        bmp = float(b.get("milepost", 0))
        ax.axvline(x=bmp, color="#1b5e20", lw=0.8, ls="-.", zorder=2, alpha=0.7)
        ax.text(bmp, y_lbl + 0.03 * y_span, "BST", ha="center",
                fontsize=7, color="#1b5e20", clip_on=True)

    # Dynamic: phase fill rectangles (updated per frame via set_x / set_width)
    n2_rect  = _mpatch.Rectangle((x_lo, y_lo), 0, y_span,
                                   fc="#bbdefb", alpha=0.18, ec="none", zorder=0)
    liq_rect = _mpatch.Rectangle((x_lo, y_lo), x_hi - x_lo, y_span,
                                   fc="#fff9c4", alpha=0.15, ec="none", zorder=0)
    ax.add_patch(n2_rect)
    ax.add_patch(liq_rect)

    # Dynamic: N2 gas pressure head (blue line)
    n2_line,  = ax.plot([], [], color="#1565c0", lw=2.2, zorder=6, label="N2 Gas Head")
    # Dynamic: liquid HGL (orange-red)
    liq_line, = ax.plot([], [], color="#e65100", lw=2.5, zorder=6, label="Liquid HGL")
    # Dynamic: IPS pump-head vertical jump (purple)
    ips_line, = ax.plot([], [], color="#7b1fa2", lw=2.5, zorder=7, label="IPS Pump Head")

    # Dynamic: pig marker + label
    pig_dot,  = ax.plot([], [], "o", ms=14, color="#0d47a1",
                        mec="white", mew=2.0, zorder=10, label="PIG")
    pig_lbl   = ax.text(0, 0, "", fontsize=8, color="#0d47a1", ha="left",
                        va="bottom", fontweight="bold", zorder=11, clip_on=True)

    # Dynamic: phase labels
    n2_lbl_t  = ax.text(0, 0, "N₂", fontsize=15, color="#1565c0", ha="center",
                         va="center", fontweight="bold", alpha=0.55, zorder=5, clip_on=True)
    liq_lbl_t = ax.text(0, 0, fluid_name, fontsize=15, color="#bf360c", ha="center",
                         va="center", fontweight="bold", alpha=0.45, zorder=5, clip_on=True)

    # Dynamic: monitor pressure annotations
    mon_texts = {}
    for p in mon_pts:
        t = ax.text(float(p["mp"]), y_lo, "", fontsize=8, ha="center", va="bottom",
                    color="#1a237e", fontweight="bold", zorder=11, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        mon_texts[p["name"]] = t

    # Dynamic: status box (top-right)
    status = ax.text(
        0.99, 0.99, "", transform=ax.transAxes, va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.92),
        zorder=12, fontfamily="monospace",
    )

    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.88, ncol=2)
    fig.tight_layout()

    # ── Pre-compute liquid HGL for every animation frame ─────────────────────
    print(f"[Animation] Pre-computing HGL for {n_frames} frames…")
    hgl_cache = []
    for fn, i in enumerate(fidx):
        pig_mp_i = float(pig_mps[i])
        v_i      = float(speeds[i]) * 5280.0 / 3600.0   # mph → ft/s
        P_e_i    = float(P_exit[i])
        im = float(ips_act[i])  if np.isfinite(ips_act[i])  and ips_act[i]  > 0 else None
        is_= float(ips_suc[i])  if np.isfinite(ips_suc[i])  and ips_suc[i]  > 0 else None
        id_= float(ips_disc[i]) if np.isfinite(ips_disc[i]) and ips_disc[i] > 0 else None
        lm_h, lh_h, lp_h = compute_liquid_hgl(
            pig_mp_i, end_mp, P_e_i, v_i,
            sys_mps, sys_elevs, pipe_diam, roughness, density,
            api_g if fluid_num == 3 else None, float(visc),
            ips_active_mp=im, ips_suction_psig=is_, ips_discharge_psig=id_,
        )
        hgl_cache.append((lm_h, lh_h, lp_h, im, is_, id_))
        if (fn + 1) % 100 == 0:
            print(f"  HGL {fn + 1}/{n_frames}")

    # ── Progress window ───────────────────────────────────────────────────────
    prog_win = tk.Toplevel(parent_widget)
    prog_win.title("Generating Animation…")
    prog_win.geometry("420x110")
    prog_win.resizable(False, False)
    tk.Label(prog_win, text="Rendering animation frames…", font=("", 10)).pack(pady=(10, 4))
    pbar     = ttk.Progressbar(prog_win, length=380, mode="determinate", maximum=100)
    pbar.pack(pady=2)
    pbar_lbl = tk.Label(prog_win, text="", font=("", 9))
    pbar_lbl.pack()
    prog_win.update()

    # ── Open video writer ─────────────────────────────────────────────────────
    try:
        writer = imageio.get_writer(output_path, fps=fps, macro_block_size=None)
    except Exception:
        try:
            writer = imageio.get_writer(output_path, fps=fps)
        except Exception as _we:
            messagebox.showerror("Writer Error",
                                  f"Could not create video writer:\n{_we}",
                                  parent=prog_win)
            prog_win.destroy()
            return None

    inj_elev = float(np.interp(purge_start, sys_mps, sys_elevs))

    with writer:
        for fn, i in enumerate(fidx):
            pig_mp_i = float(pig_mps[i])
            pig_e_i  = float(pig_elevs[i])
            spd_i    = float(speeds[i])
            Pb_i     = float(P_behind[i])
            Pi_i     = float(P_inj[i])
            q_i      = float(inj_rates[i])
            t_i      = float(times_hr[i])
            n2_i     = float(cum_n2[i])
            lm_h, lh_h, lp_h, im, is_, id_ = hgl_cache[fn]

            # N2 gas section: straight line from injection head to pig-face head
            # (uses liquid fppsi for display consistency with the HGL)
            inj_head  = inj_elev + Pi_i * fppsi
            pig_g_hd  = pig_e_i  + Pb_i * fppsi
            n2_line.set_data([purge_start, pig_mp_i], [inj_head, pig_g_hd])

            # Liquid HGL
            if len(lm_h) > 1:
                liq_line.set_data(lm_h, lh_h)
            else:
                liq_line.set_data([], [])

            # IPS pump-head vertical jump
            if im and is_ and id_:
                ips_e = float(np.interp(im, sys_mps, sys_elevs))
                ips_line.set_data([im, im],
                                   [ips_e + is_ * fppsi, ips_e + id_ * fppsi])
            else:
                ips_line.set_data([], [])

            # Pig marker
            pig_dot.set_data([pig_mp_i], [pig_g_hd])
            pig_lbl.set_position((pig_mp_i + 0.3, pig_g_hd + 0.012 * y_span))
            pig_lbl.set_text(f"← PIG  {pig_mp_i:.1f} MP")

            # Phase fills
            n2_rect.set_x(x_lo)
            n2_rect.set_width(pig_mp_i - x_lo)
            liq_rect.set_x(pig_mp_i)
            liq_rect.set_width(x_hi - pig_mp_i)

            # Phase labels
            if pig_mp_i - purge_start > 1.5:
                gc = (purge_start + pig_mp_i) / 2.0
                ge = float(np.interp(gc, sys_mps, sys_elevs))
                n2_lbl_t.set_position((gc, ge + Pb_i * fppsi * 0.45))
                n2_lbl_t.set_visible(True)
            else:
                n2_lbl_t.set_visible(False)

            if end_mp - pig_mp_i > 1.5 and len(lm_h) > 1:
                lc   = (pig_mp_i + end_mp) / 2.0
                lh_c = float(np.interp(lc, lm_h, lh_h))
                le_c = float(np.interp(lc, sys_mps, sys_elevs))
                liq_lbl_t.set_position((lc, le_c + (lh_c - le_c) * 0.4))
                liq_lbl_t.set_visible(True)
            else:
                liq_lbl_t.set_visible(False)

            # Monitor pressure annotations
            for p in mon_pts:
                nm_p = p["name"]
                mp_p = float(p["mp"])
                e_p  = float(np.interp(mp_p, sys_mps, sys_elevs))
                if nm_p in mon_press and i < len(mon_press[nm_p]):
                    P_p = float(mon_press[nm_p][i])
                    t   = mon_texts.get(nm_p)
                    if t:
                        t.set_position((mp_p, e_p + P_p * fppsi + 0.01 * y_span))
                        t.set_text(f"{P_p:.0f}")

            # Status box
            status.set_text(
                f"T = {t_i:.2f} hr    MP = {pig_mp_i:.2f}\n"
                f"Speed = {spd_i:.2f} mph    Q = {q_i:,.0f} SCFM\n"
                f"P_inj = {Pi_i:.0f} psig    P_behind = {Pb_i:.0f} psig\n"
                f"N2 cumulative = {n2_i:,.0f} SCF"
            )

            canvas.draw()
            buf    = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            w_, h_ = canvas.get_width_height()
            writer.append_data(buf.reshape(h_, w_, 3))

            pct = int((fn + 1) / n_frames * 100)
            pbar["value"] = pct
            pbar_lbl.config(text=f"Frame {fn + 1}/{n_frames}  —  MP {pig_mp_i:.1f}")
            prog_win.update()

    prog_win.destroy()
    messagebox.showinfo("Animation Complete",
                         f"Animation saved to:\n{output_path}",
                         parent=parent_widget)
    print(f"[Animation] Saved: {output_path}")
    return output_path


def visualize_results(results, inputs):
    logging.info("Entering visualize_results")
    booster_stations = results.get('booster_stations', []) or []
    n2_inventory = results.get('n2_inventory_scf', None)
    has_inventory = (n2_inventory is not None) and len(booster_stations) > 0

    if has_inventory:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18))
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))

    rel_mp = results['purge_mileposts'] - inputs['purge_start_mp']

    ax1.plot(rel_mp, results['elevations'], 'b-')
    ax1.set_title('Elevation vs. Milepost')
    ax1.set_xlabel('Miles')
    ax1.set_ylabel('Elevation (ft)')

    ax2.plot(rel_mp, results['pig_speeds'], 'r-')
    ax2.axhline(y=inputs['max_pig_speed'] * 1.25, linestyle='--', label='Max Speed × 1.25')
    ax2.axhline(y=inputs['target_pig_speed'], linestyle=':', label='Target Speed')
    try:
        _cap = inputs.get('max_pig_speed_from_max_outlet_flow_mph', None)
        if _cap is not None and _cap != "N/A":
            _cap = float(_cap)
            ax2.axhline(y=_cap, linestyle='-.', label='Max Flow Speed Cap')
    except Exception:
        pass
    ax2.set_title('Pig Speed vs. Milepost')
    ax2.set_xlabel('Miles')
    ax2.set_ylabel('Speed (mph)')
    ax2.legend()

    ax3.plot(rel_mp, results['injection_pressures'], label='Injection Pressure (inlet)')
    behind = results.get('behind_pig_pressures', results.get('drive_pressures'))
    ax3.plot(rel_mp, behind, label='Pressure Behind Pig')
    ax3.plot(rel_mp, results['exit_pressures'], label='Exit Pressure (target/taper)')
    ax3.plot(rel_mp, results['gas_dp_losses'], label='Gas DP inlet→pig')
    ax3.set_title('Pressure vs. Milepost')
    ax3.set_xlabel('Miles')
    ax3.set_ylabel('Pressure (psi)')
    ax3.legend()

    ax4.plot(rel_mp, results['cumulative_n2'], 'c-')
    ax4.set_title('Cumulative Nitrogen vs. Milepost')
    ax4.set_xlabel('Miles')
    ax4.set_ylabel('N2 Volume (SCF)')

    if has_inventory:
        try:
            inv_arr = np.asarray(n2_inventory, dtype=float)
            ax5.plot(rel_mp, inv_arr, 'm-', label='N2 Inventory in Gas Column (SCF)')
            # Mark booster locations
            for b in booster_stations:
                bmp = float(b.get('milepost', 0.0)) - inputs['purge_start_mp']
                ax5.axvline(x=bmp, linestyle='--', color='orange', alpha=0.7,
                            label=f"Booster @ MP {b.get('milepost')}")
            ax5.set_title('N2 Inventory in Gas Column vs. Milepost')
            ax5.set_xlabel('Miles')
            ax5.set_ylabel('N2 Inventory (SCF)')
            ax5.legend()
        except Exception:
            pass

    plt.tight_layout()
    plt.show()
    logging.info("Exiting visualize_results")


def print_condensed_table(results, inputs):
    logging.info("Printing condensed table")
    nps_key = resolve_nps_key(inputs['nps'])
    pipe_od = nps_data[nps_key]["OD_in"]
    pipe_id = pipe_od - 2 * float(inputs['pipe_wt'])
    pipe_diameter_ft = pipe_id / 12.0
    area = math.pi * (pipe_diameter_ft / 2) ** 2
    print("\nCondensed Table (50 evenly spaced points):")
    print(
        "Miles\tElevation (ft)\tElapsed Time (hr)\tInjection P (psi)\tGas DP (psi)\tBehind Pig P (psi)\t"
        "Liquid Fric (psi)\tHead (psi)\tExit P (psi)\tInjection Rate (SCFM)\tCumulative N2 (SCF)\t"
        "Pig Speed (mph)\tMiles to Outlet\tBarrels/hr"
    )

    n_points = len(results['purge_mileposts'])
    indices = np.linspace(0, results['last_valid_i'], min(50, n_points), dtype=int)

    for i in indices:
        miles = results['purge_mileposts'][i] - inputs['purge_start_mp']
        elevation = results['elevations'][i]
        time_hr = results['elapsed_times'][i]
        behind = results.get('behind_pig_pressures', results.get('drive_pressures'))
        inj_p = results.get('injection_pressures', np.zeros_like(behind))[i]
        gas_dp = results.get('gas_dp_losses', np.zeros_like(behind))[i]
        behind_p = behind[i]
        friction = results['friction_losses'][i]
        head = results['head_losses'][i]
        exit_p = results['exit_pressures'][i]
        inj_rate = results['injection_rates'][i]
        n2_vol = results['cumulative_n2'][i]
        speed = results['pig_speeds'][i]
        miles_to_outlet = inputs['system_end_mp'] - results['purge_mileposts'][i]
        bbl_hr = speed * 5280.0 * area / 5.614583
        print(
            f"{miles:.2f}\t{elevation:.2f}\t{time_hr:.2f}\t\t"
            f"{inj_p:.0f}\t\t{gas_dp:.1f}\t\t{behind_p:.0f}\t\t"
            f"{friction:.2f}\t\t{head:.0f}\t\t{exit_p:.0f}\t\t"
            f"{int(inj_rate):,}\t\t{int(n2_vol):,}\t\t{speed:.2f}\t\t{miles_to_outlet:.2f}\t\t{bbl_hr:.0f}"
        )


def print_alarm_summary(results, inputs):
    """Print a concise alarm summary after a run."""
    alarms = results.get('alarms', []) or []
    mv = results.get('monitor_violations', []) or []
    alarm_events = results.get('alarm_events', []) or []

    print("\nALARM SUMMARY:")
    if not alarms:
        print("  No alarms tripped.")
        return

    def _worst(rule, kind='min'):
        evs = [v for v in mv if v.get('rule') == rule]
        if not evs:
            return None
        if kind == 'max':
            return max(evs, key=lambda x: float(x.get('margin_psig', float('-inf'))))
        return min(evs, key=lambda x: float(x.get('margin_psig', float('inf'))))

    for a in alarms:
        if a == 'STALL':
            e = next((x for x in alarm_events if x.get('alarm') == 'STALL'), None)
            if e:
                print(f"  STALL: stopped early at MP {e.get('pig_mp'):.3f} (t={e.get('time_hr'):.2f} hr).")
            else:
                print("  STALL: stopped early (near-zero velocity).")

        elif a == 'BELOW_MIN_SPEED':
            e = next((x for x in alarm_events if x.get('alarm') == 'BELOW_MIN_SPEED'), None)
            if e:
                print(f"  MIN SPEED: {e.get('value'):.3g} mph < {e.get('limit'):.3g} mph at MP {e.get('pig_mp'):.3f} (t={e.get('time_hr'):.2f} hr).")
            else:
                print("  MIN SPEED: fell below min_pig_speed threshold at least once.")

        elif a == 'SLACK':
            w = _worst('SLACK', kind='min')
            if w:
                print(f"  SLACK: {w.get('pressure_psig'):.2f} psig < {w.get('limit_psig'):.2f} psig at {w.get('point_name')} (MP {w.get('point_mp'):.3f}) "
                      f"t={w.get('time_hr'):.2f} hr, margin {w.get('margin_psig'):.2f} psi.")
            else:
                print("  SLACK: slack threshold violated at least once.")

        elif a == 'MAOP_EXCEED':
            w = _worst('MAX_PRESSURE', kind='max')
            if w:
                print(f"  MAOP: {w.get('pressure_psig'):.2f} psig > {w.get('limit_psig'):.2f} psig at {w.get('point_name')} (MP {w.get('point_mp'):.3f}) "
                      f"t={w.get('time_hr'):.2f} hr, exceed {w.get('margin_psig'):.2f} psi.")
            else:
                print("  MAOP: MAOP exceeded at least once.")

        elif a == 'MIN_PRESSURE':
            w = _worst('MIN_PRESSURE', kind='min')
            if w:
                print(f"  MIN PRESSURE: {w.get('pressure_psig'):.2f} psig < {w.get('limit_psig'):.2f} psig at {w.get('point_name')} (MP {w.get('point_mp'):.3f}) "
                      f"t={w.get('time_hr'):.2f} hr, margin {w.get('margin_psig'):.2f} psi.")
            else:
                print("  MIN PRESSURE: minimum pressure constraint violated at least once.")


def export_results(dialog_root, inputs, results):
    logging.info("Entering export_results")
    try:
        nps_key = resolve_nps_key(inputs['nps'])
    except Exception:
        nps_key = str(inputs['nps'])

    inputs_data = {
        "Parameter": [
            "Nominal Pipe Size (NPS)",
            "Outside Diameter (in)",
            "Wall Thickness (in)",
            "Pipe Material",
            "Fluid Type",
            "API Gravity (if Crude)",
            "Viscosity (cSt) (if Crude)",
            "Max Nitrogen Drive Pressure (psig)",
            "Max N2 Rate (SCFM)",
            "Min Exit Pressure (Run) (psig)",
            "Exit Pressure (End) (psig)",
            "Exit Pressure Behavior",
            "Endpoint Constraint Mode",
            "Exit Pressure Min Clamp (psig)",
            "Exit Pressure Max Clamp (psig)",
            "Exit Pressure Ramp Limit (psi/hr)",
            "Slack Threshold (psig)",
            "MAOP (psig)",
            "Max Outlet Flow to Tankage (BPH)",
            "Equivalent Max Pig Speed from Max Flow (mph)",
            "Max Pig Speed (mph)",
            "Min Pig Speed (mph)",
            "Target Pig Speed (mph)",
            "Purge Start Milepost",
            "Purge End Milepost",
            "System Endpoint Milepost",
            "Throttle Down Distance (miles)",
            "Estimated N2 Pressure at End (psig)",
            "Has Intermediate Pump Stations",
            "IPS Mileposts",
            "IPS Shutdown Distance (miles)",
            "Minimum Pump Suction Pressure (psig)",
            "Minimum Pump Flow (BPH)",
            "IPS Check Valve Seal ΔP (psi)",
            "Resolution (points)",
            "Resample purge profile to resolution"
        ],
        "Value": [
            inputs.get('nps', "N/A"),
            nps_data[nps_key]["OD_in"],
            inputs.get('pipe_wt', "N/A"),
            roughness_data[int(inputs['roughness_num'])]['material'] if inputs.get('roughness_num', None) is not None else "N/A",
            fluid_data[int(inputs['fluid_num'])]['name'] if inputs.get('fluid_num', None) is not None else "N/A",
            inputs.get('api_gravity', "N/A"),
            inputs.get('viscosity_cst', "N/A"),
            inputs.get('max_nitrogen_pressure', inputs.get('max_drive_pressure', "N/A")),
            inputs.get('max_n2_rate_scfm', "N/A"),
            inputs.get('exit_pressure_run', "N/A"),
            inputs.get('exit_pressure_end', "N/A"),
            inputs.get('exit_pressure_behavior', inputs.get('exit_behavior', "N/A")),
            inputs.get('endpoint_constraint_mode', "none"),
            inputs.get('exit_pressure_min_clamp', "N/A"),
            inputs.get('exit_pressure_max_clamp', "N/A"),
            inputs.get('exit_pressure_ramp_psi_per_hr', "N/A"),
            inputs.get('slack_pressure_psig', 50.0),
            inputs.get('maop_psig', "N/A"),
            inputs.get('max_outlet_flow_bph', "N/A"),
            inputs.get('max_pig_speed_from_max_outlet_flow_mph', "N/A"),
            inputs.get('max_pig_speed', "N/A"),
            inputs.get('min_pig_speed', "N/A"),
            inputs.get('target_pig_speed', "N/A"),
            inputs.get('purge_start_mp', "N/A"),
            inputs.get('purge_end_mp', "N/A"),
            inputs.get('system_end_mp', "N/A"),
            inputs.get('throttle_down_miles', "N/A"),
            inputs.get('n2_end_pressure', "N/A"),
            inputs.get('has_ips', False),
            inputs.get('ips_mp', inputs.get('ips_mps', "N/A")) or "N/A",
            inputs.get('ips_shutdown_dist', "N/A"),
            inputs.get('min_pump_suction_pressure', "N/A"),
            inputs.get('min_pump_flow_bph', "N/A"),
            inputs.get('ips_check_dp_psi', "N/A"),
            inputs.get('resolution', "N/A"),
            inputs.get('resample_profile_to_resolution', True),
        ]
    }
    inputs_df = pd.DataFrame(inputs_data)

    nps_key2 = resolve_nps_key(inputs['nps'])
    pipe_od2 = nps_data[nps_key2]["OD_in"]
    pipe_id2 = pipe_od2 - 2 * float(inputs['pipe_wt'])
    pipe_diameter_ft2 = pipe_id2 / 12.0
    area2 = math.pi * (pipe_diameter_ft2 / 2) ** 2
    miles_to_outlet = inputs['system_end_mp'] - results['purge_mileposts']
    bbl_hr = results['pig_speeds'] * 5280.0 * area2 / 5.614583

    full_results_df = pd.DataFrame({
        "Miles": results['purge_mileposts'] - inputs['purge_start_mp'],
        "Elevation (ft)": results['elevations'],
        "Elapsed Time (hours)": results['elapsed_times'],
        "Injection Pressure (psi)": results.get('injection_pressures', results.get('behind_pig_pressures', results['drive_pressures'])),
        "Gas DP Inlet→Pig (psi)": results.get('gas_dp_losses', np.zeros_like(results.get('behind_pig_pressures', results['drive_pressures']))),
        "Pressure Behind Pig (psi)": results.get('behind_pig_pressures', results['drive_pressures']),
        "Friction Loss (psi)": results['friction_losses'],
        "Head Pressure (psi)": results['head_losses'],
        "Exit Pressure (psi)": results['exit_pressures'],
        "Injection Rate (SCFM)": results['injection_rates'],
        "Cumulative N2 (SCF)": results['cumulative_n2'],
        "Pig Speed (mph)": results['pig_speeds'],
        "Miles to Outlet": miles_to_outlet,
        "Barrels per hour": bbl_hr
    })

    # N2 inventory column (v29: present when booster stations active)
    try:
        n2_inv = results.get('n2_inventory_scf', None)
        if n2_inv is not None:
            full_results_df["N2 Inventory (SCF)"] = np.asarray(n2_inv, dtype=float)
    except Exception:
        pass

    # Derived nitrogen mass
    try:
        cum_scf = np.asarray(results['cumulative_n2'], dtype=float)
        full_results_df["Cumulative N2 (lbm)"] = cum_scf * float(N2_LBM_PER_SCF_STD)
        full_results_df["Cumulative N2 (kg)"] = cum_scf * float(N2_KG_PER_SCF_STD)
        scfm = np.asarray(results.get('injection_rates', np.zeros_like(cum_scf)), dtype=float)
        full_results_df["N2 Mass Flow (lbm/hr)"] = scfm * 60.0 * float(N2_LBM_PER_SCF_STD)
        full_results_df["N2 Mass Flow (kg/hr)"] = scfm * 60.0 * float(N2_KG_PER_SCF_STD)
    except Exception:
        pass

    if results.get('safe_max_drive_psig', None) is not None:
        try:
            full_results_df["Safe Max Behind Pig (psig)"] = np.asarray(results['safe_max_drive_psig'], dtype=float)
        except Exception:
            pass

    n_points = len(results['purge_mileposts'])
    condensed_indices = np.linspace(0, results['last_valid_i'], min(50, n_points), dtype=int)
    condensed_results_df = full_results_df.iloc[condensed_indices]

    try:
        dialog_root.update()
        start_time = time.time()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            parent=dialog_root
        )
        dialog_root.update()
        if time.time() - start_time > 45:
            logging.warning("Export file dialog timeout after 45 seconds")
            raise ValueError("Export file dialog timed out.")
        logging.info(f"Selected output file: {file_path}")
        if file_path:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                inputs_df.to_excel(writer, sheet_name="Simulation Inputs", index=False)
                full_results_df.to_excel(writer, sheet_name="Full Results", index=False)
                condensed_results_df.to_excel(writer, sheet_name="Condensed Results", index=False)

                # --- Booster Stations config sheet (v29) ---
                booster_stations_cfg = results.get('booster_stations', []) or []
                if booster_stations_cfg:
                    try:
                        bst_rows = []
                        for b in booster_stations_cfg:
                            bst_rows.append({
                                'Milepost': b.get('milepost'),
                                'Max Discharge Pressure (psig)': b.get('max_discharge_psig'),
                                'Min Suction Pressure (psig)': b.get('min_suction_psig'),
                                'Name': b.get('name', ''),
                            })
                        pd.DataFrame(bst_rows).to_excel(writer, sheet_name="Booster Stations", index=False)
                    except Exception:
                        pass

                    # --- Booster Dynamic: per-step suction/discharge per booster ---
                    try:
                        bst_diag = results.get('booster_diagnostics', []) or []
                        if bst_diag and len(bst_diag) == n_points:
                            bst_dyn_base = pd.DataFrame({
                                'Elapsed Time (hr)': results['elapsed_times'],
                                'Pig Milepost': results['purge_mileposts'],
                                'Miles into purge': results['purge_mileposts'] - inputs['purge_start_mp'],
                            })
                            for b in booster_stations_cfg:
                                bmp = float(b.get('milepost', 0.0))
                                bname = b.get('name', f"Booster@{bmp:.1f}")
                                suction_col = f"{bname} Suction (psig)"
                                discharge_col = f"{bname} Discharge (psig)"
                                suction_vals = np.full(n_points, np.nan)
                                discharge_vals = np.full(n_points, np.nan)
                                for idx, diag in enumerate(bst_diag):
                                    bs = diag.get('booster_states', []) or []
                                    for bstate in bs:
                                        if abs(float(bstate.get('milepost', -1e9)) - bmp) < 0.01:
                                            suction_vals[idx] = bstate.get('suction_psig', np.nan)
                                            discharge_vals[idx] = bstate.get('discharge_psig', np.nan)
                                bst_dyn_base[suction_col] = suction_vals
                                bst_dyn_base[discharge_col] = discharge_vals
                            bst_dyn_base.to_excel(writer, sheet_name="Booster Dynamic", index=False)
                    except Exception:
                        pass

                # --- Monitor sheets ---
                monitor_points = results.get('monitor_points', []) or []
                monitor_pressures = results.get('monitor_pressures', {}) or {}
                monitor_pressures_static = results.get('monitor_pressures_static', {}) or {}
                monitor_pressures_static_maxdrive = results.get('monitor_pressures_static_maxdrive', {}) or {}
                static_violations_maxdrive = results.get('static_violations_maxdrive', []) or []
                static_violations = results.get('static_violations', []) or []
                monitor_violations = results.get('monitor_violations', []) or []

                if monitor_points and monitor_pressures:
                    ordered = sorted(monitor_points, key=lambda p: float(p.get('mp', 0.0)))

                    monitor_points_df = pd.DataFrame([{
                        'Name': p.get('name'),
                        'Scope': p.get('scope', ''),
                        'Category': p.get('category'),
                        'Milepost': p.get('mp'),
                        'Elevation (ft)': p.get('elev'),
                        'Min Rule (psig)': p.get('min_psig'),
                        'Max Rule (psig)': p.get('max_psig'),
                    } for p in ordered])
                    monitor_points_df.to_excel(writer, sheet_name="Monitors", index=False)

                    n_steps = len(results['purge_mileposts'])
                    base = pd.DataFrame({
                        'Elapsed Time (hr)': results['elapsed_times'],
                        'Pig Milepost': results['purge_mileposts'],
                        'Miles into purge': results['purge_mileposts'] - inputs['purge_start_mp'],
                        'Exit Pressure (psig)': results['exit_pressures'],
                        'Hydraulic Boundary MP': results.get('hydraulic_boundary_mp', np.full(n_steps, np.nan, dtype=float)),
                        'Miles to Hydraulic Boundary': results.get('hydraulic_boundary_mp', np.full(n_steps, np.nan, dtype=float)) - results['purge_mileposts'],
                        'Hydraulic Boundary Pressure (psig)': results.get('hydraulic_boundary_pressure', np.full(n_steps, np.nan, dtype=float)),
                        'Head to Boundary (psi)': results.get('head_to_boundary_psi', np.full(n_steps, np.nan, dtype=float)),
                        'IPS State': results.get('ips_state', np.array([''] * n_steps, dtype=object)),
                        'IPS Active MP': results.get('ips_active_mp', np.full(n_steps, np.nan, dtype=float)),
                        'IPS Suction (psig)': results.get('ips_suction_psig', np.full(n_steps, np.nan, dtype=float)),
                        'IPS Discharge Req (psig)': results.get('ips_discharge_req_psig', np.full(n_steps, np.nan, dtype=float)),
                        'IPS Pump ΔP Req (psi)': results.get('ips_pump_dp_req_psi', np.full(n_steps, np.nan, dtype=float)),
                        'Mainline Flow (BPH)': results.get('ips_mainline_flow_bph', np.full(n_steps, np.nan, dtype=float)),
                    })

                    dyn = base.copy()
                    for p in ordered:
                        name = p.get('name')
                        if name in monitor_pressures:
                            dyn[name] = monitor_pressures[name]
                    dyn.to_excel(writer, sheet_name="Monitor Dynamic", index=False)

                    if monitor_pressures_static:
                        stat = base.copy()
                        for p in ordered:
                            name = p.get('name')
                            if name in monitor_pressures_static:
                                stat[name] = monitor_pressures_static[name]
                        stat.to_excel(writer, sheet_name="Monitor Static", index=False)

                    if monitor_pressures_static_maxdrive:
                        stat_md = base.copy()
                        for p in ordered:
                            name = p.get('name')
                            if name in monitor_pressures_static_maxdrive:
                                stat_md[name] = monitor_pressures_static_maxdrive[name]
                        stat_md.to_excel(writer, sheet_name="Monitor Static MaxDrive", index=False)

                    if monitor_violations:
                        pd.DataFrame(monitor_violations).to_excel(writer, sheet_name="Violations Dynamic", index=False)
                    if static_violations:
                        pd.DataFrame(static_violations).to_excel(writer, sheet_name="Violations Static", index=False)
                    if static_violations_maxdrive:
                        pd.DataFrame(static_violations_maxdrive).to_excel(writer, sheet_name="Violations Static MaxDrive", index=False)

                    alarm_events = results.get('alarm_events', []) or []
                    if alarm_events:
                        pd.DataFrame(alarm_events).to_excel(writer, sheet_name="Alarms", index=False)

                    try:
                        pig_mps = np.asarray(results['purge_mileposts'], dtype=float)
                        times = np.asarray(results['elapsed_times'], dtype=float)
                        summary_rows = []
                        maop = inputs.get('maop_psig', None)
                        maop = float(maop) if maop is not None else None

                        for p in ordered:
                            name = p.get('name')
                            mp_p = float(p.get('mp'))
                            P_dyn = np.asarray(monitor_pressures.get(name, np.full_like(pig_mps, np.nan)), dtype=float)
                            P_sta = np.asarray(monitor_pressures_static.get(name, np.full_like(pig_mps, np.nan)), dtype=float)
                            P_sta_md = np.asarray(monitor_pressures_static_maxdrive.get(name, np.full_like(pig_mps, np.nan)), dtype=float)

                            if P_dyn.size != pig_mps.size:
                                continue

                            max_dyn = float(np.nanmax(P_dyn)) if np.any(np.isfinite(P_dyn)) else np.nan
                            max_sta = float(np.nanmax(P_sta)) if np.any(np.isfinite(P_sta)) else np.nan
                            max_sta_md = float(np.nanmax(P_sta_md)) if np.any(np.isfinite(P_sta_md)) else np.nan

                            mask_liq = mp_p >= pig_mps - 1e-9

                            min_dyn_liq = float(np.nanmin(P_dyn[mask_liq])) if np.any(mask_liq) and np.any(np.isfinite(P_dyn[mask_liq])) else np.nan
                            max_dyn_liq = float(np.nanmax(P_dyn[mask_liq])) if np.any(mask_liq) and np.any(np.isfinite(P_dyn[mask_liq])) else np.nan
                            max_sta_liq = float(np.nanmax(P_sta[mask_liq])) if np.any(mask_liq) and np.any(np.isfinite(P_sta[mask_liq])) else np.nan
                            max_sta_liq_md = float(np.nanmax(P_sta_md[mask_liq])) if np.any(mask_liq) and np.any(np.isfinite(P_sta_md[mask_liq])) else np.nan

                            min_lim = p.get('min_psig')
                            max_lim = p.get('max_psig')

                            worst_min_margin = np.nan
                            worst_min_time = np.nan
                            worst_max_margin = np.nan
                            worst_max_time = np.nan

                            if min_lim is not None and np.any(mask_liq):
                                margins = P_dyn[mask_liq] - float(min_lim)
                                if np.any(np.isfinite(margins)):
                                    j = int(np.nanargmin(margins))
                                    worst_min_margin = float(margins[j])
                                    worst_min_time = float(times[mask_liq][j]) if times.size == pig_mps.size else np.nan

                            if max_lim is not None:
                                margins = float(max_lim) - P_dyn
                                if np.any(np.isfinite(margins)):
                                    j = int(np.nanargmin(margins))
                                    worst_max_margin = float(margins[j])
                                    worst_max_time = float(times[j]) if times.size == pig_mps.size else np.nan
                            elif maop is not None:
                                margins = maop - P_dyn
                                if np.any(np.isfinite(margins)):
                                    j = int(np.nanargmin(margins))
                                    worst_max_margin = float(margins[j])
                                    worst_max_time = float(times[j]) if times.size == pig_mps.size else np.nan

                            worst_sta_margin = np.nan
                            worst_sta_time = np.nan
                            if maop is not None and np.any(np.isfinite(P_sta)):
                                margins = maop - P_sta
                                j = int(np.nanargmin(margins))
                                worst_sta_margin = float(margins[j])
                                worst_sta_time = float(times[j]) if times.size == pig_mps.size else np.nan

                            summary_rows.append({
                                'Name': name,
                                'Milepost': mp_p,
                                'Elevation (ft)': float(p.get('elev', np.nan)),
                                'Category': p.get('category'),
                                'Max Dynamic (psig)': max_dyn,
                                'Min Dynamic (liquid) (psig)': min_dyn_liq,
                                'Max Dynamic (liquid) (psig)': max_dyn_liq,
                                'Max Static (block-in) (psig)': max_sta,
                                'Max Static (liquid) (psig)': max_sta_liq,
                                'Max Static (max-drive) (psig)': max_sta_md,
                                'Max Static (max-drive, liquid) (psig)': max_sta_liq_md,
                                'MaxDrive Static Mode': results.get('static_maxdrive_mode', ''),
                                'Min Rule (psig)': min_lim,
                                'Worst MIN margin (dyn) (psig)': worst_min_margin,
                                'Worst MIN time (dyn) (hr)': worst_min_time,
                                'Max Rule (psig)': max_lim,
                                'Worst MAX margin (dyn) (psig)': worst_max_margin,
                                'Worst MAX time (dyn) (hr)': worst_max_time,
                                'MAOP (psig)': maop,
                                'Worst MAOP margin (static) (psig)': worst_sta_margin,
                                'Worst MAOP time (static) (hr)': worst_sta_time,
                            })

                        if summary_rows:
                            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Monitor Summary", index=False)
                    except Exception:
                        pass

                    try:
                        from openpyxl.styles import Font, Alignment
                        from openpyxl.utils import get_column_letter

                        def _fmt(ws):
                            ws.freeze_panes = "A2"
                            ws.auto_filter.ref = ws.dimensions
                            for cell in ws[1]:
                                cell.font = Font(bold=True)
                                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                            for col in range(1, ws.max_column + 1):
                                max_len = 0
                                col_letter = get_column_letter(col)
                                for row in range(1, min(ws.max_row, 3000) + 1):
                                    v = ws.cell(row=row, column=col).value
                                    if v is None:
                                        continue
                                    s = str(v)
                                    if len(s) > max_len:
                                        max_len = len(s)
                                ws.column_dimensions[col_letter].width = max(10, min(48, max_len + 2))

                        for sh in ("Monitors", "Monitor Dynamic", "Monitor Static", "Violations Dynamic",
                                   "Violations Static", "Alarms", "Monitor Summary",
                                   "Booster Stations", "Booster Dynamic"):
                            if sh in writer.sheets:
                                _fmt(writer.sheets[sh])
                    except Exception:
                        pass

            messagebox.showinfo("Success", f"Results exported to {file_path}", parent=dialog_root)
        else:
            logging.warning("No output file selected")
            messagebox.showwarning("File Selection", "No output file selected. Results not saved.", parent=dialog_root)
    except Exception as e:
        logging.error(f"Failed to export results: {str(e)}")
        raise ValueError(f"Error exporting results: {str(e)}")
    logging.info("Exiting export_results")

def configure_profile_segments(dialog_root, file_paths):
    """Let the user review selected profile segments, choose which ones to invert, and set append order.

    Returns:
        List[Dict[str, Any]] like: [{'path': 'C:/...', 'invert': True}, ...]
    """
    file_paths = list(file_paths or [])
    if not file_paths:
        return []

    if len(file_paths) == 1:
        fp = file_paths[0]
        invert = messagebox.askyesno(
            "Invert Profile Segment?",
            "Invert (reverse) this profile segment?\n\n"
            f"{os.path.basename(fp)}",
            parent=dialog_root
        )
        return [{"path": fp, "invert": bool(invert)}]

    top = tk.Toplevel(dialog_root)
    top.title("Profile Segments: Order & Inversion")

    w, h = 980, 420
    try:
        sw = top.winfo_screenwidth()
        sh = top.winfo_screenheight()
        x = max(50, (sw - w) // 2)
        y = max(50, (sh - h) // 3)
        top.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        top.geometry("980x420")
    top.minsize(740, 320)

    try:
        top.update_idletasks()
        top.deiconify()
        top.lift()
        top.attributes('-topmost', True)
        top.focus_force()
        top.after(750, lambda: top.attributes('-topmost', False))
    except Exception:
        pass

    top.grab_set()
    logging.info("Profile segment order/inversion dialog opened; awaiting user action.")

    instructions = (
        "1) Review the segments you selected\n"
        "2) Toggle 'Invert' for any segment that runs backwards\n"
        "3) Use Up/Down to set the order they should be appended\n"
        "4) Click OK to continue"
    )
    ttk.Label(top, text=instructions, justify="left").pack(anchor="w", padx=12, pady=(12, 6))

    frame = ttk.Frame(top)
    frame.pack(fill="both", expand=True, padx=12, pady=6)

    columns = ("invert", "path")
    tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="extended")
    tree.heading("invert", text="Invert?")
    tree.heading("path", text="File")
    tree.column("invert", width=80, anchor="center", stretch=False)
    tree.column("path", width=820, anchor="w", stretch=True)

    yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    for fp in file_paths:
        tree.insert("", "end", values=("No", fp))

    btns = ttk.Frame(top)
    btns.pack(fill="x", padx=12, pady=(6, 12))

    def _selected_sorted():
        items = list(tree.selection())
        items.sort(key=lambda iid: tree.index(iid))
        return items

    def move(delta):
        items = _selected_sorted()
        if not items:
            return
        all_items = list(tree.get_children())
        n = len(all_items)

        if delta < 0:
            for iid in items:
                idx = tree.index(iid)
                new_idx = max(idx + delta, 0)
                tree.move(iid, "", new_idx)
        else:
            for iid in reversed(items):
                idx = tree.index(iid)
                new_idx = min(idx + delta, n - 1)
                tree.move(iid, "", new_idx)

    def toggle_invert():
        items = list(tree.selection())
        if not items:
            return
        for iid in items:
            cur = str(tree.set(iid, "invert")).strip().lower()
            new_val = "Yes" if cur in ("no", "false", "0", "") else "No"
            tree.set(iid, "invert", new_val)

    result = {"segments": None}

    def on_ok():
        segs = []
        for iid in tree.get_children():
            inv = str(tree.set(iid, "invert")).strip().lower() in ("yes", "true", "1", "y")
            fp = str(tree.set(iid, "path"))
            segs.append({"path": fp, "invert": inv})
        result["segments"] = segs
        top.destroy()

    def on_cancel():
        result["segments"] = None
        top.destroy()

    ttk.Button(btns, text="Move Up", command=lambda: move(-1)).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Move Down", command=lambda: move(1)).pack(side="left", padx=(0, 12))
    ttk.Button(btns, text="Toggle Invert", command=toggle_invert).pack(side="left")

    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btns, text="OK", command=on_ok).pack(side="right")

    top.protocol("WM_DELETE_WINDOW", on_cancel)
    top.bind("<Escape>", lambda e: on_cancel())
    top.wait_window()
    logging.info("Profile segment dialog closed.")

    if result["segments"] is None:
        raise ValueError("Profile segment selection cancelled.")
    return result["segments"]


def preview_profile_dialog(parent, mileposts, elevations, segment_bounds=None, elevation_units_label="ft"):
    """Modal preview of the combined elevation profile.

    Returns one of: 'approve', 'reselect', 'cancel'.
    """
    mp = np.asarray(mileposts)
    el = np.asarray(elevations)
    if len(mp) > 30000:
        idx = np.linspace(0, len(mp) - 1, 30000, dtype=int)
        mp_plot = mp[idx]
        el_plot = el[idx]
    else:
        mp_plot = mp
        el_plot = el

    if FigureCanvasTkAgg is None or Figure is None:
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(mp_plot, el_plot)
            plt.xlabel("Milepost (mi)")
            plt.ylabel(f"Elevation ({elevation_units_label})")
            plt.title("Pipeline Profile Preview")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

        resp = messagebox.askyesno(
            "Approve Profile?",
            "Approve this profile and continue?",
            parent=parent
        )
        return "approve" if resp else "reselect"

    top = tk.Toplevel(parent)
    top.title("Pipeline Profile Preview")
    top.geometry("1050x700")
    top.transient(parent)
    top.grab_set()

    hdr = ttk.Frame(top, padding=(12, 10, 12, 6))
    hdr.pack(fill="x")

    try:
        mp0 = float(np.min(mp))
        mp1 = float(np.max(mp))
        el0 = float(np.min(el))
        el1 = float(np.max(el))
        stats = f"MP: {mp0:.2f} → {mp1:.2f} mi   |   Elev: {el0:.0f} → {el1:.0f} {elevation_units_label}   |   Points: {len(mp)}"
    except Exception:
        stats = f"Points: {len(mp)}"

    ttk.Label(hdr, text=stats, font=("Segoe UI", 10, "bold")).pack(anchor="w")
    ttk.Label(
        hdr,
        text="Inspect the combined profile below. If something looks wrong, choose Re-select Segments to change invert/order before continuing.",
        wraplength=1000
    ).pack(anchor="w", pady=(4, 0))

    plot_frame = ttk.Frame(top, padding=(12, 6, 12, 6))
    plot_frame.pack(fill="both", expand=True)

    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(mp_plot, el_plot)
    ax.set_xlabel("Milepost (mi)")
    ax.set_ylabel(f"Elevation ({elevation_units_label})")
    ax.grid(True, alpha=0.3)

    if segment_bounds:
        try:
            y_min = float(np.min(el))
            y_max = float(np.max(el))
            y_span = (y_max - y_min) if (y_max > y_min) else 1.0
            y_label = y_max + 0.02 * y_span
        except Exception:
            y_label = None

        for b in segment_bounds:
            try:
                x0 = float(b["start_mp"])
                x1 = float(b["end_mp"])
                name = str(b.get("name", "segment"))
                inv = " (inv)" if b.get("invert") else ""
                ax.axvline(x0, linestyle="--", alpha=0.22)
                ax.axvline(x1, linestyle="--", alpha=0.22)
                if y_label is not None:
                    ax.text((x0 + x1) / 2.0, y_label, name + inv, ha="center", va="bottom", fontsize=8, alpha=0.85)
            except Exception:
                continue

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    try:
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
    except Exception:
        toolbar = None

    btns = ttk.Frame(top, padding=(12, 8, 12, 12))
    btns.pack(fill="x")

    result = {"action": "cancel"}

    def on_approve():
        result["action"] = "approve"
        top.destroy()

    def on_reselect():
        result["action"] = "reselect"
        top.destroy()

    def on_cancel():
        result["action"] = "cancel"
        top.destroy()

    ttk.Button(btns, text="Re-select Segments", command=on_reselect).pack(side="left")
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(btns, text="Approve Profile", command=on_approve).pack(side="right", padx=(0, 8))

    top.protocol("WM_DELETE_WINDOW", on_cancel)
    top.wait_window()
    return result["action"]


def process_elevation_profile(dialog_root, initial_inputs):
    logging.info("Processing elevation profile")
    elevation_units = initial_inputs['elevation_units']

    try:
        dialog_root.update()
        start_time = time.time()
        files = filedialog.askopenfilenames(
            title="Select Elevation Profile Files (Multiple OK)",
            filetypes=[
                ("All Supported", ("*.kmz", "*.kml", "*.xlsx", "*.xls", "*.txt", "*.csv")),
                ("KMZ/KML", ("*.kmz", "*.kml")),
                ("Excel", ("*.xlsx", "*.xls")),
                ("TXT/CSV", ("*.txt", "*.csv")),
            ],
            parent=dialog_root
        )
        dialog_root.update()
        logging.info(f"Selected files: {files}")
    except Exception as e:
        logging.error(f"Failed to open file dialog: {str(e)}")
        messagebox.showerror("File Dialog Error", f"Failed to open file dialog: {str(e)}. Try copying files to C:\\Users\\YourName\\Documents.", parent=dialog_root)
        raise ValueError(f"Failed to open file dialog: {str(e)}")

    if not files:
        logging.warning("No elevation profile files selected")
        messagebox.showerror("File Selection Error", "No elevation profile files selected.", parent=dialog_root)
        raise ValueError("No elevation profile files selected.")

    segments = configure_profile_segments(dialog_root, files)

    mileposts = []
    elevations = []
    current_mp = 0.0
    segment_bounds = []

    for seg in segments:
        file_path = seg["path"]
        invert_segment = bool(seg.get("invert", False))
        seg_start_mp = current_mp
        logging.info(f"Processing file: {file_path} (invert={invert_segment})")

        if file_path.lower().endswith(('.kmz', '.kml')):
            try:
                parser = _get_secure_lxml_parser()
                if file_path.lower().endswith('.kmz'):
                    with zipfile.ZipFile(file_path, 'r') as z:
                        kml_files = [f for f in z.namelist() if f.lower().endswith('.kml')]
                        if not kml_files:
                            raise ValueError(f"No KML found in KMZ: {file_path}")
                        kml_choice = None
                        for f in kml_files:
                            if os.path.basename(f).lower() == 'doc.kml':
                                kml_choice = f
                                break
                        if kml_choice is None:
                            kml_choice = sorted(kml_files)[0]
                        with z.open(kml_choice) as kml_file:
                            tree = ET.parse(kml_file, parser=parser)
                else:
                    tree = ET.parse(file_path, parser=parser)

                root = tree.getroot()

                coords = extract_pipeline_coords_from_kml_root(
                    root, file_path=file_path, multiline_mode='longest', max_points=200000
                )
                if not coords:
                    raise ValueError(f"No valid coordinates found in {file_path}")

                if invert_segment:
                    coords = list(reversed(coords))

                file_mileposts = [current_mp]
                file_elevations = [coords[0][2]]
                for j in range(1, len(coords)):
                    try:
                        dist_miles = geodesic(coords[j-1][:2], coords[j][:2]).miles
                    except Exception as e:
                        logging.warning(f"Geodesic failed at point {j}: {str(e)}, using Haversine")
                        dist_miles = haversine_distance(coords[j-1][0], coords[j-1][1], coords[j][0], coords[j][1])
                    current_mp += dist_miles
                    file_mileposts.append(current_mp)
                    file_elevations.append(coords[j][2])

                mileposts.extend(file_mileposts)
                elevations.extend(file_elevations)
                segment_bounds.append({"name": os.path.basename(file_path), "start_mp": seg_start_mp, "end_mp": current_mp, "invert": invert_segment})
            except Exception as e:
                logging.error(f"Failed to parse KMZ/KML {file_path}: {str(e)}")
                raise ValueError(f"Failed to parse KMZ/KML {file_path}: {str(e)}")

        elif file_path.lower().endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file_path)
                if 'Milepost' not in df.columns or 'Elevation' not in df.columns:
                    raise ValueError(f"Excel {file_path} must have 'Milepost' and 'Elevation' columns.")

                mp = pd.to_numeric(df['Milepost'], errors='coerce').to_numpy()
                el = pd.to_numeric(df['Elevation'], errors='coerce').to_numpy()
                valid = (~np.isnan(mp)) & (~np.isnan(el))
                mp = mp[valid]
                el = el[valid]
                if len(mp) < 2:
                    raise ValueError(f"Excel {file_path} must contain at least 2 valid points.")

                order = np.argsort(mp)
                mp = mp[order]
                el = el[order]
                mp = mp - mp[0]

                if invert_segment:
                    mp_end = mp[-1]
                    mp_rev = mp[::-1]
                    el = el[::-1]
                    mp = mp_end - mp_rev

                file_mileposts = (mp + current_mp).tolist()
                file_elevations = el.tolist()
                current_mp = file_mileposts[-1]
                mileposts.extend(file_mileposts)
                elevations.extend(file_elevations)
                segment_bounds.append({"name": os.path.basename(file_path), "start_mp": seg_start_mp, "end_mp": current_mp, "invert": invert_segment})
            except Exception as e:
                logging.error(f"Failed to parse Excel {file_path}: {str(e)}")
                raise ValueError(f"Failed to parse Excel {file_path}: {str(e)}")

        elif file_path.lower().endswith(('.txt', '.csv')):
            # v29: flexible column detection for ILI/survey exports with various naming conventions
            try:
                parsed = _parse_flexible_profile(file_path)
                if parsed is None:
                    raise ValueError(f"Could not detect lat/lon/elevation columns in {file_path}. "
                                     "Expected columns like 'latitude', 'longitude', 'altitude (ft)', 'elevation', "
                                     "'milepost', 'OD', 'WT', etc.")
                mode = parsed.get('mode', 'latlon')

                if mode == 'milepost':
                    # Milepost+elevation mode: rebase and append
                    mp_arr = np.asarray(parsed['milepost'], dtype=float)
                    el_arr = np.asarray(parsed['elevation_ft'], dtype=float)
                    mp_arr = mp_arr - mp_arr[0]

                    if invert_segment:
                        mp_end = mp_arr[-1]
                        el_arr = el_arr[::-1]
                        mp_arr = mp_end - mp_arr[::-1]

                    file_mileposts = (mp_arr + current_mp).tolist()
                    file_elevations = el_arr.tolist()
                    current_mp = file_mileposts[-1]
                    mileposts.extend(file_mileposts)
                    elevations.extend(file_elevations)
                else:
                    # lat/lon mode: compute incremental distances
                    lats = parsed['lat']
                    lons = parsed['lon']
                    elev_ft = parsed['elevation_ft']
                    coords = list(zip(lats, lons, elev_ft))

                    if invert_segment:
                        coords = list(reversed(coords))

                    file_mileposts = [current_mp]
                    file_elevations = [coords[0][2]]
                    for j in range(1, len(coords)):
                        try:
                            dist_miles = geodesic((coords[j-1][0], coords[j-1][1]),
                                                  (coords[j][0], coords[j][1])).miles
                        except Exception as ge:
                            logging.warning(f"Geodesic failed at point {j}: {ge}, using Haversine")
                            dist_miles = haversine_distance(coords[j-1][0], coords[j-1][1],
                                                            coords[j][0], coords[j][1])
                        current_mp += dist_miles
                        file_mileposts.append(current_mp)
                        file_elevations.append(coords[j][2])

                    mileposts.extend(file_mileposts)
                    elevations.extend(file_elevations)

                segment_bounds.append({"name": os.path.basename(file_path), "start_mp": seg_start_mp, "end_mp": current_mp, "invert": invert_segment})
            except Exception as e:
                logging.error(f"Failed to parse TXT/CSV {file_path}: {str(e)}")
                raise ValueError(f"Failed to parse TXT/CSV {file_path}: {str(e)}")

        else:
            # Fallback: try flexible parser for any other text-like file
            try:
                parsed = _parse_flexible_profile(file_path)
                if parsed is None:
                    raise ValueError(f"Unsupported file type or unrecognized column layout: {file_path}")
                mode = parsed.get('mode', 'latlon')

                if mode == 'milepost':
                    mp_arr = np.asarray(parsed['milepost'], dtype=float)
                    el_arr = np.asarray(parsed['elevation_ft'], dtype=float)
                    mp_arr = mp_arr - mp_arr[0]
                    if invert_segment:
                        mp_end = mp_arr[-1]
                        el_arr = el_arr[::-1]
                        mp_arr = mp_end - mp_arr[::-1]
                    file_mileposts = (mp_arr + current_mp).tolist()
                    file_elevations = el_arr.tolist()
                    current_mp = file_mileposts[-1]
                    mileposts.extend(file_mileposts)
                    elevations.extend(file_elevations)
                else:
                    lats = parsed['lat']
                    lons = parsed['lon']
                    elev_ft = parsed['elevation_ft']
                    coords = list(zip(lats, lons, elev_ft))
                    if invert_segment:
                        coords = list(reversed(coords))
                    file_mileposts = [current_mp]
                    file_elevations = [coords[0][2]]
                    for j in range(1, len(coords)):
                        try:
                            dist_miles = geodesic((coords[j-1][0], coords[j-1][1]),
                                                  (coords[j][0], coords[j][1])).miles
                        except Exception as ge:
                            logging.warning(f"Geodesic failed at point {j}: {ge}, using Haversine")
                            dist_miles = haversine_distance(coords[j-1][0], coords[j-1][1],
                                                            coords[j][0], coords[j][1])
                        current_mp += dist_miles
                        file_mileposts.append(current_mp)
                        file_elevations.append(coords[j][2])
                    mileposts.extend(file_mileposts)
                    elevations.extend(file_elevations)

                segment_bounds.append({"name": os.path.basename(file_path), "start_mp": seg_start_mp, "end_mp": current_mp, "invert": invert_segment})
            except Exception as e:
                logging.error(f"Unsupported file type: {file_path}: {str(e)}")
                raise ValueError(f"Unsupported file type: {file_path}: {str(e)}")

    if not mileposts:
        raise ValueError("No valid data extracted from files.")

    elevations = np.array(elevations)
    if elevation_units == "Meters":
        elevations *= 3.28084

    mileposts = np.array(mileposts)
    sorted_idx = np.argsort(mileposts)
    mileposts = mileposts[sorted_idx]
    elevations = elevations[sorted_idx]
    unique_mask = np.diff(mileposts, prepend=mileposts[0] - 1) > 0
    mileposts = mileposts[unique_mask]
    elevations = elevations[unique_mask]

    profile_start_mp = mileposts.min()
    profile_end_mp = mileposts.max()

    purge_mileposts = mileposts.copy()
    system_mileposts = mileposts.copy()
    system_elevations = elevations.copy()

    logging.info(f"Profile processed: Start MP={profile_start_mp:.1f}, End MP={profile_end_mp:.1f}, Points={len(mileposts)}")
    action = preview_profile_dialog(dialog_root, system_mileposts, system_elevations, segment_bounds=segment_bounds, elevation_units_label="ft")
    if action == "approve":
        return purge_mileposts, elevations, system_mileposts, system_elevations, profile_start_mp, profile_end_mp
    elif action == "reselect":
        return process_elevation_profile(dialog_root, initial_inputs)
    else:
        raise ValueError("Profile preview cancelled.")


def main():
    logging.info("Starting Pipeline Pigging Simulation")
    print("Starting Pipeline Pigging Simulation")
    try:
        dialog_root = tk.Tk()
        dialog_root.title('Pipeline Pigging Simulation')
        dialog_root.geometry('1x1+0+0')
        try:
            dialog_root.attributes('-alpha', 0.0)
        except Exception:
            dialog_root.withdraw()
        dialog_root.update_idletasks()
        initial_inputs = {
            'elevation_format': 'TXT',
            'elevation_units': 'Feet',
            'purge_start_mp': 0
        }
        logging.info("Calling process_elevation_profile")
        purge_mileposts, elevations, system_mileposts, system_elevations, profile_start_mp, profile_end_mp = process_elevation_profile(dialog_root, initial_inputs)
        logging.info("Calling get_user_inputs")
        inputs = get_user_inputs(dialog_root, profile_start_mp, profile_end_mp)
        inputs['elevation_format'] = initial_inputs['elevation_format']
        inputs['elevation_units'] = initial_inputs['elevation_units']

        print_inputs(inputs)

        if len(system_mileposts) > inputs['resolution'] * 2:
            new_system_mp = np.linspace(profile_start_mp, profile_end_mp, inputs['resolution'] * 2)
            cs_system = CubicSpline(system_mileposts, system_elevations)
            system_elevations = cs_system(new_system_mp)
            system_mileposts = new_system_mp

        mask = (purge_mileposts >= inputs['purge_start_mp']) & (purge_mileposts <= inputs['purge_end_mp'])
        purge_mileposts = purge_mileposts[mask]
        elevations = elevations[mask]

        resample_to_resolution = bool(inputs.get('resample_profile_to_resolution', True))
        cs_purge = CubicSpline(system_mileposts, system_elevations, extrapolate=True)

        if resample_to_resolution:
            try:
                purge_res = int(inputs.get('resolution', 500))
            except Exception:
                purge_res = 500
            purge_res = max(2, purge_res)

            new_purge_mp = np.linspace(inputs['purge_start_mp'], inputs['purge_end_mp'], purge_res)
            elevations = cs_purge(new_purge_mp)
            purge_mileposts = new_purge_mp

            logging.info(
                f"Resampled purge segment to {purge_res} points "
                f"from MP {inputs['purge_start_mp']:.3f} to {inputs['purge_end_mp']:.3f}"
            )
        else:
            if purge_mileposts.size == 0:
                purge_mileposts = np.array([inputs['purge_start_mp'], inputs['purge_end_mp']])
                elevations = cs_purge(purge_mileposts)
            else:
                tol = 1e-9
                if purge_mileposts[0] - inputs['purge_start_mp'] > tol:
                    purge_mileposts = np.insert(purge_mileposts, 0, inputs['purge_start_mp'])
                    elevations = np.insert(elevations, 0, cs_purge(inputs['purge_start_mp']))
                if inputs['purge_end_mp'] - purge_mileposts[-1] > tol:
                    purge_mileposts = np.append(purge_mileposts, inputs['purge_end_mp'])
                    elevations = np.append(elevations, cs_purge(inputs['purge_end_mp']))

            logging.info(
                f"Using native purge profile points: {len(purge_mileposts)} points "
                f"from MP {purge_mileposts[0]:.3f} to {purge_mileposts[-1]:.3f}"
            )

        while True:
            logging.info("Running simulation")
            inputs.setdefault('n2_optional_cutoff_scf', None)

            results = run_simulation(dialog_root, inputs, purge_mileposts, elevations, system_mileposts, system_elevations)
            visualize_results(results, inputs)
            print_condensed_table(results, inputs)
            print_alarm_summary(results, inputs)
            resp = messagebox.askyesnocancel("Output Satisfactory", "Is the output satisfactory?", parent=dialog_root)
            if resp is True:
                export_results(dialog_root, inputs, results)
                if messagebox.askyesno(
                    "Generate Animation",
                    "Generate a hydraulic animation video (MP4) of the purge?\n\n"
                    "Shows elevation profile, MAOP envelope, liquid HGL (orange),\n"
                    "N2 gas pressure (blue), pig movement, and pressure annotations.",
                    parent=dialog_root,
                ):
                    try:
                        _fps = (tk.simpledialog.askinteger(
                            "Animation FPS",
                            "Frames per second (6–30):",
                            initialvalue=12, minvalue=6, maxvalue=30,
                            parent=dialog_root,
                        ) or 12)
                        _mf = (tk.simpledialog.askinteger(
                            "Animation Length",
                            "Max frames to render (100–800):\n"
                            "(More frames = longer animation, slower render)",
                            initialvalue=400, minvalue=100, maxvalue=800,
                            parent=dialog_root,
                        ) or 400)
                        generate_purge_animation(
                            results, inputs,
                            fps=_fps, max_frames=_mf,
                            parent_widget=dialog_root,
                        )
                    except Exception as _ae:
                        logging.error(f"Animation generation error: {_ae}")
                        messagebox.showerror("Animation Error", str(_ae),
                                              parent=dialog_root)
                break
            elif resp is None:
                logging.info("User cancelled after reviewing outputs; exiting without export.")
                break
            else:
                try:
                    dialog_root.update()
                    last_total_n2 = None
                    try:
                        last_i = int(results.get('last_valid_i', -1))
                        if ('cumulative_n2' in results) and (last_i is not None) and (last_i >= 0) and (last_i < len(results['cumulative_n2'])):
                            last_total_n2 = float(results['cumulative_n2'][last_i])
                    except Exception:
                        last_total_n2 = None
                    cutoff_current = inputs.get('n2_cutoff_scf', None)
                    cutoff_disp = 'auto' if (cutoff_current is None) else f"{float(cutoff_current):.0f}"
                    if last_total_n2 is None:
                        prompt = f"New TOTAL N2 injected cutoff (SCF) [blank=keep, 'auto'=clear]\nLast setting: {cutoff_disp}"
                    else:
                        prompt = f"New TOTAL N2 injected cutoff (SCF) [blank=keep, 'auto'=clear]\nLast setting: {cutoff_disp}\nLast run injected: {last_total_n2:.0f} SCF"
                    new_cutoff_str = tk.simpledialog.askstring('Adjust N2 Injection', prompt, parent=dialog_root)
                    if new_cutoff_str is not None:
                        s = str(new_cutoff_str).strip().lower()
                        if s == '':
                            pass
                        elif s in ('auto', 'none', 'clear', 'off'):
                            inputs['n2_cutoff_scf'] = None
                        else:
                            inputs['n2_cutoff_scf'] = float(s)

                    new_n2_rate = tk.simpledialog.askfloat(
                        "Input", f"New Max N2 Rate (SCFM) [optional] (Last={inputs.get('max_n2_rate_scfm', None)}):",
                        parent=dialog_root, minvalue=0.1
                    )
                    if new_n2_rate is not None:
                        inputs['max_n2_rate_scfm'] = new_n2_rate
                    new_exit_run = tk.simpledialog.askfloat(
                        "Input", f"New Min Exit Pressure (Run) (Last={inputs['exit_pressure_run']:.1f}):",
                        parent=dialog_root, minvalue=0
                    )
                    if new_exit_run is not None:
                        inputs['exit_pressure_run'] = new_exit_run
                    new_exit_end = tk.simpledialog.askfloat(
                        "Input", f"New Exit Pressure (End) (Last={inputs['exit_pressure_end']:.1f}):",
                        parent=dialog_root, minvalue=0
                    )
                    if new_exit_end is not None:
                        inputs['exit_pressure_end'] = new_exit_end
                    new_target_speed = tk.simpledialog.askfloat(
                        "Input", f"New Target Pig Speed (mph) (Last={inputs['target_pig_speed']:.1f}):",
                        parent=dialog_root, minvalue=0.1, maxvalue=inputs['max_pig_speed']
                    )
                    if new_target_speed is not None:
                        inputs['target_pig_speed'] = new_target_speed
                    dialog_root.update()
                    print_inputs(inputs)
                    continue
                except ValueError as e:
                    logging.error(f"Input revision error: {str(e)}")
                    messagebox.showerror("Input Error", f"Invalid input: {str(e)}", parent=dialog_root)

    except Exception as e:
        error_msg = f"Simulation failed: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        try:
            messagebox.showerror("Error", error_msg, parent=dialog_root)
        except Exception as tk_e:
            logging.error(f"Failed to show error messagebox: {str(tk_e)}")
            print(f"Error: Failed to show error messagebox: {str(tk_e)}")
        raise
    finally:
        dialog_root.destroy()


class TestPurgeSimulation(unittest.TestCase):
    def test_friction_loss(self):
        v, L, D, rho, mu, eps = 10, 1000, 0.5, 62.4, 1e-5, 0.00015
        loss = calculate_friction_loss(v, L, D, rho, mu, eps)
        self.assertGreater(loss, 0, "Friction loss should be positive.")

    def test_trendline_slope(self):
        mileposts = np.array([0, 1, 2])
        elevations = np.array([100, 110, 120])
        slope = calculate_trendline_slope(mileposts, elevations, 0, 2)
        self.assertEqual(slope, 10, "Slope should be 10 ft/mile.")


if __name__ == "__main__":
    configure_logging()
    try:
        verify_tkinter()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    main()
