"""
purge_report.py — Export SimResults to a formatted xlsx workbook.

Produces a client-deliverable spreadsheet that matches the "Formatted Purge Report"
layout used for field operations: header block, input parameters, condensed inflection-
point table, full step table, and an elevation/MOP proof sheet.

Sheets produced:
  Summary              — project header + key run metrics
  Simulation Inputs    — all input parameters (name / value)
  Condensed Results    — ~25-50 elevation-inflection-point rows (field roadmap)
  Full Results         — complete step data (every step or every-N-steps, ≤ 2000 rows)
  Elevation MOP Check  — high-point proof: drive_psi vs MOP at every local elevation peak
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from .simulator import SimResults, SimConfig, SimStep
from .physics import pipe_area_ft2

try:
    import openpyxl
    from openpyxl.styles import (
        Font, PatternFill, Alignment, Border, Side, numbers
    )
    from openpyxl.utils import get_column_letter
    _OPENPYXL = True
except ImportError:
    _OPENPYXL = False

try:
    import xlsxwriter
    _XLSXWRITER = True
except ImportError:
    _XLSXWRITER = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HDR_FILL   = "1F3864"   # dark navy — column headers
_SUB_FILL   = "2E75B6"   # mid-blue  — sub-headers / section labels
_WARN_FILL  = "FF0000"   # red       — MOP violation cells
_OK_FILL    = "E2EFDA"   # light green — pass cells
_TITLE_FILL = "17375E"   # very dark blue — workbook title

_HDR_FONT   = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
_BODY_FONT  = Font(name="Calibri", size=10)
_TITLE_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=14)
_LABEL_FONT = Font(name="Calibri", bold=True, size=10)

PSIG_PER_FT_PER_SG = 0.433   # (psi / ft) at SG = 1.0; multiply by SG for actual fluid
SCF_PER_BARREL = 5.615        # ft³ per barrel


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_purge_report(
    results: SimResults,
    path: str,
    scenario_name: str = "",
    project_info: Optional[dict] = None,
) -> None:
    """
    Write a formatted xlsx purge report to *path*.

    project_info keys (all optional):
        project, job_no, client, date, roughness_spec
    """
    if not _OPENPYXL:
        raise ImportError("openpyxl is required for xlsx export.  pip install openpyxl")

    cfg = results.config
    pi  = project_info or {}

    # Pre-compute derived step arrays once
    rows = _build_rows(results)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)   # remove default sheet

    _sheet_summary(wb, results, cfg, scenario_name, pi, rows)
    _sheet_inputs(wb, cfg, scenario_name, pi)
    _sheet_condensed(wb, rows, cfg, scenario_name)
    _sheet_full(wb, rows, cfg, scenario_name)
    _sheet_mop_proof(wb, rows, cfg, scenario_name)
    _sheet_booster_roadmap(wb, results, cfg, scenario_name)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------------------
# Per-row data builder
# ---------------------------------------------------------------------------

def _build_rows(results: SimResults) -> List[dict]:
    """Convert SimStep list to report-row dicts with all derived columns."""
    cfg  = results.config
    sg   = cfg.fluid_sg
    elev = cfg.elevation_profile             # shape (N, 2)
    end_mp = cfg.purge_end_mp

    elev_interp = np.interp(
        [s.pig_mp for s in results.steps],
        elev[:, 0], elev[:, 1]
    )
    exit_elev_interp = np.interp(
        [s.exit_mp for s in results.steps],
        elev[:, 0], elev[:, 1]
    )

    od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    area_ft2 = pipe_area_ft2(od, wt)        # ft² — assumed constant for simple 6" runs

    rows = []
    for i, s in enumerate(results.steps):
        pig_elev  = elev_interp[i]
        exit_elev = exit_elev_interp[i]

        # Hydrostatic head: positive when exit is HIGHER than pig (uphill run costs pressure)
        head_psi = (exit_elev - pig_elev) * sg * PSIG_PER_FT_PER_SG

        # Friction = what's left after head and exit back-pressure
        friction_psi = s.pig_face_psig - s.exit_psig - head_psi

        # Flow rate in BPH
        bph = s.pig_speed_mph * 5280.0 * area_ft2 / SCF_PER_BARREL

        # MOP headroom at pig position
        mop_at_pig = _mop_at_mp(s.pig_mp, cfg)
        mop_margin = mop_at_pig - s.pig_face_psig if mop_at_pig else None

        rows.append({
            "miles":          s.pig_mp,
            "elevation_ft":   pig_elev,
            "elapsed_hr":     s.t_hr,
            "drive_psi":      s.pig_face_psig,
            "friction_psi":   friction_psi,
            "head_psi":       head_psi,
            "exit_psi":       s.exit_psig,
            "inj_scfm":       s.injection_scfm,
            "cum_scf":        s.total_scf,
            "speed_mph":      s.pig_speed_mph,
            "miles_to_outlet": end_mp - s.pig_mp,
            "bph":            bph,
            "mop_at_pig":     mop_at_pig,
            "mop_margin":     mop_margin,
            "slack":          s.slack_line_risk,
            "mop_viol":       s.mop_violations > 0,
        })
    return rows


def _mop_at_mp(mp: float, cfg: SimConfig) -> Optional[float]:
    """Interpolate MOP at a given milepost from the joint list."""
    if not cfg.mop_joints:
        return None
    mps  = [j.mp       for j in cfg.mop_joints]
    mops = [j.mop_psig for j in cfg.mop_joints]
    return float(np.interp(mp, mps, mops))


# ---------------------------------------------------------------------------
# Inflection-point selector (for Condensed Results sheet)
# ---------------------------------------------------------------------------

def _select_inflection_rows(rows: List[dict], target: int = 40) -> List[dict]:
    """
    Return a reduced list hitting ~target rows by keeping:
      - start and end
      - local elevation minima and maxima (inflection points)
      - injection phase transitions (start / stop)
      - evenly-spaced fill to reach target count
    """
    if len(rows) <= target:
        return rows

    n = len(rows)
    elevs = np.array([r["elevation_ft"] for r in rows])
    inj   = np.array([r["inj_scfm"]     for r in rows])

    keep = {0, n - 1}

    # Elevation inflection points (local min/max using sign change of derivative)
    if n > 2:
        d1 = np.diff(elevs)
        for i in range(1, len(d1)):
            if d1[i - 1] * d1[i] < 0:     # sign flip = local extremum
                keep.add(i)

    # Injection phase transitions
    for i in range(1, n):
        was_injecting = inj[i - 1] > 1.0
        now_injecting = inj[i]      > 1.0
        if was_injecting != now_injecting:
            keep.add(max(0, i - 1))
            keep.add(i)

    # MOP warning / violation steps
    for i, r in enumerate(rows):
        if r["mop_viol"] or (r["mop_margin"] is not None and r["mop_margin"] < 50):
            keep.add(i)

    # Even-spaced fill if still under target
    if len(keep) < target:
        step = max(1, n // (target - len(keep)))
        for i in range(0, n, step):
            keep.add(i)

    # If still too many, thin the even-spaced additions
    ordered = sorted(keep)
    while len(ordered) > target * 1.5:
        evens = [i for j, i in enumerate(ordered) if j % 2 == 1
                 and i not in {0, n - 1}]
        for i in evens[:len(ordered) - target]:
            keep.discard(i)
        ordered = sorted(keep)

    return [rows[i] for i in sorted(keep)]


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

_COL_HEADERS = [
    "Miles",
    "Elevation (ft)",
    "Elapsed Time (hr)",
    "Drive Pressure (psi)",
    "Friction Loss (psi)",
    "Head Pressure (psi)",
    "Exit Pressure (psi)",
    "Injection Rate (SCFM)",
    "Cumulative N2 (SCF)",
    "Pig Speed (mph)",
    "Miles to Outlet",
    "Barrels / hr",
]

_COL_KEYS = [
    "miles", "elevation_ft", "elapsed_hr", "drive_psi", "friction_psi",
    "head_psi", "exit_psi", "inj_scfm", "cum_scf", "speed_mph",
    "miles_to_outlet", "bph",
]

_COL_FMT = [
    "0.000", "0.0", "0.000", "0.0", "0.0",
    "0.0", "0.0", "0.0", "#,##0", "0.000",
    "0.000", "0.0",
]


def _make_border(style="thin"):
    s = Side(style=style)
    return Border(left=s, right=s, top=s, bottom=s)


def _set_col_widths(ws, widths: List[float]):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _hdr_cell(ws, row, col, value, fill_hex=_HDR_FILL):
    c = ws.cell(row=row, column=col, value=value)
    c.font      = _HDR_FONT
    c.fill      = PatternFill("solid", fgColor=fill_hex)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border    = _make_border()
    return c


def _write_data_table(ws, start_row: int, data_rows: List[dict], zebra=True):
    """Write column headers + data rows; returns last row written."""
    # Column headers
    for ci, hdr in enumerate(_COL_HEADERS, start=1):
        _hdr_cell(ws, start_row, ci, hdr)
    r = start_row + 1

    fills = [PatternFill("solid", fgColor="EBF3FB"),
             PatternFill("solid", fgColor="FFFFFF")]

    for ri, row in enumerate(data_rows):
        fill = fills[ri % 2] if zebra else fills[1]
        for ci, (key, fmt) in enumerate(zip(_COL_KEYS, _COL_FMT), start=1):
            val = row.get(key)
            c = ws.cell(row=r, column=ci, value=val)
            c.fill   = fill
            c.font   = _BODY_FONT
            c.border = _make_border("thin")
            c.number_format = fmt
            c.alignment = Alignment(horizontal="right")
            # Highlight slack or MOP violation rows
            if row.get("mop_viol") and key == "drive_psi":
                c.fill = PatternFill("solid", fgColor="FF9999")
            elif row.get("slack") and key == "speed_mph":
                c.fill = PatternFill("solid", fgColor="FFFF99")
        r += 1
    return r - 1


def _write_header_block(ws, row, cfg: SimConfig, scenario_name: str,
                        pi: dict, results: SimResults) -> int:
    """Write the project info header block; returns next free row."""
    last = results.steps[-1] if results.steps else None
    od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    id_in  = od - 2 * wt

    fields_left = [
        ("PROJECT:", pi.get("project", scenario_name)),
        ("JOB No:", pi.get("job_no", "")),
        ("CLIENT:", pi.get("client", "")),
        ("LENGTH (miles):", f"{cfg.purge_end_mp - cfg.purge_start_mp:.2f}"),
        ("ROUGHNESS SPEC:", pi.get("roughness_spec", "Steel")),
        ("PURGED FLUID:", cfg.fluid_name if hasattr(cfg, "fluid_name") else "Diesel"),
        ("EXIT PRESSURE TARGET (psi):", f"{cfg.exit_pressure_run_psig:.0f}"),
        ("CUM. INJ. N2 (SCF):", f"{results.total_scf_injected:,.0f}"),
        ("PURGE START (MP):", f"{cfg.purge_start_mp:.3f}"),
        ("PURGE END (MP):", f"{cfg.purge_end_mp:.3f}"),
    ]
    fields_right = [
        ("DATE:", pi.get("date", datetime.now().strftime("%Y-%m-%d"))),
        ("SYSTEM:", pi.get("system", "")),
        ("NOMINAL PIPE SIZE (NPS):", f"{od:.3f} in OD"),
        ("OD (in):", f"{od:.3f}"),
        ("WALL THICKNESS (in):", f"{wt:.3f}"),
        ("ID (in):", f"{id_in:.3f}"),
        ("MAX N2 PRESSURE (psi):", f"{cfg.max_injection_psig:.0f}" if math.isfinite(cfg.max_injection_psig) else "unlimited"),
        ("FINAL N2 PRESSURE (psi):", f"{last.pig_face_psig:.0f}" if last else ""),
        ("PIG VELOCITY MAX (mph):", f"{cfg.max_speed_mph:.1f}"),
        ("PIG VELOCITY MIN (mph):", f"{cfg.min_speed_mph:.1f}"),
    ]

    for i, ((lbl, lval), (rlbl, rval)) in enumerate(zip(fields_left, fields_right)):
        r = row + i
        c = ws.cell(row=r, column=1, value=lbl)
        c.font = _LABEL_FONT
        c = ws.cell(row=r, column=3, value=lval)
        c.font = _BODY_FONT
        c = ws.cell(row=r, column=6, value=rlbl)
        c.font = _LABEL_FONT
        c = ws.cell(row=r, column=8, value=rval)
        c.font = _BODY_FONT

    return row + len(fields_left) + 1


def _sheet_summary(wb, results: SimResults, cfg: SimConfig,
                   scenario_name: str, pi: dict, rows: List[dict]):
    ws = wb.create_sheet("Summary")
    ws.freeze_panes = "A4"

    # Title row
    c = ws.cell(row=1, column=1, value="PURGE SIMULATION REPORT")
    c.font = _TITLE_FONT
    c.fill = PatternFill("solid", fgColor=_TITLE_FILL)
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.merge_cells("A1:L1")
    ws.row_dimensions[1].height = 28

    c = ws.cell(row=2, column=1, value=scenario_name)
    c.font = Font(name="Calibri", bold=True, size=12)
    c.alignment = Alignment(horizontal="center")
    ws.merge_cells("A2:L2")

    # Header block
    last = results.steps[-1] if results.steps else None
    od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    id_in  = od - 2 * wt

    def kv(r, label, val, col=1):
        ws.cell(row=r, column=col, value=label).font = _LABEL_FONT
        ws.cell(row=r, column=col+2, value=val).font = _BODY_FONT

    kv(4, "Project:",        pi.get("project", scenario_name))
    kv(5, "Client:",         pi.get("client", ""))
    kv(6, "Job No:",         pi.get("job_no", ""))
    kv(7, "Date:",           pi.get("date", datetime.now().strftime("%Y-%m-%d")))

    kv(4, "NPS OD (in):",    f"{od:.3f}",    col=7)
    kv(5, "Wall Thickness:", f"{wt:.3f} in",  col=7)
    kv(6, "Pipe ID:",        f"{id_in:.3f} in", col=7)
    kv(7, "Fluid:",          getattr(cfg, "fluid_name", "Diesel"), col=7)

    # Key results
    speeds = [r["speed_mph"] for r in rows]
    drives = [r["drive_psi"] for r in rows]

    kv(9,  "Status:",             "COMPLETED" if results.completed else "ABORTED", col=1)
    kv(10, "Route:",              f"MP {cfg.purge_start_mp:.3f} → {cfg.purge_end_mp:.3f}")
    kv(11, "Distance (mi):",      f"{cfg.purge_end_mp - cfg.purge_start_mp:.2f}")
    kv(12, "Duration (hr):",      f"{last.t_hr:.2f}" if last else "")
    kv(13, "Total N2 (SCF):",     f"{results.total_scf_injected:,.0f}")
    kv(14, "N2 Vented (SCF):",    f"{results.total_scf_vented:,.0f}")
    kv(15, "MOP Violations:",     f"{sum(s.mop_violations for s in results.steps):,}")

    kv(9,  "Avg Speed (mph):",    f"{sum(speeds)/len(speeds):.3f}" if speeds else "", col=7)
    kv(10, "Min Speed (mph):",    f"{min(speeds):.3f}"             if speeds else "", col=7)
    kv(11, "Max Speed (mph):",    f"{max(speeds):.3f}"             if speeds else "", col=7)
    kv(12, "Avg Drive (psi):",    f"{sum(drives)/len(drives):.1f}" if drives else "", col=7)
    kv(13, "Max Drive (psi):",    f"{max(drives):.1f}"             if drives else "", col=7)
    kv(14, "MOP (psi):",          f"{cfg.maop_psig:.0f}" if math.isfinite(cfg.maop_psig) else "per profile", col=7)
    kv(15, "Max Injection (psi):",f"{cfg.max_injection_psig:.0f}" if math.isfinite(cfg.max_injection_psig) else "unlimited", col=7)

    _set_col_widths(ws, [22, 2, 22, 2, 2, 2, 22, 2, 22, 2, 2, 2])


def _sheet_inputs(wb, cfg: SimConfig, scenario_name: str, pi: dict):
    ws = wb.create_sheet("Simulation Inputs")

    c = ws.cell(row=1, column=1, value=scenario_name)
    c.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells("A1:B1")

    _hdr_cell(ws, 2, 1, "Parameter")
    _hdr_cell(ws, 2, 2, "Value")

    od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    params = [
        ("Nominal Pipe Size (NPS)",     f"{od:.3f} in OD"),
        ("Outside Diameter (in)",       f"{od:.3f}"),
        ("Wall Thickness (in)",         f"{wt:.3f}"),
        ("Pipe ID (in)",                f"{od - 2*wt:.3f}"),
        ("Fluid Type",                  getattr(cfg, "fluid_name", "Diesel")),
        ("Fluid SG",                    f"{cfg.fluid_sg:.3f}"),
        ("Fluid Viscosity (cSt)",       f"{cfg.fluid_viscosity_cst:.2f}"),
        ("Pipe Roughness (ft)",         f"{cfg.fluid_roughness_ft}"),
        ("N2 Temperature (°F)",         f"{cfg.n2_temperature_f:.1f}"),
        ("Max Injection Pressure (psi)",f"{cfg.max_injection_psig:.0f}" if math.isfinite(cfg.max_injection_psig) else "unlimited"),
        ("Max Injection Rate (SCFM)",   f"{cfg.max_injection_scfm:.0f}"),
        ("Max Drive Pressure (psi)",    f"{cfg.max_drive_psig:.0f}" if math.isfinite(cfg.max_drive_psig) else "unlimited"),
        ("MAOP (psi)",                  f"{cfg.maop_psig:.0f}" if math.isfinite(cfg.maop_psig) else "per profile"),
        ("MOP Warning Fraction",        f"{cfg.mop_warning_fraction:.2f}"),
        ("Exit Pressure (run) (psi)",   f"{cfg.exit_pressure_run_psig:.0f}"),
        ("Exit Pressure (end) (psi)",   f"{cfg.exit_pressure_end_psig:.0f}"),
        ("Throttle Down (miles)",       f"{cfg.throttle_down_miles:.1f}"),
        ("Target Speed (mph)",          f"{cfg.target_speed_mph:.1f}"),
        ("Min Speed (mph)",             f"{cfg.min_speed_mph:.1f}"),
        ("Max Speed (mph)",             f"{cfg.max_speed_mph:.1f}"),
        ("Purge Start (MP)",            f"{cfg.purge_start_mp:.3f}"),
        ("Purge End (MP)",              f"{cfg.purge_end_mp:.3f}"),
        ("Pump Stations",               f"{len(cfg.pump_stations)}"),
        ("Booster Stations",            f"{len(cfg.booster_configs)}"),
        ("Check Valves",                f"{len(cfg.check_valves)}"),
        ("MOP Joints",                  f"{len(cfg.mop_joints)}"),
        ("BPCV",                        f"MP {cfg.bpcv.mp:.2f}" if cfg.bpcv else "none"),
    ]

    fills = [PatternFill("solid", fgColor="EBF3FB"),
             PatternFill("solid", fgColor="FFFFFF")]
    for ri, (lbl, val) in enumerate(params, start=3):
        ws.cell(row=ri, column=1, value=lbl).font  = _BODY_FONT
        ws.cell(row=ri, column=2, value=val).font   = _BODY_FONT
        ws.cell(row=ri, column=1).fill = fills[ri % 2]
        ws.cell(row=ri, column=2).fill = fills[ri % 2]

    _set_col_widths(ws, [36, 24])


def _sheet_condensed(wb, rows: List[dict], cfg: SimConfig, scenario_name: str):
    ws = wb.create_sheet("Condensed Results")
    ws.freeze_panes = "A4"

    title = f"{scenario_name} — Condensed (Inflection Points)"
    c = ws.cell(row=1, column=1, value=title)
    c.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells(f"A1:{get_column_letter(len(_COL_HEADERS))}1")

    c = ws.cell(row=2, column=1, value="Key elevation inflection points + phase transitions. "
                "Proves MOP not exceeded at high points; no slack at low points.")
    c.font = Font(name="Calibri", italic=True, size=9, color="595959")
    ws.merge_cells(f"A2:{get_column_letter(len(_COL_HEADERS))}2")

    condensed = _select_inflection_rows(rows, target=40)
    _write_data_table(ws, 3, condensed)

    widths = [9, 11, 11, 13, 13, 12, 12, 14, 14, 11, 12, 11]
    _set_col_widths(ws, widths)


def _sheet_full(wb, rows: List[dict], cfg: SimConfig, scenario_name: str,
                max_rows: int = 2000):
    ws = wb.create_sheet("Full Results")
    ws.freeze_panes = "A4"

    title = f"{scenario_name} — Full Results"
    c = ws.cell(row=1, column=1, value=title)
    c.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells(f"A1:{get_column_letter(len(_COL_HEADERS))}1")

    # Thin to max_rows if needed
    if len(rows) > max_rows:
        step = len(rows) // max_rows
        thin = rows[::step]
        if rows[-1] not in thin:
            thin.append(rows[-1])
    else:
        thin = rows

    resolution = f"{len(thin)}-point"
    c = ws.cell(row=2, column=1,
                value=f"All simulation steps ({resolution}). "
                      f"Total steps: {len(rows):,}.")
    c.font = Font(name="Calibri", italic=True, size=9, color="595959")
    ws.merge_cells(f"A2:{get_column_letter(len(_COL_HEADERS))}2")

    _write_data_table(ws, 3, thin, zebra=False)

    widths = [9, 11, 11, 13, 13, 12, 12, 14, 14, 11, 12, 11]
    _set_col_widths(ws, widths)


def _sheet_mop_proof(wb, rows: List[dict], cfg: SimConfig, scenario_name: str):
    """
    Elevation/MOP proof sheet — shows every local elevation peak with:
      milepost, elevation, drive_psi, MOP, headroom, slack risk
    This is the 'inflection point' proof the client wants.
    """
    ws = wb.create_sheet("Elevation MOP Check")
    ws.freeze_panes = "A4"

    c = ws.cell(row=1, column=1,
                value=f"{scenario_name} — Elevation & MOP Verification")
    c.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells("A1:I1")

    c = ws.cell(row=2, column=1,
                value="Local elevation HIGH points: verifies drive pressure never exceeded MOP "
                      "and pig never lost traction (no slack). "
                      "LOCAL LOWS: verifies no slack line.")
    c.font = Font(name="Calibri", italic=True, size=9, color="595959")
    ws.merge_cells("A2:I2")

    # Table headers
    proof_hdrs = [
        "Miles", "Elevation (ft)", "Point Type",
        "Drive Pressure (psi)", "MOP (psi)", "MOP Headroom (psi)",
        "% of MOP", "Pig Speed (mph)", "Slack Risk",
    ]
    for ci, h in enumerate(proof_hdrs, start=1):
        _hdr_cell(ws, 3, ci, h)

    # Select elevation extrema
    elevs = np.array([r["elevation_ft"] for r in rows])
    n     = len(elevs)

    proof_rows = []
    if n > 2:
        d1 = np.diff(elevs)
        for i in range(1, len(d1)):
            if d1[i - 1] * d1[i] < 0:
                ptype = "HIGH" if d1[i - 1] > 0 else "LOW"
                proof_rows.append((i, ptype))
    proof_rows = [(0, "START")] + proof_rows + [(n - 1, "END")]

    fills_ok   = PatternFill("solid", fgColor="E2EFDA")
    fills_warn = PatternFill("solid", fgColor="FFEB9C")
    fills_viol = PatternFill("solid", fgColor="FF9999")
    fills_alt  = [PatternFill("solid", fgColor="EBF3FB"),
                  PatternFill("solid", fgColor="FFFFFF")]

    for ri, (idx, ptype) in enumerate(proof_rows, start=4):
        row = rows[idx]
        mop  = row["mop_at_pig"] or math.inf
        head = row["mop_margin"] if row["mop_margin"] is not None else (mop - row["drive_psi"])
        pct  = row["drive_psi"] / mop * 100 if mop and math.isfinite(mop) else 0.0

        if row["mop_viol"] or head < 0:
            fill = fills_viol
        elif pct > 95 or row["slack"]:
            fill = fills_warn
        else:
            fill = fills_ok if ptype == "HIGH" else fills_alt[ri % 2]

        vals = [
            row["miles"], row["elevation_ft"], ptype,
            row["drive_psi"], mop if math.isfinite(mop) else None, head,
            pct, row["speed_mph"], "YES" if row["slack"] else "no",
        ]
        fmts = ["0.000", "0.0", "@", "0.0", "0.0", "0.0", "0.0%", "0.000", "@"]
        for ci, (v, fmt) in enumerate(zip(vals, fmts), start=1):
            c = ws.cell(row=ri, column=ci, value=v)
            c.fill   = fill
            c.font   = _BODY_FONT
            c.border = _make_border()
            c.number_format = fmt
            c.alignment = Alignment(horizontal="right" if ci not in (3, 9) else "center")

        # Colour-code headroom cell
        head_cell = ws.cell(row=ri, column=6)
        if head < 0:
            head_cell.font = Font(name="Calibri", size=10, bold=True, color="CC0000")
        elif head < 100:
            head_cell.font = Font(name="Calibri", size=10, bold=True, color="9C5700")

    _set_col_widths(ws, [9, 11, 10, 16, 12, 16, 10, 13, 10])


def _sheet_booster_roadmap(wb, results: SimResults, cfg: SimConfig, scenario_name: str):
    """
    Booster / spread operations roadmap sheet.
    Shows activation events, move sequence, and per-hour performance summary.
    If no boosters ran, records a single 'no boosters' note.
    """
    ws = wb.create_sheet("Booster Roadmap")

    c = ws.cell(row=1, column=1,
                value=f"{scenario_name} — Booster / Spread Operations Roadmap")
    c.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells("A1:J1")

    # ---- Section 1: Spread move log ----------------------------------------
    _hdr_cell(ws, 2, 1, "SPREAD MOVE LOG", fill_hex=_SUB_FILL)
    ws.merge_cells("A2:J2")

    move_hdrs = [
        "Spread #", "From MP", "To MP", "Departed (hr)",
        "Arrived (hr)", "Transit Time (hr)", "Reason",
    ]
    for ci, h in enumerate(move_hdrs, start=1):
        _hdr_cell(ws, 3, ci, h)

    if results.spread_events:
        for ri, ev in enumerate(results.spread_events, start=4):
            vals = [
                ev.get("spread_id", 0) + 1,
                ev.get("from_mp", ""),
                ev.get("to_mp", ""),
                ev.get("t_hr", ""),
                ev.get("arrive_at_hr", ""),
                round(ev.get("arrive_at_hr", 0) - ev.get("t_hr", 0), 3),
                ev.get("reason", ""),
            ]
            fmts = ["0", "0.000", "0.000", "0.000", "0.000", "0.000", "@"]
            fill = PatternFill("solid", fgColor="EBF3FB" if ri % 2 == 0 else "FFFFFF")
            for ci, (v, fmt) in enumerate(zip(vals, fmts), start=1):
                c = ws.cell(row=ri, column=ci, value=v)
                c.fill = fill; c.font = _BODY_FONT
                c.number_format = fmt
                c.border = _make_border()
        next_row = 4 + len(results.spread_events) + 1
    else:
        c = ws.cell(row=4, column=1, value="No spread moves — no mobile booster spreads in this scenario.")
        c.font = Font(name="Calibri", italic=True, size=10, color="595959")
        next_row = 6

    # ---- Section 2: First activation per booster station -------------------
    ws.cell(row=next_row, column=1)
    _hdr_cell(ws, next_row, 1, "BOOSTER STATION ACTIVATIONS", fill_hex=_SUB_FILL)
    ws.merge_cells(f"A{next_row}:J{next_row}")
    next_row += 1

    act_hdrs = [
        "Station", "MP", "Activated (hr)", "Pig MP at Activation",
        "Suction (psi)", "Discharge (psi)", "Flow (SCFM)", "Flow Limited?",
    ]
    for ci, h in enumerate(act_hdrs, start=1):
        _hdr_cell(ws, next_row, ci, h)
    next_row += 1

    seen: set = set()
    act_rows = []
    for step in results.steps:
        for bs in step.booster_states:
            key = (bs.get("mp"), bs.get("name"))
            if bs.get("running") and key not in seen:
                seen.add(key)
                act_rows.append({
                    "name":      bs.get("name", ""),
                    "mp":        bs.get("mp", ""),
                    "t_hr":      step.t_hr,
                    "pig_mp":    step.pig_mp,
                    "suct_psi":  bs.get("suction_psig", 0),
                    "disc_psi":  bs.get("discharge_psig", 0),
                    "flow_scfm": bs.get("flow_scfm", 0),
                    "lim":       "YES" if bs.get("flow_limited") else "no",
                })

    if act_rows:
        fills = [PatternFill("solid", fgColor="EBF3FB"),
                 PatternFill("solid", fgColor="FFFFFF")]
        for ri, ar in enumerate(act_rows):
            fill = fills[ri % 2]
            vals = [ar["name"], ar["mp"], ar["t_hr"], ar["pig_mp"],
                    ar["suct_psi"], ar["disc_psi"], ar["flow_scfm"], ar["lim"]]
            fmts = ["@", "0.000", "0.000", "0.000", "0.0", "0.0", "0.0", "@"]
            for ci, (v, fmt) in enumerate(zip(vals, fmts), start=1):
                c = ws.cell(row=next_row, column=ci, value=v)
                c.fill = fill; c.font = _BODY_FONT
                c.number_format = fmt; c.border = _make_border()
                if ar["lim"] == "YES" and ci == 8:
                    c.font = Font(name="Calibri", size=10, bold=True, color="CC0000")
            next_row += 1
    else:
        c = ws.cell(row=next_row, column=1,
                    value="No booster stations activated in this run.")
        c.font = Font(name="Calibri", italic=True, size=10, color="595959")
        next_row += 2

    # ---- Section 3: Booster N2 summary per station -------------------------
    # Tally total SCF moved through each booster
    ws.cell(row=next_row, column=1)
    _hdr_cell(ws, next_row, 1, "BOOSTER N2 THROUGHPUT SUMMARY", fill_hex=_SUB_FILL)
    ws.merge_cells(f"A{next_row}:J{next_row}")
    next_row += 1

    thr_hdrs = ["Station", "MP", "Active Steps", "Avg Flow (SCFM)",
                "Peak Flow (SCFM)", "Est. SCF Moved"]
    for ci, h in enumerate(thr_hdrs, start=1):
        _hdr_cell(ws, next_row, ci, h)
    next_row += 1

    # Aggregate by station mp
    from collections import defaultdict
    tally: dict = defaultdict(lambda: {"n": 0, "flow_sum": 0.0, "flow_max": 0.0, "name": ""})
    for step in results.steps:
        dt = step.t_hr  # for first step; subsequent steps use diff — approximate with avg dt
        for bs in step.booster_states:
            if bs.get("running"):
                mp = bs.get("mp", 0.0)
                f  = bs.get("flow_scfm", 0.0)
                tally[mp]["n"]        += 1
                tally[mp]["flow_sum"] += f
                tally[mp]["flow_max"]  = max(tally[mp]["flow_max"], f)
                tally[mp]["name"]      = bs.get("name", "")

    if tally:
        # Approximate run duration per step from total span
        total_hr = results.steps[-1].t_hr if results.steps else 1.0
        dt_avg   = total_hr / max(1, len(results.steps))
        fills = [PatternFill("solid", fgColor="EBF3FB"),
                 PatternFill("solid", fgColor="FFFFFF")]
        for ri, (mp, d) in enumerate(sorted(tally.items())):
            avg_flow = d["flow_sum"] / max(1, d["n"])
            est_scf  = avg_flow * d["n"] * dt_avg * 60   # SCFM * hr * 60 min/hr
            fill = fills[ri % 2]
            vals = [d["name"], mp, d["n"], avg_flow, d["flow_max"], est_scf]
            fmts = ["@", "0.000", "0", "0.0", "0.0", "#,##0"]
            for ci, (v, fmt) in enumerate(zip(vals, fmts), start=1):
                c = ws.cell(row=next_row, column=ci, value=v)
                c.fill = fill; c.font = _BODY_FONT
                c.number_format = fmt; c.border = _make_border()
            next_row += 1
    else:
        c = ws.cell(row=next_row, column=1, value="No booster throughput to report.")
        c.font = Font(name="Calibri", italic=True, size=10, color="595959")

    _set_col_widths(ws, [18, 9, 14, 16, 14, 12, 13, 12, 12, 12])
