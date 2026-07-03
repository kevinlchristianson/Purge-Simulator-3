"""
formatted_report.py  —  Client-facing purge deliverable xlsx.

Three sheets:

  FILL REPORT      — project header + 50-point field ops table
                     (matches the hand-formatted Bridger-style layout)
  Pressure Profile — embedded matplotlib chart: drive/friction/head/exit vs milepost
  Run Profile      — embedded matplotlib chart: pig speed + cumulative N2 vs time

This is the document given to the client and field crew.
The companion run_report.xlsx contains the complete technical data.
"""

from __future__ import annotations

import io
import math
import os
from datetime import datetime
from typing import List, Optional

import numpy as np

from .simulator import SimResults, SimConfig
from .purge_report import _build_rows, _select_n, _mop_at, _elev_interp, _PSIG_PER_FT

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage
    _OPENPYXL = True
except ImportError:
    _OPENPYXL = False

try:
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# ---------------------------------------------------------------------------
# Style — matches company dark-blue / light palette
# ---------------------------------------------------------------------------

_C_TITLE  = "1F3864"   # dark navy title bar
_C_HDR    = "2E75B6"   # column headers
_C_SUB    = "344E6E"   # sub-header / section label
_C_ZEBRA  = "EBF3FB"   # alternating row
_C_WHITE  = "FFFFFF"

_F_TITLE  = Font(name="Calibri", bold=True, color="FFFFFF", size=16)
_F_HDR    = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
_F_LBL    = Font(name="Calibri", bold=True, size=10)
_F_BODY   = Font(name="Calibri", size=10)
_F_ITAL   = Font(name="Calibri", italic=True, size=9, color="595959")
_F_NUM    = Font(name="Calibri", size=10)

_FILL_COLS = [
    "Miles", "Elevation (ft)", "Elapsed Time (hr)",
    "Drive Pressure (psi)", "Friction Loss (psi)", "Head Pressure (psi)",
    "Exit Pressure (psi)", "Injection Rate (SCFM)", "Cumulative N2 (SCF)",
    "Pig Speed (mph)",
]
_DATA_KEYS = [
    "miles", "elevation_ft", "elapsed_hr",
    "drive_psi", "friction_psi", "head_psi",
    "exit_psi", "inj_scfm", "cum_scf", "speed_mph",
]
_DATA_FMTS = [
    "0.000", "0.0", "0.000",
    "0.0", "0.0", "0.0",
    "0.0", "0.0", "#,##0", "0.000",
]


def _border():
    s = Side(style="thin")
    return Border(left=s, right=s, top=s, bottom=s)


def _fill(hex_c):
    return PatternFill("solid", fgColor=hex_c)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_formatted_report(
    results: SimResults,
    path: str,
    scenario_name: str = "",
    project_info: Optional[dict] = None,
) -> None:
    """
    Write the client-facing formatted report to *path*.

    project_info keys (all optional):
        project, job_no, client, system, date, notes,
        roughness_spec, fluid_name
    """
    if not _OPENPYXL:
        raise ImportError("openpyxl required:  pip install openpyxl")

    cfg = results.config
    pi  = project_info or {}
    rows = _build_rows(results)
    rows_50 = _select_n(rows, 50)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    _sheet_fill_report(wb, results, cfg, scenario_name, pi, rows_50)
    _sheet_pressure_profile(wb, results, cfg, scenario_name, rows)
    _sheet_run_profile(wb, results, cfg, scenario_name, rows)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------------------
# Sheet 1 — FILL REPORT (field ops document)
# ---------------------------------------------------------------------------

def _sheet_fill_report(wb, results, cfg, scenario_name, pi, rows_50):
    ws = wb.create_sheet("FILL REPORT")

    last  = results.steps[-1] if results.steps else None
    od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    id_in = od - 2.0 * wt
    length = cfg.purge_end_mp - cfg.purge_start_mp
    n_b   = len(cfg.booster_configs)
    n_ps  = len(cfg.pump_stations)

    # ---- Title bar -----------------------------------------------------------
    t = ws.cell(row=1, column=1, value="NITROGEN PURGE SIMULATION")
    t.font = _F_TITLE; t.fill = _fill(_C_TITLE)
    t.alignment = Alignment(horizontal="center", vertical="center")
    ws.merge_cells("A1:J1"); ws.row_dimensions[1].height = 34

    t2 = ws.cell(row=2, column=1, value=scenario_name)
    t2.font = Font(name="Calibri", bold=True, size=12)
    t2.alignment = Alignment(horizontal="center")
    ws.merge_cells("A2:J2"); ws.row_dimensions[2].height = 20

    # ---- Header block (2-column layout, rows 4–13) ---------------------------
    def _pair(r, lbl, val, rlbl, rval):
        ws.cell(row=r, column=1, value=lbl).font   = _F_LBL
        ws.cell(row=r, column=3, value=val).font   = _F_BODY
        ws.cell(row=r, column=6, value=rlbl).font  = _F_LBL
        ws.cell(row=r, column=8, value=rval).font  = _F_BODY

    _pair(4,  "PROJECT:",      pi.get("project", scenario_name),
               "DATE:",        pi.get("date", datetime.now().strftime("%Y-%m-%d")))
    _pair(5,  "JOB No:",       pi.get("job_no", ""),
               "SYSTEM:",      pi.get("system", ""))
    _pair(6,  "CLIENT:",       pi.get("client", ""),
               "NPS:",         f"{od:.3f} in OD")
    _pair(7,  "LENGTH (mi):",  f"{length:.2f}",
               "OD (in):",     f"{od:.3f}")
    _pair(8,  "ROUGHNESS:",    pi.get("roughness_spec", "Steel pipe"),
               "WALL THICK (in):", f"{wt:.3f}")
    _pair(9,  "PURGED FLUID:", pi.get("fluid_name", getattr(cfg, "fluid_name", "Diesel")),
               "ID (in):",     f"{id_in:.3f}")
    _pair(10, "EXIT PRESS TARGET (psi):", f"{cfg.exit_pressure_run_psig:.0f}",
               "MAX N2 PRESS (psi):",
               f"{cfg.max_injection_psig:.0f}" if math.isfinite(cfg.max_injection_psig) else "1500")
    _pair(11, "CUM. INJ. N2 (SCF):", f"{results.total_scf_n2:,.0f}",
               "FINAL N2 PRESS (psi):", f"{last.pig_face_psig:.0f}" if last else "")
    _pair(12, "PURGE START (MP):", f"{cfg.purge_start_mp:.3f}",
               "PIG VEL MAX (mph):", f"{cfg.max_speed_mph:.2f}")
    _pair(13, "PURGE END (MP):",   f"{cfg.purge_end_mp:.3f}",
               "PIG VEL MIN (mph):", f"{cfg.min_speed_mph:.2f}")
    _pair(14, "PUMP STATIONS:", f"{n_ps}",
               "BOOSTER STATIONS:", f"{n_b}")
    _pair(15, "DURATION (hr):", f"{last.t_hr:.2f}" if last else "",
               "RESOLUTION:", f"{len(rows_50)}-point")

    # Separator
    for col in range(1, 11):
        ws.cell(row=16, column=col).fill = _fill(_C_TITLE)
    ws.row_dimensions[16].height = 4

    # ---- 50-point data table -------------------------------------------------
    hdr_row = 17
    for ci, h in enumerate(_FILL_COLS, 1):
        c = ws.cell(row=hdr_row, column=ci, value=h)
        c.font = _F_HDR; c.fill = _fill(_C_HDR)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = _border()
    ws.row_dimensions[hdr_row].height = 30

    fills = [_fill(_C_ZEBRA), _fill(_C_WHITE)]
    for ri, row in enumerate(rows_50):
        r = hdr_row + 1 + ri
        row_fill = fills[ri % 2]
        for ci, (key, fmt) in enumerate(zip(_DATA_KEYS, _DATA_FMTS), 1):
            v = row.get(key)
            c = ws.cell(row=r, column=ci, value=v)
            c.font = _F_NUM; c.fill = row_fill
            c.number_format = fmt; c.border = _border()
            c.alignment = Alignment(horizontal="right")
            # Flag MOP-near drive pressure in orange font
            if key == "drive_psi" and row.get("mop_margin") is not None:
                if row["mop_margin"] < 0:
                    c.font = Font(name="Calibri", size=10, bold=True, color="CC0000")
                elif row["mop_margin"] < 100:
                    c.font = Font(name="Calibri", size=10, bold=True, color="9C5700")
            # Flag slack risk in speed column
            if key == "speed_mph" and row.get("slack"):
                c.fill = _fill("FF9999")

    # Notes footer
    note_row = hdr_row + len(rows_50) + 2
    if n_b > 0:
        booster_note = "BOOSTER STATIONS: " + ", ".join(
            f"{b.name} @ MP {b.mp:.1f}" for b in cfg.booster_configs)
        ws.cell(row=note_row, column=1, value=booster_note).font = _F_ITAL
        ws.merge_cells(f"A{note_row}:J{note_row}")
        note_row += 1
    if n_ps > 0:
        pump_note = "PUMP STATIONS: " + ", ".join(
            f"{p.name} @ MP {p.mp:.1f}" for p in cfg.pump_stations)
        ws.cell(row=note_row, column=1, value=pump_note).font = _F_ITAL
        ws.merge_cells(f"A{note_row}:J{note_row}")

    _set_col_widths(ws, [15, 2, 20, 2, 2, 25, 2, 20, 2, 2])

    # Wide data columns
    col_widths_data = [10, 12, 12, 14, 14, 13, 13, 15, 15, 12]
    for i, w in enumerate(col_widths_data, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _set_col_widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ---------------------------------------------------------------------------
# Sheet 2 — Pressure Profile (embedded chart)
# ---------------------------------------------------------------------------

def _sheet_pressure_profile(wb, results, cfg, scenario_name, rows):
    ws = wb.create_sheet("Pressure Profile")

    t = ws.cell(row=1, column=1,
                value=f"{scenario_name}  —  Pressure Profile (Pig-Centric)")
    t.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells("A1:N1")

    sub = ws.cell(row=2, column=1,
                  value="Drive, friction, head, and exit pressure vs pig milepost. "
                        "Elevation profile (shaded) shows terrain. "
                        "MAOP line in red.")
    sub.font = _F_ITAL
    ws.merge_cells("A2:N2")

    if not _MPL:
        ws.cell(row=3, column=1, value="matplotlib not available — chart skipped")
        return

    thin = _select_n(rows, 500)
    mps       = [r["miles"]       for r in thin]
    drive     = [r["drive_psi"]   for r in thin]
    friction  = [r["friction_psi"]for r in thin]
    head      = [r["head_psi"]    for r in thin]
    exit_p    = [r["exit_psi"]    for r in thin]
    elevs     = [r["elevation_ft"]for r in thin]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#F7FBFF")

    # Elevation as shaded background (right axis)
    ax2 = ax.twinx()
    ax2.fill_between(mps, elevs, alpha=0.12, color="steelblue", label="_nolegend_")
    ax2.plot(mps, elevs, color="navy", lw=0.8, alpha=0.4, label="Elevation (ft)")
    ax2.set_ylabel("Elevation (ft)", color="navy", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="navy")
    ax2.set_ylim(min(elevs) * 0.9, max(elevs) * 1.25 if max(elevs) > 0 else 1000)

    # Pressure lines (left axis)
    ax.plot(mps, drive,    color="#1F77B4", lw=2.0, label="Drive Pressure")
    ax.plot(mps, friction, color="#FF7F0E", lw=1.5, label="Friction Loss")
    ax.plot(mps, head,     color="#2CA02C", lw=1.5, label="Head Pressure")
    ax.plot(mps, exit_p,   color="#9467BD", lw=1.5, ls="--", label="Exit Pressure")

    # MAOP line
    if math.isfinite(cfg.maop_psig):
        ax.axhline(cfg.maop_psig, color="red", ls="--", lw=1.2, label=f"MAOP ({cfg.maop_psig:.0f} psi)")
    if math.isfinite(cfg.max_drive_psig) and cfg.max_drive_psig != cfg.maop_psig:
        ax.axhline(cfg.max_drive_psig, color="darkorange", ls=":", lw=1.2,
                   label=f"Max Drive ({cfg.max_drive_psig:.0f} psi)")

    ax.set_xlabel("Milepost", fontsize=11)
    ax.set_ylabel("Pressure (psi)", fontsize=11)
    ax.set_title(f"{scenario_name} — Pressure Profile", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()

    img = _fig_to_xl_image(fig)
    img.anchor = "A4"
    img.width  = 900
    img.height = 480
    ws.add_image(img)


# ---------------------------------------------------------------------------
# Sheet 3 — Run Profile (embedded chart)
# ---------------------------------------------------------------------------

def _sheet_run_profile(wb, results, cfg, scenario_name, rows):
    ws = wb.create_sheet("Run Profile")

    t = ws.cell(row=1, column=1,
                value=f"{scenario_name}  —  Run Profile (Pig Speed & N2 Inventory)")
    t.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells("A1:N1")

    sub = ws.cell(row=2, column=1,
                  value="Left Y: pig speed (mph) vs milepost.  "
                        "Right Y: cumulative N2 injected (SCF).  "
                        "Elevation terrain in background.")
    sub.font = _F_ITAL
    ws.merge_cells("A2:N2")

    if not _MPL:
        ws.cell(row=3, column=1, value="matplotlib not available — chart skipped")
        return

    thin = _select_n(rows, 500)
    mps   = [r["miles"]     for r in thin]
    speed = [r["speed_mph"] for r in thin]
    n2    = [r["cum_scf"]   for r in thin]
    elevs = [r["elevation_ft"] for r in thin]
    inj   = [r["inj_scfm"] for r in thin]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#F7FBFF")

    # Elevation background
    ax3 = ax.twinx()
    ax3.fill_between(mps, elevs, alpha=0.10, color="steelblue")
    ax3.set_ylabel("")
    ax3.tick_params(axis="y", labelright=False)
    ax3.set_ylim(min(elevs) * 0.9, max(elevs) * 2.0 if max(elevs) > 0 else 1000)

    # Speed (left axis)
    ax.plot(mps, speed, color="#1F77B4", lw=2.0, label="Pig Speed (mph)")
    ax.axhline(cfg.target_speed_mph, color="#1F77B4", ls=":", lw=1.0,
               label=f"Target ({cfg.target_speed_mph:.1f} mph)")
    if cfg.min_speed_mph > 0:
        ax.axhline(cfg.min_speed_mph, color="red", ls="--", lw=0.8,
                   label=f"Min Speed ({cfg.min_speed_mph:.1f} mph)")
    ax.set_xlabel("Milepost", fontsize=11)
    ax.set_ylabel("Pig Speed (mph)", color="#1F77B4", fontsize=11)
    ax.tick_params(axis="y", labelcolor="#1F77B4")

    # N2 cumulative (right axis)
    ax2 = ax.twinx()
    ax2.spines["right"].set_position(("outward", 0))
    ax2.plot(mps, n2, color="#2CA02C", lw=1.8, ls="-", label="Cumulative N2 (SCF)")
    ax2.set_ylabel("Cumulative N2 (SCF)", color="#2CA02C", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#2CA02C")
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
                                        else f"{x:,.0f}"))

    # Injection rate — show as shaded bars when >0
    inj_arr = np.array(inj)
    mp_arr  = np.array(mps)
    if inj_arr.max() > 0:
        ax_inj = ax.twinx()
        ax_inj.spines["right"].set_visible(False)
        ax_inj.tick_params(axis="y", right=False, labelright=False)
        ax_inj.fill_between(mp_arr, inj_arr, alpha=0.12, color="green",
                            label="_nolegend_")
        ax_inj.set_ylim(0, inj_arr.max() * 6)

    ax.set_title(f"{scenario_name} — Run Profile", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()

    img = _fig_to_xl_image(fig)
    img.anchor = "A4"
    img.width  = 900
    img.height = 480
    ws.add_image(img)


# ---------------------------------------------------------------------------
# Utility: matplotlib figure -> openpyxl Image (in-memory, no temp file)
# ---------------------------------------------------------------------------

def _fig_to_xl_image(fig) -> XLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return XLImage(buf)
