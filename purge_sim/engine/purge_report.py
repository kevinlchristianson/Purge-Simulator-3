"""
purge_report.py  —  Comprehensive run-data xlsx workbook.

Produced after every headless run; intended as the complete technical record
to be delivered to the client on request.  Seven sheets:

  1. Summary             — project header + full run metrics
  2. Simulation Inputs   — all parameters including booster detail
  3. Condensed Results   — 50-point evenly-spaced field-ops roadmap
  4. Full Results        — up to 1 000-point pig-centric table
  5. Elevation MOP Check — HGL-powered: pig-position rows x inflection-point cols
  6. Booster Roadmap     — per-step booster state (condensed + summary)
  7. Raw Data            — every SimStep field at every simulation timestep
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from .simulator import SimResults, SimConfig, SimStep
from .log_export import legacy_exit_schedule
from .hgl import compute_hgl
from .physics import pipe_area_ft2
from ..branding import xlsx_color

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    _OPENPYXL = True
except ImportError:
    _OPENPYXL = False


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_C_NAVY    = "brand:primary"     # main headers (the brand chosen in Settings, see purge_sim/branding.py)
_C_BLUE    = "brand:secondary"   # section labels
_C_DKBLUE  = "brand:dark"        # title bar
_C_RED     = "FF0000"   # violation
_C_AMBER   = "FFEB9C"   # warning fill
_C_GREEN   = "E2EFDA"   # ok / safe fill
_C_ZEBRA_A = "brand:tint"        # zebra row A
_C_ZEBRA_B = "FFFFFF"   # zebra row B (white)
_C_SLACK   = "FF9999"   # slack / overpressure cell
_C_OVER    = "F4B183"   # pig over max speed (the exit could not hold it)
_C_NEAR    = "FFD966"   # within 100 psi of MOP

_FNT_TITLE = Font(name="Calibri", bold=True, color="FFFFFF", size=14)
_FNT_HDR   = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
_FNT_SUB   = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
_FNT_LBL   = Font(name="Calibri", bold=True, size=10)
_FNT_BODY  = Font(name="Calibri", size=10)
_FNT_ITAL  = Font(name="Calibri", italic=True, size=9, color="595959")

_PSIG_PER_FT = 0.433   # psi per ft at SG=1; multiply by SG
_SCF_PER_BBL = 5.615   # ft3 per barrel


def _border(style="thin"):
    s = Side(style=style)
    return Border(left=s, right=s, top=s, bottom=s)


def _fill(hex_color):
    return PatternFill("solid", fgColor=xlsx_color(hex_color))


def _col_ltr(n):
    return get_column_letter(n)


def _hdr(ws, row, col, value, hex_bg=_C_NAVY, font=_FNT_HDR, wrap=True):
    c = ws.cell(row=row, column=col, value=value)
    c.font      = font
    c.fill      = _fill(hex_bg)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=wrap)
    c.border    = _border()
    return c


def _set_widths(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[_col_ltr(i)].width = w


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_purge_report(
    results: SimResults,
    path: str,
    scenario_name: str = "",
    project_info: Optional[dict] = None,
) -> None:
    """Write the 7-tab run-data xlsx to *path*."""
    if not _OPENPYXL:
        raise ImportError("openpyxl required:  pip install openpyxl")

    cfg = results.config
    pi  = project_info or {}
    rows = _build_rows(results)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    _sheet_summary(wb, results, cfg, scenario_name, pi, rows)
    _sheet_inputs(wb, cfg, scenario_name, pi)
    _sheet_condensed(wb, rows, cfg, scenario_name)
    _sheet_full(wb, rows, cfg, scenario_name)
    _sheet_mop_check(wb, results, cfg, scenario_name)
    _sheet_booster_roadmap(wb, results, cfg, scenario_name, rows)
    _sheet_raw_data(wb, results, cfg, scenario_name)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------------------
# Row builder — pig-centric derived columns
# ---------------------------------------------------------------------------

# "Exit Pressure" is what the exit (running pump suction, BPCV or tank inlet) holds at that
# moment: its minimum plus "Endpoint Added", the back-pressure a modulating exit put on to
# hold the pig at max speed. "Over Max Speed" marks the steps where it could not.
_COL_HDRS = [
    "Miles", "Elevation (ft)", "Elapsed Time (hr)",
    "Drive Pressure (psi)", "Friction Loss (psi)", "Head Pressure (psi)",
    "Exit Pressure (psi)", "Endpoint Added (psi)", "Injection Rate (SCFM)", "Cumulative N2 (SCF)",
    "Pig Speed (mph)", "Miles to Outlet", "Barrels / hr", "Over Max Speed",
]
_COL_KEYS = [
    "miles", "elevation_ft", "elapsed_hr",
    "drive_psi", "friction_psi", "head_psi",
    "exit_psi", "endpoint_added_psi", "inj_scfm", "cum_scf",
    "speed_mph", "miles_to_outlet", "bph", "overspeed",
]
_COL_FMTS = [
    "0.000", "0.0", "0.000",
    "0.0", "0.0", "0.0",
    "0.0", "0.0", "0.0", "#,##0",
    "0.000", "0.000", "0.0", "@",
]
_COL_WIDTHS = [9, 11, 11, 13, 13, 12, 12, 12, 14, 14, 11, 12, 11, 10]


def _mop_at(mp: float, cfg: SimConfig) -> Optional[float]:
    if not cfg.mop_joints:
        return cfg.maop_psig if math.isfinite(cfg.maop_psig) else None
    return float(np.interp(mp,
                           [j.mp for j in cfg.mop_joints],
                           [j.mop_psig for j in cfg.mop_joints]))


def _elev_interp(cfg: SimConfig):
    ep = np.asarray(cfg.elevation_profile, dtype=float)
    return lambda mp: float(np.interp(mp, ep[:, 0], ep[:, 1]))


def _build_rows(results: SimResults) -> List[dict]:
    cfg   = results.config
    sg    = cfg.fluid_sg
    end   = cfg.purge_end_mp
    elev  = _elev_interp(cfg)
    od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    area  = pipe_area_ft2(od, wt)

    rows = []
    _cum_b: dict = {}   # booster name -> cumulative SCF transferred (relayed) so far
    for s in results.steps:
        pe    = elev(s.pig_mp)
        ee    = elev(s.exit_mp)
        head  = (ee - pe) * sg * _PSIG_PER_FT
        fric  = s.pig_face_psig - s.exit_psig - head
        bph   = s.pig_speed_mph * 5280.0 * area / _SCF_PER_BBL
        mop   = _mop_at(s.pig_mp, cfg)
        # per-booster instantaneous flow + running cumulative relayed volume
        bflow: dict = {}
        for bs in (s.booster_states or []):
            nm = bs.get("name")
            if not nm:
                continue
            bflow[nm] = bs.get("flow_scfm", 0.0) or 0.0
            _cum_b[nm] = _cum_b.get(nm, 0.0) + (bs.get("scf_transferred", 0.0) or 0.0)
        rows.append({
            "miles":          s.pig_mp,
            "elevation_ft":   pe,
            "elapsed_hr":     s.t_hr,
            "drive_psi":      s.pig_face_psig,
            "friction_psi":   fric,
            "head_psi":       head,
            "exit_psi":       s.exit_psig,
            "exit_min_psi":   s.exit_min_psig,
            "endpoint_added_psi": s.endpoint_added_psi,
            "overspeed":      bool(s.overspeed),
            "inj_scfm":       s.injection_scfm,
            "cum_scf":        s.total_scf,
            "speed_mph":      s.pig_speed_mph,
            "miles_to_outlet": end - s.pig_mp,
            "bph":            bph,
            "mop":            mop,
            "mop_margin":     (mop - s.pig_face_psig) if mop else None,
            "slack":          s.slack_line_risk,
            "mop_viol":       s.mop_violations > 0,
            "booster_flow":   bflow,          # {name: SCFM this step}
            "booster_cum":    dict(_cum_b),   # {name: cumulative SCF relayed}
        })
    return rows


def _deployed_boosters(results: SimResults) -> List[tuple]:
    """Ordered (name, mp) of boosters that actually RUN during the run, by milepost.
    booster_states carries every configured booster (most with running=False); only the
    ones that actually run (deployed) should appear in reports/charts."""
    seen: dict = {}
    for s in results.steps:
        for bs in (s.booster_states or []):
            nm = bs.get("name")
            if nm and bs.get("running") and nm not in seen:
                seen[nm] = bs.get("mp", 0.0)
    return sorted(seen.items(), key=lambda kv: kv[1])


# ---------------------------------------------------------------------------
# Purge Report rows — fixed distance increments from the launch end
# ---------------------------------------------------------------------------

PURGE_REPORT_STEP_MI = 0.25

# landmark kind -> label (data/landmarks.py KINDS, repeated so the engine needn't import it)
_FEATURE_KIND = {
    "block_valve": "Block valve", "launcher_receiver": "Launcher / receiver",
    "aerial_marker": "Aerial marker", "ground_marker": "Above-ground marker", "crossing": "Crossing",
}
_INTERP_KEYS = ("elapsed_hr", "drive_psi", "friction_psi", "head_psi", "exit_psi",
                "exit_min_psi", "endpoint_added_psi", "inj_scfm", "cum_scf", "speed_mph", "bph")


def report_features(results: SimResults, landmarks: Optional[List[dict]] = None) -> List[dict]:
    """Stations, valves, markers and crossings inside the purged span, as [{mp, label}] in
    milepost order: the optional extra rows of the Purge Report. landmarks: the scenario's
    inputs.landmarks (block valves, AGMs, crossings...) from the ILI / PxP import."""
    cfg = results.config
    lo, hi = sorted((cfg.purge_start_mp, cfg.purge_end_mp))
    out = [{"mp": p.mp, "label": f"Pump station: {p.name}"} for p in cfg.pump_stations]
    out += [{"mp": c.mp, "label": f"Check valve: {c.name}"} for c in cfg.check_valves if not c.is_pump_station]
    if cfg.bpcv is not None:
        out.append({"mp": cfg.bpcv.mp, "label": f"Back-pressure control valve: {cfg.bpcv.name}"})
    out += [{"mp": mp, "label": f"N2 booster: {n}"} for n, mp in _deployed_boosters(results)]
    for lm in landmarks or []:
        try:
            mp = float(lm.get("mp"))
        except (TypeError, ValueError):
            continue
        kind = _FEATURE_KIND.get(lm.get("kind"), "Feature")
        name = (lm.get("name") or "").strip()
        out.append({"mp": mp, "label": kind if not name or name == kind else f"{kind}: {name}"})
    return sorted((f for f in out if lo - 1e-9 <= f["mp"] <= hi + 1e-9), key=lambda f: f["mp"])


def purge_report_rows(results: SimResults, step_mi: float = PURGE_REPORT_STEP_MI,
                      features: Optional[List[dict]] = None) -> List[dict]:
    """The Purge Report table: one row each time the pig has travelled another step_mi from
    the launch (plus the final pig position), values interpolated between the timesteps
    either side. "miles" counts from 0 at the launch, not the data file's milepost ("mp").
    features ([{mp, label}], see report_features) add a row at each one, with "feature" set."""
    rows = _build_rows(results)
    if not rows:
        return []
    cfg = results.config
    start = cfg.purge_start_mp
    sign = 1.0 if cfg.purge_end_mp >= start else -1.0
    dist = np.maximum.accumulate(np.array([sign * (r["miles"] - start) for r in rows], dtype=float))
    total = float(dist[-1])
    elev = _elev_interp(cfg) if len(cfg.elevation_profile) else None

    def at(d: float, feature: str = "") -> dict:
        i = int(np.searchsorted(dist, d - 1e-9))   # first step at or past d
        i = min(i, len(rows) - 1)
        b = rows[i]
        a = rows[i - 1] if i > 0 else b
        span = dist[i] - dist[i - 1] if i > 0 else 0.0
        f = (d - dist[i - 1]) / span if span > 1e-12 else 1.0
        f = min(max(f, 0.0), 1.0)
        row = {k: (a[k] + (b[k] - a[k]) * f if a.get(k) is not None and b.get(k) is not None else b.get(k))
               for k in _INTERP_KEYS}
        mp = start + sign * d
        mop = _mop_at(mp, cfg)
        row.update({
            "miles": d, "mp": mp, "feature": feature,
            "elevation_ft": elev(mp) if elev else b.get("elevation_ft"),
            "miles_to_outlet": abs(cfg.purge_end_mp - mp),
            "mop": mop, "mop_margin": (mop - row["drive_psi"]) if mop else None,
            "slack": bool(b.get("slack")), "mop_viol": bool(b.get("mop_viol")),
            "overspeed": bool(b.get("overspeed")),
            "booster_flow": b.get("booster_flow", {}), "booster_cum": b.get("booster_cum", {}),
        })
        return row

    n = int(math.floor(total / step_mi + 1e-9))
    out = [at(k * step_mi) for k in range(n + 1)]
    if total - n * step_mi > 1e-6:
        out.append(at(total))   # where the pig stopped
    for ft in features or []:
        d = sign * (float(ft["mp"]) - start)
        if 0.0 <= d <= total + 1e-9:
            out.append(at(d, ft.get("label") or "Feature"))
    out.sort(key=lambda r: (r["miles"], r["feature"] != ""))
    return out


def _select_n(rows: List[dict], n: int) -> List[dict]:
    """Select exactly n rows evenly spaced by pig milepost."""
    if len(rows) <= n:
        return rows
    idxs = np.round(np.linspace(0, len(rows) - 1, n)).astype(int)
    return [rows[i] for i in dict.fromkeys(idxs)]   # preserve order, no dupes


# ---------------------------------------------------------------------------
# Shared table writer
# ---------------------------------------------------------------------------

def _write_table(ws, start_row: int, data: List[dict], zebra=True) -> int:
    for ci, hdr in enumerate(_COL_HDRS, 1):
        _hdr(ws, start_row, ci, hdr)
    r = start_row + 1
    fills = [_fill(_C_ZEBRA_A), _fill(_C_ZEBRA_B)]
    for ri, row in enumerate(data):
        zf = fills[ri % 2] if zebra else fills[1]
        for ci, (key, fmt) in enumerate(zip(_COL_KEYS, _COL_FMTS), 1):
            v = row.get(key)
            if key == "overspeed":
                v = "YES" if v else ""
            c = ws.cell(row=r, column=ci, value=v)
            c.font   = _FNT_BODY
            c.fill   = zf
            c.border = _border()
            c.number_format = fmt
            c.alignment = Alignment(horizontal="right")
            if row.get("mop_viol") and key == "drive_psi":
                c.fill = _fill(_C_SLACK)
            elif row.get("overspeed") and key in ("speed_mph", "overspeed"):
                c.fill = _fill(_C_OVER)
            elif row.get("slack") and key == "speed_mph":
                c.fill = _fill(_C_AMBER)
        r += 1
    return r - 1


# ---------------------------------------------------------------------------
# Sheet 1 — Summary
# ---------------------------------------------------------------------------

def _sheet_summary(wb, results, cfg, scenario_name, pi, rows):
    ws = wb.create_sheet("Summary")

    # Title bar
    t = ws.cell(row=1, column=1, value="PURGE SIMULATION RUN REPORT")
    t.font = _FNT_TITLE; t.fill = _fill(_C_DKBLUE)
    t.alignment = Alignment(horizontal="center", vertical="center")
    ws.merge_cells("A1:L1"); ws.row_dimensions[1].height = 28

    ws.cell(row=2, column=1, value=scenario_name).font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells("A2:L2")

    last  = results.steps[-1] if results.steps else None
    od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    speeds = [r["speed_mph"] for r in rows]
    drives = [r["drive_psi"] for r in rows]

    def kv(r, lbl, val, col=1):
        ws.cell(row=r, column=col).value = lbl;   ws.cell(row=r, column=col).font = _FNT_LBL
        ws.cell(row=r, column=col+2).value = val; ws.cell(row=r, column=col+2).font = _FNT_BODY

    kv(4, "Project:",         pi.get("project", scenario_name))
    kv(5, "Client:",          pi.get("client", ""))
    kv(6, "Job No:",          pi.get("job_no", ""))
    kv(7, "Date:",            pi.get("date", datetime.now().strftime("%Y-%m-%d")))
    kv(8, "Notes:",           pi.get("notes", ""))

    kv(4, "Pipe OD (in):",        f"{od:.3f}",    col=7)
    kv(5, "Wall Thickness (in):", f"{wt:.3f}",    col=7)
    kv(6, "Pipe ID (in):",        f"{od-2*wt:.3f}", col=7)
    kv(7, "Fluid SG:",            f"{cfg.fluid_sg:.3f}", col=7)
    kv(8, "Fluid Viscosity:",     f"{cfg.fluid_viscosity_cst:.2f} cSt", col=7)

    kv(10, "Status:",         "COMPLETED" if results.completed else f"ABORTED: {results.abort_reason}")
    kv(11, "Route:",          f"MP {cfg.purge_start_mp:.3f} -> MP {cfg.purge_end_mp:.3f}")
    kv(12, "Distance (mi):",  f"{cfg.purge_end_mp - cfg.purge_start_mp:.2f}")
    kv(13, "Duration (hr):",  f"{last.t_hr:.3f}" if last else "")
    kv(14, "Sim Steps:",      f"{len(results.steps):,}")
    kv(15, "Total N2 (SCF):", f"{results.total_scf_n2:,.0f}")
    kv(16, "N2 Vented (SCF):",f"{results.total_scf_vented:,.0f}")
    kv(17, "MOP Violations:", f"{sum(s.mop_violations for s in results.steps):,}")
    over = [s for s in results.steps if s.overspeed]
    kv(18, "Over Max Speed:", (f"{len(over):,} steps (up to {max(s.overspeed_mph for s in over):.2f} mph over, "
                               f"MP {over[0].pig_mp:.2f} to {over[-1].pig_mp:.2f})") if over else "0 steps")

    kv(10, "Target Speed (mph):", f"{cfg.target_speed_mph:.2f}", col=7)
    kv(11, "Avg Speed (mph):",    f"{sum(speeds)/len(speeds):.3f}" if speeds else "", col=7)
    kv(12, "Min Speed (mph):",    f"{min(speeds):.3f}" if speeds else "", col=7)
    kv(13, "Max Speed (mph):",    f"{max(speeds):.3f}" if speeds else "", col=7)
    kv(14, "Avg Drive (psi):",    f"{sum(drives)/len(drives):.1f}" if drives else "", col=7)
    kv(15, "Max Drive (psi):",    f"{max(drives):.1f}" if drives else "", col=7)
    kv(16, "MAOP (psi):",         f"{cfg.maop_psig:.0f}" if math.isfinite(cfg.maop_psig) else "per MOP profile", col=7)
    kv(17, "Max Inj Pressure:",   f"{cfg.max_injection_psig:.0f} psi" if math.isfinite(cfg.max_injection_psig) else "unlimited", col=7)
    exits = [s.exit_psig for s in results.steps]
    kv(18, "Exit Pressure (psi):", (f"{min(exits):.0f} to {max(exits):.0f} held "
                                    f"({getattr(cfg, 'exit_behavior', 'modulating')} exit)") if exits else "", col=7)

    _set_widths(ws, [22, 2, 24, 2, 2, 2, 24, 2, 20, 2, 2, 2])


# ---------------------------------------------------------------------------
# Sheet 2 — Simulation Inputs
# ---------------------------------------------------------------------------

def _exit_input_rows(cfg) -> List[tuple]:
    """Simulation Inputs rows for the exit: its minimum, its behavior and how far it may
    rise, plus the legacy position schedule when a scenario still carries one."""
    behavior = getattr(cfg, "exit_behavior", "modulating")
    mx = getattr(cfg, "exit_max_pressure_psig", None)
    rows = [
        ("Exit Pressure — Minimum (psi)",    f"{cfg.exit_pressure_run_psig:.0f}"),
        ("Exit Behavior",                    "fixed (pig may overspeed, flagged)" if behavior == "fixed"
                                             else "modulating (rises to hold max pig speed)"),
        ("Exit Pressure — Maximum (psi)",    "= minimum (fixed)" if behavior == "fixed"
                                             else (f"{float(mx):.0f}" if mx is not None else "MOP at the exit")),
    ]
    legacy = legacy_exit_schedule(cfg)
    if legacy:
        rows.append(("Exit Schedule (legacy)", legacy))
    return rows


def _sheet_inputs(wb, cfg, scenario_name, pi):
    ws = wb.create_sheet("Simulation Inputs")

    ws.cell(row=1, column=1, value=scenario_name).font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells("A1:B1")
    _hdr(ws, 2, 1, "Parameter"); _hdr(ws, 2, 2, "Value")

    od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    params = [
        ("--- PIPE ---", ""),
        ("Nominal Pipe Size (NPS)",          f"{od:.3f} in OD"),
        ("Outside Diameter (in)",            f"{od:.3f}"),
        ("Wall Thickness (in)",              f"{wt:.3f}"),
        ("Pipe ID (in)",                     f"{od - 2*wt:.3f}"),
        ("--- FLUID ---", ""),
        ("Fluid Type",                       getattr(cfg, "fluid_name", "Diesel")),
        ("Fluid SG",                         f"{cfg.fluid_sg:.4f}"),
        ("Fluid Viscosity (cSt)",            f"{cfg.fluid_viscosity_cst:.3f}"),
        ("Pipe Roughness (ft)",              f"{cfg.fluid_roughness_ft}"),
        ("N2 Temperature (deg F)",           f"{cfg.n2_temperature_f:.1f}"),
        ("--- OPERATING LIMITS ---", ""),
        ("Max Injection Pressure (psi)",     f"{cfg.max_injection_psig:.0f}" if math.isfinite(cfg.max_injection_psig) else "unlimited"),
        ("Max Injection Rate (SCFM)",        f"{cfg.max_injection_scfm:.0f}"),
        ("Max Drive Pressure (psi)",         f"{cfg.max_drive_psig:.0f}" if math.isfinite(cfg.max_drive_psig) else "unlimited"),
        ("MAOP (psi)",                       f"{cfg.maop_psig:.0f}" if math.isfinite(cfg.maop_psig) else "per MOP profile"),
        ("MOP Warning Fraction",             f"{cfg.mop_warning_fraction:.2f}"),
        ("--- PURGE PARAMETERS ---", ""),
        ("Purge Start (MP)",                 f"{cfg.purge_start_mp:.3f}"),
        ("Purge End (MP)",                   f"{cfg.purge_end_mp:.3f}"),
        *_exit_input_rows(cfg),
        ("Target Speed (mph)",               f"{cfg.target_speed_mph:.2f}"),
        ("Min Speed (mph)",                  f"{cfg.min_speed_mph:.2f}"),
        ("Max Speed (mph)",                  f"{cfg.max_speed_mph:.2f}"),
        ("N2 Budget (SCF)",                  f"{cfg.n2_budget_scf:,.0f}" if cfg.n2_budget_scf else "unlimited"),
        ("--- INFRASTRUCTURE ---", ""),
        ("Pump Stations",                    f"{len(cfg.pump_stations)}"),
        ("Check Valves",                     f"{len(cfg.check_valves)}"),
        ("BPCV",                             f"MP {cfg.bpcv.mp:.2f}" if cfg.bpcv else "none"),
        ("MOP Profile Joints",               f"{len(cfg.mop_joints)}"),
    ]

    # Booster section
    if cfg.booster_configs:
        params.append(("--- BOOSTER STATIONS ---", ""))
        params.append(("Number of Boosters",       f"{len(cfg.booster_configs)}"))
        params.append(("Spread Discharge (psi)",   f"{cfg.booster_configs[0].discharge_psig:.0f}"))
        params.append(("Spread Suction Min (psi)", f"{cfg.booster_configs[0].suction_min_psig:.0f}"))
        params.append(("Max Flow / Booster (SCFM)",f"{cfg.booster_configs[0].max_flow_scfm:.0f}"))
        for b in cfg.booster_configs:
            params.append((f"  Booster: {b.name}", f"MP {b.mp:.2f}"))
    else:
        params.append(("Booster Stations", "none"))

    fills = [_fill(_C_ZEBRA_A), _fill(_C_ZEBRA_B)]
    for ri, (lbl, val) in enumerate(params, start=3):
        is_section = lbl.startswith("---")
        if is_section:
            c = ws.cell(row=ri, column=1, value=lbl.strip("-").strip())
            c.font = _FNT_HDR; c.fill = _fill(_C_BLUE)
            ws.merge_cells(f"A{ri}:B{ri}")
        else:
            ws.cell(row=ri, column=1, value=lbl).font = _FNT_BODY
            ws.cell(row=ri, column=2, value=val).font  = _FNT_BODY
            ws.cell(row=ri, column=1).fill = fills[ri % 2]
            ws.cell(row=ri, column=2).fill = fills[ri % 2]

    _set_widths(ws, [38, 26])


# ---------------------------------------------------------------------------
# Sheet 3 — Condensed Results (50-point evenly spaced)
# ---------------------------------------------------------------------------

def _sheet_condensed(wb, rows, cfg, scenario_name):
    ws = wb.create_sheet("Condensed Results")
    ws.freeze_panes = "A3"

    n = 50
    title = f"{scenario_name}  —  {n}-Point Resolution (Field Operations Roadmap)"
    t = ws.cell(row=1, column=1, value=title)
    t.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells(f"A1:{_col_ltr(len(_COL_HDRS))}1")

    sub = ws.cell(row=2, column=1,
                  value="Evenly-spaced pig-position checkpoints. "
                        "Injection schedule for field crew. "
                        "High points prove no slack; low points prove MOP not exceeded.")
    sub.font = _FNT_ITAL
    ws.merge_cells(f"A2:{_col_ltr(len(_COL_HDRS))}2")

    condensed = _select_n(rows, n)
    _write_table(ws, 3, condensed)
    _set_widths(ws, _COL_WIDTHS)


# ---------------------------------------------------------------------------
# Sheet 4 — Full Results (1000-point evenly spaced)
# ---------------------------------------------------------------------------

def _sheet_full(wb, rows, cfg, scenario_name, max_rows=1000):
    ws = wb.create_sheet("Full Results")
    ws.freeze_panes = "A3"

    thin = _select_n(rows, max_rows)
    title = f"{scenario_name}  —  {len(thin)}-Point Full Results"
    t = ws.cell(row=1, column=1, value=title)
    t.font = Font(name="Calibri", bold=True, size=12)
    ws.merge_cells(f"A1:{_col_ltr(len(_COL_HDRS))}1")

    sub = ws.cell(row=2, column=1,
                  value=f"Pig-centric data at {len(thin)} evenly-spaced positions "
                        f"(total sim steps: {len(rows):,}).")
    sub.font = _FNT_ITAL
    ws.merge_cells(f"A2:{_col_ltr(len(_COL_HDRS))}2")

    _write_table(ws, 3, thin, zebra=False)
    _set_widths(ws, _COL_WIDTHS)


# ---------------------------------------------------------------------------
# Sheet 5 — Elevation MOP Check (HGL-powered, inflection points as columns)
# ---------------------------------------------------------------------------

def _find_inflection_points(cfg: SimConfig) -> List[dict]:
    """Return elevation high and low points with minimum prominence filtering."""
    ep  = np.asarray(cfg.elevation_profile, dtype=float)
    mps = ep[:, 0]
    els = ep[:, 1]
    n   = len(mps)
    pts = []

    if n < 3:
        return pts

    d1 = np.diff(els)
    for i in range(1, len(d1)):
        if d1[i - 1] * d1[i] < 0:
            kind = "HP" if d1[i - 1] > 0 else "LP"
            pts.append({"mp": float(mps[i]), "elev": float(els[i]), "kind": kind})

    # Prominence filter: keep only if elev change to adjacent inflection >= 30 ft
    if len(pts) > 2:
        kept = []
        for j, p in enumerate(pts):
            prev_e = pts[j - 1]["elev"] if j > 0 else els[0]
            next_e = pts[j + 1]["elev"] if j < len(pts) - 1 else els[-1]
            prom = min(abs(p["elev"] - prev_e), abs(p["elev"] - next_e))
            if prom >= 30.0:
                kept.append(p)
        pts = kept

    # Cap at 30 columns — keep every other if more
    if len(pts) > 30:
        pts = pts[::max(1, len(pts) // 30)]

    return pts


def _sheet_mop_check(wb, results, cfg, scenario_name):
    ws = wb.create_sheet("Elevation MOP Check")
    ws.freeze_panes = "E3"

    title = f"{scenario_name}  —  Elevation & MOP Verification"
    t = ws.cell(row=1, column=1, value=title)
    t.font = Font(name="Calibri", bold=True, size=12)

    inflections = _find_inflection_points(cfg)

    if not inflections:
        ws.cell(row=2, column=1,
                value="No significant elevation inflection points found.").font = _FNT_ITAL
        ws.merge_cells("A1:D1")
        return

    ws.merge_cells(f"A1:{_col_ltr(4 + len(inflections))}1")

    sub = ws.cell(
        row=2, column=1,
        value=(f"Rows = pig position ({50}-point). "
               f"Columns = fixed inflection-point mileposts computed from the HGL. "
               f"RED = overpressure or slack; AMBER = within 100 psi of MOP or near 0; GREEN = safe.")
    )
    sub.font = _FNT_ITAL
    ws.merge_cells(f"A2:{_col_ltr(4 + len(inflections))}2")

    # Header row — pig info columns + one column per inflection point
    pig_hdrs = ["Pig MP", "Pig Elev (ft)", "Time (hr)", "Drive Psi"]
    for ci, h in enumerate(pig_hdrs, 1):
        _hdr(ws, 3, ci, h)

    elev_fn = _elev_interp(cfg)
    for j, pt in enumerate(inflections):
        mop_j  = _mop_at(pt["mp"], cfg)
        mop_lbl = f"MOP:{mop_j:.0f}" if mop_j else "no MOP"
        lbl = f"MP {pt['mp']:.1f}\n{pt['kind']}\n({mop_lbl})"
        c = _hdr(ws, 3, 5 + j, lbl, hex_bg=(_C_BLUE if pt["kind"] == "HP" else _C_NAVY))
        ws.column_dimensions[_col_ltr(5 + j)].width = 12

    # Select ~50 evenly-spaced steps
    steps = results.steps
    idxs  = np.round(np.linspace(0, len(steps) - 1, min(50, len(steps)))).astype(int)
    check_steps = [steps[i] for i in dict.fromkeys(idxs)]

    # Build rows
    for ri, step in enumerate(check_steps, start=4):
        zebra_fill = _fill(_C_ZEBRA_A if ri % 2 == 0 else _C_ZEBRA_B)

        # Pig info cells
        pig_vals = [step.pig_mp, elev_fn(step.pig_mp), step.t_hr, step.pig_face_psig]
        pig_fmts = ["0.000", "0.0", "0.000", "0.0"]
        for ci, (v, fmt) in enumerate(zip(pig_vals, pig_fmts), 1):
            c = ws.cell(row=ri, column=ci, value=v)
            c.fill = zebra_fill; c.font = _FNT_BODY
            c.number_format = fmt; c.border = _border()
            c.alignment = Alignment(horizontal="right")

        # Compute full HGL for this step
        try:
            hgl = compute_hgl(step, cfg)
        except Exception:
            hgl = None

        # Inflection point pressure cells
        for j, pt in enumerate(inflections):
            ci = 5 + j
            mop_j  = _mop_at(pt["mp"], cfg)
            elev_j = pt["elev"]

            if hgl is not None:
                p = float(np.interp(pt["mp"], hgl.mp, hgl.pressure_psig))
            else:
                p = float("nan")

            # Colour logic
            if math.isnan(p):
                cell_fill = zebra_fill
            elif mop_j and p > mop_j:
                cell_fill = _fill(_C_SLACK)      # overpressure — RED
            elif p < 0:
                cell_fill = _fill(_C_SLACK)      # slack/vacuum — RED
            elif mop_j and p > (mop_j - 100):
                cell_fill = _fill(_C_NEAR)       # approaching MOP — AMBER
            elif 0 <= p < 15:
                cell_fill = _fill(_C_AMBER)      # near-slack — AMBER
            else:
                cell_fill = _fill(_C_GREEN)      # safe — GREEN

            c = ws.cell(row=ri, column=ci,
                        value=round(p, 1) if not math.isnan(p) else None)
            c.fill = cell_fill; c.font = _FNT_BODY
            c.number_format = "0.0"; c.border = _border()
            c.alignment = Alignment(horizontal="right")

    _set_widths(ws, [9, 11, 9, 11])
    ws.row_dimensions[3].height = 40   # taller header to fit 3-line inflection labels


# ---------------------------------------------------------------------------
# Sheet 6 — Booster Roadmap (per-step booster state table)
# ---------------------------------------------------------------------------

def _sheet_booster_roadmap(wb, results, cfg, scenario_name, base_rows):
    ws = wb.create_sheet("Booster Roadmap")

    title = f"{scenario_name}  —  Booster / Spread Operations Roadmap"
    t = ws.cell(row=1, column=1, value=title)
    t.font = Font(name="Calibri", bold=True, size=12)

    if not cfg.booster_configs:
        ws.merge_cells("A1:J1")
        ws.cell(row=2, column=1,
                value="No booster stations in this scenario.").font = _FNT_ITAL
        return

    # Identify unique boosters from results
    booster_mps: List[float] = []
    booster_names: List[str] = []
    seen: set = set()
    for step in results.steps:
        for bs in (step.booster_states or []):
            key = (round(bs.get("mp", 0), 2), bs.get("name", ""))
            if key not in seen:
                seen.add(key)
                booster_mps.append(bs.get("mp", 0))
                booster_names.append(bs.get("name", f"MP {bs.get('mp','')}"))
    n_b = len(booster_mps)

    # Column headers
    base_h = ["Pig MP", "Pig Elev (ft)", "Time (hr)", "Drive Psi",
              "Exit Psi", "Inj SCFM", "Cum N2 (SCF)", "Speed (mph)"]
    booster_h: List[str] = []
    for nm in booster_names:
        booster_h += [f"{nm}\nRunning", f"{nm}\nFlow (SCFM)",
                      f"{nm}\nSuction (psi)", f"{nm}\nDischarge (psi)"]

    total_cols = len(base_h) + len(booster_h)
    ws.merge_cells(f"A1:{_col_ltr(total_cols)}1")

    sub = ws.cell(row=2, column=1,
                  value=f"50-point booster state table. "
                        f"Suction / discharge / flow at each pig-position checkpoint.")
    sub.font = _FNT_ITAL
    ws.merge_cells(f"A2:{_col_ltr(total_cols)}2")

    for ci, h in enumerate(base_h + booster_h, 1):
        c = _hdr(ws, 3, ci, h,
                 hex_bg=_C_NAVY if ci <= len(base_h) else _C_BLUE)
        ws.column_dimensions[_col_ltr(ci)].width = 12 if ci <= len(base_h) else 13
    ws.row_dimensions[3].height = 40

    elev_fn = _elev_interp(cfg)

    # 50 evenly-spaced steps
    steps = results.steps
    idxs  = np.round(np.linspace(0, len(steps) - 1, min(50, len(steps)))).astype(int)
    sample = [steps[i] for i in dict.fromkeys(idxs)]

    for ri, step in enumerate(sample, start=4):
        zebra = _fill(_C_ZEBRA_A if ri % 2 == 0 else _C_ZEBRA_B)

        bv = [step.pig_mp, elev_fn(step.pig_mp), step.t_hr, step.pig_face_psig,
              step.exit_psig, step.injection_scfm, step.total_scf, step.pig_speed_mph]
        bf = ["0.000", "0.0", "0.000", "0.0", "0.0", "0.0", "#,##0", "0.000"]
        for ci, (v, fmt) in enumerate(zip(bv, bf), 1):
            c = ws.cell(row=ri, column=ci, value=v)
            c.fill = zebra; c.font = _FNT_BODY
            c.number_format = fmt; c.border = _border()
            c.alignment = Alignment(horizontal="right")

        # Look up each booster's state
        bs_lookup = {round(bs.get("mp", 0), 2): bs
                     for bs in (step.booster_states or [])}
        for j, (bmp, bnm) in enumerate(zip(booster_mps, booster_names)):
            bs = bs_lookup.get(round(bmp, 2), {})
            ci_base = len(base_h) + j * 4 + 1
            running  = "YES" if bs.get("running") else "no"
            flow     = bs.get("flow_scfm", 0.0)
            suction  = bs.get("suction_psig", 0.0)
            discharge= bs.get("discharge_psig", 0.0)
            bvals = [running, flow, suction, discharge]
            bfmts = ["@", "0.0", "0.0", "0.0"]
            for k, (bv2, bf2) in enumerate(zip(bvals, bfmts)):
                c = ws.cell(row=ri, column=ci_base + k, value=bv2)
                c.fill = (zebra if running == "no" else _fill(_C_GREEN))
                c.font = _FNT_BODY; c.number_format = bf2
                c.border = _border()
                c.alignment = Alignment(horizontal="right" if k > 0 else "center")

    # Summary section
    next_r = 4 + len(sample) + 2
    _hdr(ws, next_r, 1, "BOOSTER THROUGHPUT SUMMARY", hex_bg=_C_BLUE)
    ws.merge_cells(f"A{next_r}:{_col_ltr(total_cols)}{next_r}")
    next_r += 1

    sum_hdrs = ["Station", "MP", "Active Steps", "Avg Flow (SCFM)",
                "Peak Flow (SCFM)", "Est. N2 Moved (SCF)", "Avg Suction (psi)", "Avg Discharge (psi)"]
    for ci, h in enumerate(sum_hdrs, 1):
        _hdr(ws, next_r, ci, h)
    next_r += 1

    total_hr = results.steps[-1].t_hr if results.steps else 1.0
    dt_avg   = total_hr / max(1, len(results.steps))
    tally: dict = defaultdict(lambda: {"n": 0, "fs": 0.0, "fm": 0.0,
                                        "ss": 0.0, "ds": 0.0, "name": ""})
    for step in results.steps:
        for bs in (step.booster_states or []):
            if bs.get("running"):
                k = round(bs.get("mp", 0), 2)
                tally[k]["n"]    += 1
                tally[k]["fs"]   += bs.get("flow_scfm", 0)
                tally[k]["fm"]    = max(tally[k]["fm"], bs.get("flow_scfm", 0))
                tally[k]["ss"]   += bs.get("suction_psig", 0)
                tally[k]["ds"]   += bs.get("discharge_psig", 0)
                tally[k]["name"]  = bs.get("name", "")

    fills2 = [_fill(_C_ZEBRA_A), _fill(_C_ZEBRA_B)]
    for ri2, (mp, d) in enumerate(sorted(tally.items())):
        n_active = max(1, d["n"])
        avg_f    = d["fs"] / n_active
        est_scf  = avg_f * n_active * dt_avg * 60
        row_fill = fills2[ri2 % 2]
        vals = [d["name"], mp, d["n"], avg_f, d["fm"], est_scf,
                d["ss"] / n_active, d["ds"] / n_active]
        fmts = ["@", "0.000", "0", "0.0", "0.0", "#,##0", "0.0", "0.0"]
        for ci, (v, fmt) in enumerate(zip(vals, fmts), 1):
            c = ws.cell(row=next_r, column=ci, value=v)
            c.fill = row_fill; c.font = _FNT_BODY
            c.number_format = fmt; c.border = _border()
        next_r += 1


# ---------------------------------------------------------------------------
# Sheet 7 — Raw Data (every SimStep field, all data)
# ---------------------------------------------------------------------------

def _sheet_raw_data(wb, results, cfg, scenario_name, max_rows=2000):
    ws = wb.create_sheet("Raw Data")
    ws.freeze_panes = "A3"

    title = f"{scenario_name}  —  Raw Simulation Data (All Model Outputs)"
    t = ws.cell(row=1, column=1, value=title)
    t.font = Font(name="Calibri", bold=True, size=12)

    steps = results.steps
    thin_idxs = np.round(np.linspace(0, len(steps) - 1,
                                      min(max_rows, len(steps)))).astype(int)
    thin = [steps[i] for i in dict.fromkeys(thin_idxs)]

    elev_fn   = _elev_interp(cfg)
    od, wt    = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
    area      = pipe_area_ft2(od, wt)
    sg        = cfg.fluid_sg

    # Identify all unique boosters
    b_keys: List[Tuple] = []
    b_seen: set = set()
    for step in steps:
        for bs in (step.booster_states or []):
            k = (round(bs.get("mp", 0), 2), bs.get("name", ""))
            if k not in b_seen:
                b_seen.add(k); b_keys.append(k)

    # Station keys
    st_keys: List[Tuple] = []
    st_seen: set = set()
    for step in steps:
        for sp in (step.station_pressures or []):
            k = (round(sp.get("mp", 0), 2), sp.get("name", ""))
            if k not in st_seen:
                st_seen.add(k); st_keys.append(k)

    # Build column definitions: (header, extractor_fn, format)
    cols = [
        ("Elapsed Time (hr)",         lambda s: s.t_hr,             "0.0000"),
        ("Pig MP",                    lambda s: s.pig_mp,            "0.000"),
        ("Pig Elevation (ft)",        lambda s: elev_fn(s.pig_mp),   "0.0"),
        ("Pig Speed (mph)",           lambda s: s.pig_speed_mph,     "0.000"),
        ("Drive Pressure (psi)",      lambda s: s.pig_face_psig,     "0.0"),
        ("Injection Pressure (psi)",  lambda s: s.injection_psig,    "0.0"),
        ("Injection Rate (SCFM)",     lambda s: s.injection_scfm,    "0.0"),
        ("Exit Pressure (psi)",       lambda s: s.exit_psig,         "0.0"),
        ("Exit MP",                   lambda s: s.exit_mp,           "0.000"),
        ("Cumulative N2 (SCF)",       lambda s: s.total_scf,         "#,##0"),
        ("Head Pressure (psi)",
         lambda s: (elev_fn(s.exit_mp) - elev_fn(s.pig_mp)) * sg * _PSIG_PER_FT, "0.0"),
        ("Friction Loss (psi)",
         lambda s: (s.pig_face_psig - s.exit_psig
                    - (elev_fn(s.exit_mp) - elev_fn(s.pig_mp)) * sg * _PSIG_PER_FT), "0.0"),
        ("Barrels / hr",              lambda s: s.pig_speed_mph * 5280.0 * area / _SCF_PER_BBL, "0.0"),
        ("Miles to Outlet",           lambda s: cfg.purge_end_mp - s.pig_mp, "0.000"),
        ("MOP at Pig (psi)",          lambda s: _mop_at(s.pig_mp, cfg) or "", "0.0"),
        ("MOP Violations",            lambda s: s.mop_violations,    "0"),
        ("MOP Warnings",              lambda s: s.mop_warnings,      "0"),
        ("Worst MOP Location (MP)",   lambda s: s.worst_mop_mp or "", "0.000"),
        ("Worst MOP Margin (psi)",    lambda s: s.worst_mop_margin or "", "0.0"),
        ("Slack Line Risk",           lambda s: "YES" if s.slack_line_risk else "no", "@"),
        ("Exit Pressure Min (psi)",   lambda s: s.exit_min_psig,     "0.0"),
        ("Endpoint Added (psi)",      lambda s: s.endpoint_added_psi, "0.0"),
        ("Over Max Speed",            lambda s: "YES" if s.overspeed else "no", "@"),
        ("Over Max Speed (mph)",      lambda s: s.overspeed_mph,     "0.00"),
        ("N2 Segments",               lambda s: len(s.segments or []),    "0"),
        ("Gas Profile Points",        lambda s: len(s.gas_pressure_profile or []), "0"),
        ("Injection Gas Psi",
         lambda s: (s.gas_pressure_profile[0][1] if s.gas_pressure_profile else 0), "0.0"),
        ("Gas Psi at Pig",
         lambda s: (s.gas_pressure_profile[-1][1] if s.gas_pressure_profile else 0), "0.0"),
    ]

    # Per-booster columns
    for bmp, bnm in b_keys:
        label = bnm or f"MP {bmp}"
        def _b_get(s, _bmp=bmp, _field=None):
            for bs in (s.booster_states or []):
                if abs(bs.get("mp", -999) - _bmp) < 0.01:
                    return bs
            return {}
        cols.append((f"{label} Running",
                      lambda s, _bmp=bmp: "YES" if any(
                          abs(bs.get("mp",-999)-_bmp)<0.01 and bs.get("running")
                          for bs in (s.booster_states or [])) else "no", "@"))
        cols.append((f"{label} Flow (SCFM)",
                      lambda s, _bmp=bmp: next(
                          (bs.get("flow_scfm", 0) for bs in (s.booster_states or [])
                           if abs(bs.get("mp",-999)-_bmp)<0.01), 0), "0.0"))
        cols.append((f"{label} Suction (psi)",
                      lambda s, _bmp=bmp: next(
                          (bs.get("suction_psig", 0) for bs in (s.booster_states or [])
                           if abs(bs.get("mp",-999)-_bmp)<0.01), 0), "0.0"))
        cols.append((f"{label} Discharge (psi)",
                      lambda s, _bmp=bmp: next(
                          (bs.get("discharge_psig", 0) for bs in (s.booster_states or [])
                           if abs(bs.get("mp",-999)-_bmp)<0.01), 0), "0.0"))

    # Per-station columns
    for smp, snm in st_keys:
        label = snm or f"MP {smp}"
        cols.append((f"{label} Discharge (psi)",
                      lambda s, _smp=smp: next(
                          (sp.get("discharge_psig", 0) for sp in (s.station_pressures or [])
                           if abs(sp.get("mp",-999)-_smp)<0.01), 0), "0.0"))
        cols.append((f"{label} Status",
                      lambda s, _smp=smp: next(
                          (sp.get("status", "") for sp in (s.station_pressures or [])
                           if abs(sp.get("mp",-999)-_smp)<0.01), ""), "@"))

    total_cols = len(cols)
    ws.merge_cells(f"A1:{_col_ltr(total_cols)}1")
    sub = ws.cell(row=2, column=1,
                  value=f"Complete model output at {len(thin):,} time steps "
                        f"({len(steps):,} total — downsampled to {max_rows:,} max). "
                        "All scalar and per-booster / per-station fields.")
    sub.font = _FNT_ITAL
    ws.merge_cells(f"A2:{_col_ltr(total_cols)}2")

    for ci, (hdr, _, _) in enumerate(cols, 1):
        _hdr(ws, 3, ci, hdr)
        ws.column_dimensions[_col_ltr(ci)].width = max(10, min(22, len(hdr) * 0.9 + 2))

    fills = [_fill(_C_ZEBRA_A), _fill(_C_ZEBRA_B)]
    for ri, step in enumerate(thin, start=4):
        zf = fills[ri % 2]
        for ci, (_, extfn, fmt) in enumerate(cols, 1):
            try:
                v = extfn(step)
            except Exception:
                v = None
            c = ws.cell(row=ri, column=ci, value=v)
            c.fill = zf; c.font = _FNT_BODY
            c.number_format = fmt; c.border = _border()
            c.alignment = Alignment(horizontal="right")
