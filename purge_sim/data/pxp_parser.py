"""
Point-by-point (PxP) MOP and operating pressure sheet parser.

This is the operator's pressure sheet format (verified on the P66 GL-09 Buffalo to
Billings 12" PxP, 2023), as opposed to a Rosen ILI tally:
  - Data sheet named 'PxPData' (other sheets are a guide)
  - Row 1: system banner (system name, PODS date, light/heavy specified gravity)
  - Rows 2-3: two-row long headers ('Dist. from Origin (miles)', 'MOP (psig)', ...)
  - Row 4: short field names ('Dist', 'Elevation', 'PipeOD', 'PipeWT', 'MOP', ...)
  - Data below, one row per point (coordinate points and PODS features interleaved,
    roughly every 30 ft), ordered by distance from the line's origin

Mapped onto the same ILIData the Rosen parser returns, so everything downstream
(importers, the desktop app) treats both the same way:
  - milepost      = 'Dist. from Origin (miles)', which runs in the flow direction.
                    ('MilePost by Station' usually runs backwards and is not used.)
  - elevation     = 'Elevation (ft)'
  - OD / WT       = 'OD (in)' / 'WT (in)'
  - per-point MOP = 'MOP (psig)', the sheet's key MOP column (column AA). It is the
                    established MOP at each point, which can sit below the pipe's own
                    'MOP Limit' where pressure control governs; the lower one is used
                    because that's the one the operator holds the line to.
  - pump stations = feature 'PMP' rows (or a value in the 'Pump Station' column),
                    named from 'Feature Name'
  - check valves  = valve type 'CHECK', or a feature name saying check valve / CV;
                    one within 0.25 mi downstream of a station is that station's check valve
  - BPCV          = a feature name mentioning back pressure / BPCV (none on GL-09)
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..engine.mop_check import MOPJoint
from .elevation import valid_coords
from .xlsx_reader import read_sheets
from .ili_parser import (BPCVRecord, CheckValveRecord, ILIData, PumpStationRecord,
                         _build_pipe_geometry, _semantic_thin_indices)

# Short field name (header row 4) -> fallback long header text (rows 2-3), lowercase.
# The short names are stable across PxP sheets; the long ones are a fallback.
_FIELDS: Dict[str, Tuple[str, str]] = {
    "dist":      ("dist", "dist. from origin (miles)"),
    "milepost":  ("milepost", "milepost by station (miles)"),
    "elev":      ("elevation", "elevation (ft)"),
    "od":        ("pipeod", "od (in)"),
    "wt":        ("pipewt", "wt (in)"),
    "mop":       ("mop", "mop (psig)"),
    "mop_limit": ("moplimit", "mop limit\n(psig)"),
    "pop":       ("pop", "pop (psig)"),
    "desc":      ("descriptions", "feature/\ndescriptions"),
    "feature":   ("feature", "feature name"),
    "valve":     ("vlavetype", "type"),     # sic: the sheet spells it 'VlaveType'
    "pump":      ("pumploc", "pump station"),
    "lat":       ("y", "y (deg)"),
    "lon":       ("x", "x (deg)"),
}

# How far downstream of a station its check valve may sit and still be "its" valve.
_STATION_CV_TOL_MI = 0.25
# Infrastructure this close to either end of the file is the launch / receipt site.
_END_TOL_MI = 0.1

_CHECK_RE = re.compile(r"\bcheck\s*valve\b|\bcv\b", re.I)
_BPCV_RE = re.compile(r"back\s*pressure|\bbpcv\b", re.I)


def _norm(v) -> str:
    return str(v).strip().lower() if v is not None else ""


def find_pxp_header(rows: List[tuple]) -> Optional[int]:
    """Index of the short field-name row ('Dist', 'MOP', 'PipeOD'...) in the first rows of a
    sheet, or of the long header row if there's no short one. None if this isn't a PxP sheet."""
    for i, row in enumerate(rows):
        cells = {_norm(v) for v in row}
        if {"dist", "mop", "pipeod", "elevation"} <= cells:
            return i
    for i, row in enumerate(rows):
        cells = {_norm(v) for v in row}
        if "dist. from origin (miles)" in cells and "mop (psig)" in cells:
            return i
    return None


def _column_map(rows: List[tuple], header_idx: int) -> Dict[str, int]:
    """Field key -> column index. Short names come from the header row; anything missing
    is looked up by its long header text in the rows above it."""
    out: Dict[str, int] = {}
    short = [_norm(v) for v in rows[header_idx]]
    for key, (s, _) in _FIELDS.items():
        if s in short:
            out[key] = short.index(s)          # first match: 'Dist' appears twice
    for key, (_, long_name) in _FIELDS.items():
        if key in out:
            continue
        for r in rows[max(0, header_idx - 3):header_idx + 1]:
            cells = [_norm(v) for v in r]
            if long_name in cells:
                out[key] = cells.index(long_name)
                break
    return out


def _banner(rows: List[tuple]) -> Dict[str, object]:
    """Key/value pairs from the first row: 'SYSTEM NAME', 'Specified Gravity - Light'..."""
    info: Dict[str, object] = {}
    if not rows:
        return info
    row = list(rows[0])
    for i, v in enumerate(row):
        label = _norm(v)
        if label in ("system name", "specified gravity - light", "specified gravity - heavy", "pods date"):
            val = next((x for x in row[i + 1:i + 4] if x is not None), None)
            info[label] = val
    return info


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _read_sheet(file_path: str) -> Tuple[str, List[tuple], int, Dict[str, int]]:
    """(sheet name, rows, header row index, column map) for the PxP data sheet. Only the
    columns the parser uses are read: the sheet has ~90 and most are derived results."""
    heads = read_sheets(file_path, max_rows=30)
    names = sorted(heads, key=lambda n: 0 if "pxp" in n.lower() and "data" in n.lower() else 1)
    for name in names:
        head = [tuple(r) for r in heads[name]]
        h = find_pxp_header(head)
        if h is None:
            continue
        cols = _column_map(head, h)
        rows = read_sheets(file_path, sheet=name, max_col=max(cols.values()) + 1)[name]
        return name, [tuple(r) for r in rows], h, cols
    raise ValueError(f"{os.path.basename(file_path)} doesn't look like a point-by-point (PxP) sheet: "
                     "no sheet has 'Dist', 'Elevation', 'PipeOD' and 'MOP' columns")


def parse_pxp(file_path: str) -> ILIData:
    sheet, rows, h, cols = _read_sheet(file_path)
    for need in ("dist", "elev"):
        if need not in cols:
            raise ValueError(f"PxP sheet '{sheet}' has no '{_FIELDS[need][1]}' column")

    def col(key, r):
        i = cols.get(key)
        return r[i] if i is not None and i < len(r) else None

    recs = []
    for r in rows[h + 1:]:
        d = _num(col("dist", r))
        if not np.isfinite(d):
            continue
        recs.append({
            "mp": d,
            "elev": _num(col("elev", r)),
            "od": _num(col("od", r)),
            "wt": _num(col("wt", r)),
            "mop": _num(col("mop", r)),
            "mop_limit": _num(col("mop_limit", r)),
            "desc": str(col("desc", r) or "").strip(),
            # 'COORD_PT' is a plain survey point, not a feature
            "feature": "" if _norm(col("feature", r)) == "coord_pt" else str(col("feature", r) or "").strip(),
            "valve": str(col("valve", r) or "").strip(),
            "pump": col("pump", r),
            "lat": _num(col("lat", r)),
            "lon": _num(col("lon", r)),
        })
    if len(recs) < 2:
        raise ValueError(f"PxP sheet '{sheet}' has fewer than 2 data rows")

    df = pd.DataFrame(recs)
    # stable sort keeps the sheet's order for points sharing a distance
    df = df.sort_values("mp", kind="mergesort").reset_index(drop=True)

    # --- Elevation profile: one point per distance ---
    ev = df[df["elev"].notna()].drop_duplicates("mp", keep="last")
    elevation_profile = np.column_stack([ev["mp"].to_numpy(float), ev["elev"].to_numpy(float)])

    # --- Per-point MOP, thinned the same way as an ILI tally ---
    mop_joints: List[MOPJoint] = []
    if "mop" in cols:
        sub = df[df["mop"].notna() & df["od"].notna() & df["wt"].notna() & df["elev"].notna()]
        sub = sub.reset_index(drop=True)
        for i in _semantic_thin_indices(sub, "mop", "elev", None, "desc", "feature", mop_step_psig=5.0):
            row = sub.iloc[i]
            mop_joints.append(MOPJoint(mp=float(row["mp"]), mop_psig=float(row["mop"]),
                                       elevation_ft=float(row["elev"]), od_in=float(row["od"]),
                                       wt_in=float(row["wt"])))

    pipe_geometry = _build_pipe_geometry(df, "mp", "od", "wt")
    stations, cvs, bpcv, notes = _detect_infrastructure(df)

    info = _banner(rows)
    lat, lon = df["lat"].to_numpy(float), df["lon"].to_numpy(float)
    ok = valid_coords(lat, lon)
    sg_l, sg_h = _num(info.get("specified gravity - light")), _num(info.get("specified gravity - heavy"))
    data = ILIData(source_file=str(file_path), elevation_profile=elevation_profile,
                   mop_joints=mop_joints, pipe_geometry=pipe_geometry,
                   pump_station_records=stations, standalone_check_valves=cvs,
                   bpcv_record=bpcv, raw_df=df, format="PxP",
                   system_name=str(info.get("system name") or "").strip(),
                   sg_light=sg_l if np.isfinite(sg_l) else None,
                   sg_heavy=sg_h if np.isfinite(sg_h) else None,
                   route_latlon=[[float(a), float(b)] for a, b in zip(lat[ok], lon[ok])],
                   notes=notes)
    return data


def _detect_infrastructure(df: pd.DataFrame):
    start, end = float(df["mp"].iloc[0]), float(df["mp"].iloc[-1])
    notes: List[str] = []
    stations: List[PumpStationRecord] = []
    checks: List[Tuple[float, str]] = []
    bpcv_rows = []
    for r in df.itertuples(index=False):
        name = r.feature or r.desc
        pump_val = r.pump
        is_pump = r.desc.upper() == "PMP" or (
            pump_val is not None and str(pump_val).strip() not in ("", "#N/A") and np.isfinite(_num(pump_val)))
        if is_pump:
            stations.append(PumpStationRecord(mp=float(r.mp), name=_title(name) or f"Station MP{r.mp:.2f}",
                                              site=name, check_valve_mp=float(r.mp)))
        elif r.valve.upper() == "CHECK" or r.desc.upper() == "CHECK" or _CHECK_RE.search(r.feature):
            checks.append((float(r.mp), r.feature))
        if _BPCV_RE.search(r.feature):
            bpcv_rows.append(r)

    # one record per station / valve (a feature often shows up on 2-3 nearby rows)
    stations = _dedupe(stations, 0.5)
    checks = _dedupe_pairs(checks, 0.1)

    # the station at the launch end is where the purge starts and the one at the far end
    # is the receipt; neither pushes liquid ahead of the pig
    kept: List[PumpStationRecord] = []
    for s in stations:
        if s.mp - start <= _END_TOL_MI or end - s.mp <= _END_TOL_MI:
            notes.append(f"{s.name} (MP {s.mp:.2f}) is at the end of the data, so it's treated as the "
                         "launch/receipt site rather than a pump station.")
        else:
            kept.append(s)

    standalone: List[CheckValveRecord] = []
    for mp, name in checks:
        if mp - start <= _END_TOL_MI or end - mp <= _END_TOL_MI:
            continue
        owner = next((s for s in kept if 0.0 <= mp - s.mp <= _STATION_CV_TOL_MI
                      or 0.0 <= s.mp - mp <= 0.05), None)
        if owner is not None:
            owner.check_valve_mp = mp
        else:
            standalone.append(CheckValveRecord(mp=mp, name=_cv_name(name, mp), is_pump_station=False))

    bpcv = None
    if bpcv_rows:
        r = bpcv_rows[0]
        bpcv = BPCVRecord(suction_mp=float(r.mp), discharge_mp=float(r.mp), mp=float(r.mp),
                          elevation_ft=float(r.elev) if np.isfinite(r.elev) else 0.0, site=r.feature)
    return kept, standalone, bpcv, notes


def _title(s: str) -> str:
    return " ".join(w.capitalize() if w.isalpha() else w for w in str(s).split())


def _cv_name(feature: str, mp: float) -> str:
    f = re.sub(r"^AGR\s*\|\s*", "", feature or "").strip()
    return f"{f} (check valve)" if f and not _CHECK_RE.search(f) else (f or f"Check Valve MP{mp:.2f}")


def _dedupe(records: List[PumpStationRecord], tol: float) -> List[PumpStationRecord]:
    out: List[PumpStationRecord] = []
    for r in sorted(records, key=lambda r: r.mp):
        if not out or r.mp - out[-1].mp > tol:
            out.append(r)
    return out


def _dedupe_pairs(items: List[Tuple[float, str]], tol: float) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for mp, name in sorted(items):
        if not out or mp - out[-1][0] > tol:
            out.append((mp, name))
    return out
