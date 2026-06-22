"""
Rosen ILI Excel file parser.

File structure (verified on 2018 24ML Rosen ILI):
  - Sheet index 0
  - Row 9 (0-indexed): column headers
  - Row 10: units row (skip)
  - Row 11+: data

Auto-detected infrastructure via 'Additional description' keyword matching:
  - Pump station check valves: "station check valve"
  - Standalone check valves:   "check valve" (not station)
  - BPCV suction/discharge:    "backpressure"

Returns:
  ILIData dataclass with:
    - elevation profile (mp, elev_ft pairs)
    - per-joint MOP records
    - auto-detected pump stations, check valves, BPCV
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from ..engine.mop_check import MOPJoint
from ..engine.check_valve import CheckValve
from ..engine.pump_stations import PumpStationConfig
from ..engine.bpcv import BPCVConfig, BCPVDownstreamJoint, build_bpcv_from_ili
from ..engine.segment_model import PipeGeometry


# Column header aliases (lowercase)
_COL_MP        = 'calculated mile post [mi.]'
_COL_ELEV      = 'elevation [ft.]'
_COL_OD        = 'pipe diameter [in.]'
_COL_WT        = 'pipe nominal wall thickness [in.]'
_COL_MOP       = 'maximum operating pressure (mop) [psi]'
_COL_SITE      = 'site'
_COL_ADDL_DESC = 'additional description'
_COL_COMMENTS  = 'comments'
_COL_LAT       = 'latitude'
_COL_LON       = 'longitude'

# Fallback aliases for each key column (first match wins)
_ALIASES: dict[str, list[str]] = {
    'mp':        [_COL_MP, 'milepost', 'log distance', 'mp', 'calculated mile post'],
    'elev':      [_COL_ELEV, 'elevation [ft]', 'elevation', 'elev', 'altitude (ft)'],
    'od':        [_COL_OD, 'pipe diameter', 'od', 'outside diameter', 'diameter [in.]'],
    'wt':        [_COL_WT, 'wall thickness', 'wt', 'pipe nominal wall thickness'],
    'mop':       [_COL_MOP, 'maximum operating pressure (mop)', 'mop', 'maop', 'maximum operating pressure'],
    'site':      [_COL_SITE, 'site name', 'location'],
    'addl_desc': [_COL_ADDL_DESC, 'additional description', 'description', 'comments 2'],
    'comments':  [_COL_COMMENTS, 'comment', 'notes'],
    'lat':       [_COL_LAT, 'latitude', 'lat'],
    'lon':       [_COL_LON, 'longitude', 'lon', 'long'],
}


@dataclass
class PumpStationRecord:
    mp: float
    name: str
    site: str
    check_valve_mp: float     # mainline check valve milepost (may equal station mp)


@dataclass
class CheckValveRecord:
    mp: float
    name: str
    is_pump_station: bool


@dataclass
class BPCVRecord:
    suction_mp: float
    discharge_mp: float
    mp: float                  # midpoint used as nominal BPCV location
    elevation_ft: float
    site: str


@dataclass
class ILIData:
    """Parsed output from a Rosen ILI Excel file."""
    source_file: str

    # Elevation profile — sorted by mp
    elevation_profile: np.ndarray     # shape (N, 2): [[mp, elev_ft], ...]

    # Per-joint MOP data
    mop_joints: List[MOPJoint]

    # Pipe geometry changes (OD/WT transitions)
    pipe_geometry: PipeGeometry

    # Auto-detected infrastructure
    pump_station_records: List[PumpStationRecord] = field(default_factory=list)
    standalone_check_valves: List[CheckValveRecord] = field(default_factory=list)
    bpcv_record: Optional[BPCVRecord] = None

    # Raw DataFrame for advanced queries
    raw_df: Optional[pd.DataFrame] = None

    def check_valves(self) -> List[CheckValve]:
        """All check valves: pump station + standalone."""
        out = []
        for ps in self.pump_station_records:
            out.append(CheckValve(
                mp=ps.check_valve_mp,
                name=f"{ps.name} Check Valve",
                is_pump_station=True,
            ))
        for cv in self.standalone_check_valves:
            out.append(CheckValve(
                mp=cv.mp,
                name=cv.name,
                is_pump_station=False,
            ))
        return sorted(out, key=lambda v: v.mp)

    def pump_station_configs(self) -> List[PumpStationConfig]:
        return [
            PumpStationConfig(mp=ps.mp, name=ps.name)
            for ps in self.pump_station_records
        ]

    def bpcv_config(
        self,
        downstream_od_in: float = 22.0,
        downstream_wt_in: float = 0.313,
        downstream_sg: float = 0.85,
        downstream_viscosity_cst: float = 2.7,
        downstream_roughness_ft: float = 0.00015,
    ) -> Optional[BPCVConfig]:
        if self.bpcv_record is None:
            return None
        bpcv_mp = self.bpcv_record.mp
        joints_downstream = [
            {'mp': j.mp, 'mop_psig': j.mop_psig, 'elevation_ft': j.elevation_ft}
            for j in self.mop_joints
            if j.mp > bpcv_mp
        ]
        return build_bpcv_from_ili(
            bpcv_mp=bpcv_mp,
            bpcv_elevation_ft=self.bpcv_record.elevation_ft,
            ili_joints_downstream=joints_downstream,
            downstream_od_in=downstream_od_in,
            downstream_wt_in=downstream_wt_in,
            downstream_sg=downstream_sg,
            downstream_viscosity_cst=downstream_viscosity_cst,
            downstream_roughness_ft=downstream_roughness_ft,
        )


def _find_col(df_cols_lower: dict[str, str], key: str) -> Optional[str]:
    """Return the first matching column name from alias list for key."""
    for alias in _ALIASES.get(key, []):
        if alias in df_cols_lower:
            return df_cols_lower[alias]
    return None


def parse_ili(file_path: str, sheet_index: int = 0) -> ILIData:
    """
    Parse a Rosen-format ILI Excel file.

    Automatically finds the header row by searching for 'mile post' or 'milepost'
    in the first 20 rows, so it works even if the header isn't always row 9.
    """
    # --- Find header row ---
    probe = pd.read_excel(file_path, sheet_name=sheet_index, header=None, nrows=25)
    header_row = None
    for i, row in probe.iterrows():
        row_text = ' '.join(str(v).lower() for v in row if pd.notna(v))
        if 'mile post' in row_text or 'milepost' in row_text or 'calculated mile' in row_text:
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"Cannot find header row in {file_path}. "
            "Expected a row containing 'mile post' or 'milepost' in the first 25 rows."
        )

    # --- Read full file from header row, skip units row ---
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_index,
        header=header_row,
        skiprows=[header_row + 1],   # skip units row immediately after header
    )

    # Build lowercase → original column name map
    cols_lower = {str(c).strip().lower(): str(c) for c in df.columns}

    # --- Locate required columns ---
    mp_col    = _find_col(cols_lower, 'mp')
    elev_col  = _find_col(cols_lower, 'elev')
    od_col    = _find_col(cols_lower, 'od')
    wt_col    = _find_col(cols_lower, 'wt')
    mop_col   = _find_col(cols_lower, 'mop')
    site_col  = _find_col(cols_lower, 'site')
    desc_col  = _find_col(cols_lower, 'addl_desc')

    if mp_col is None:
        raise ValueError(f"Cannot find milepost column in {file_path}. Columns: {list(df.columns)}")
    if elev_col is None:
        raise ValueError(f"Cannot find elevation column in {file_path}. Columns: {list(df.columns)}")

    # --- Clean numeric columns ---
    df[mp_col]   = pd.to_numeric(df[mp_col],   errors='coerce')
    df[elev_col] = pd.to_numeric(df[elev_col], errors='coerce')
    if od_col:
        df[od_col] = pd.to_numeric(df[od_col], errors='coerce')
    if wt_col:
        df[wt_col] = pd.to_numeric(df[wt_col], errors='coerce')
    if mop_col:
        df[mop_col] = pd.to_numeric(df[mop_col], errors='coerce')

    df = df.dropna(subset=[mp_col]).copy()
    df = df.sort_values(mp_col).reset_index(drop=True)

    # --- Elevation profile ---
    valid_elev = df[elev_col].notna()
    elev_mps   = df.loc[valid_elev, mp_col].to_numpy(dtype=float)
    elev_vals  = df.loc[valid_elev, elev_col].to_numpy(dtype=float)
    elevation_profile = np.column_stack([elev_mps, elev_vals])

    # --- Per-joint MOP ---
    mop_joints: List[MOPJoint] = []
    if mop_col and od_col and wt_col:
        mop_mask = df[mop_col].notna() & df[od_col].notna() & df[wt_col].notna()
        for _, row in df.loc[mop_mask].iterrows():
            mop_joints.append(MOPJoint(
                mp=float(row[mp_col]),
                mop_psig=float(row[mop_col]),
                elevation_ft=float(row[elev_col]) if pd.notna(row[elev_col]) else 0.0,
                od_in=float(row[od_col]),
                wt_in=float(row[wt_col]),
            ))
    elif mop_col:
        mop_mask = df[mop_col].notna()
        for _, row in df.loc[mop_mask].iterrows():
            mop_joints.append(MOPJoint(
                mp=float(row[mp_col]),
                mop_psig=float(row[mop_col]),
                elevation_ft=float(row[elev_col]) if pd.notna(row[elev_col]) else 0.0,
            ))

    # --- Pipe geometry (variable OD/WT) ---
    pipe_geometry = _build_pipe_geometry(df, mp_col, od_col, wt_col)

    # --- Infrastructure detection ---
    pump_stations, standalone_cvs, bpcv_rec = _detect_infrastructure(
        df, mp_col, elev_col, site_col, desc_col
    )

    return ILIData(
        source_file=str(file_path),
        elevation_profile=elevation_profile,
        mop_joints=mop_joints,
        pipe_geometry=pipe_geometry,
        pump_station_records=pump_stations,
        standalone_check_valves=standalone_cvs,
        bpcv_record=bpcv_rec,
        raw_df=df,
    )


def _build_pipe_geometry(df: pd.DataFrame, mp_col: str,
                          od_col: Optional[str], wt_col: Optional[str]) -> PipeGeometry:
    """Build PipeGeometry from ILI OD/WT columns, detecting transition points."""
    geom = PipeGeometry()
    if od_col is None or wt_col is None:
        return geom

    mask = df[od_col].notna() & df[wt_col].notna()
    sub  = df.loc[mask, [mp_col, od_col, wt_col]].copy()
    if sub.empty:
        return geom

    # Find OD/WT change points
    sub['od_prev'] = sub[od_col].shift(1)
    sub['wt_prev'] = sub[wt_col].shift(1)
    changes = sub[(sub[od_col] != sub['od_prev']) | (sub[wt_col] != sub['wt_prev'])].copy()

    mps  = sub[mp_col].to_numpy(dtype=float)
    ods  = sub[od_col].to_numpy(dtype=float)
    wts  = sub[wt_col].to_numpy(dtype=float)

    # Build segment list from change points
    change_indices = [0] + list(np.where(
        (np.diff(ods) != 0) | (np.diff(wts) != 0)
    )[0] + 1) + [len(mps)]

    for k in range(len(change_indices) - 1):
        i_start = change_indices[k]
        i_end   = change_indices[k + 1] - 1
        if i_start > i_end:
            continue
        mp_start = float(mps[i_start])
        mp_end   = float(mps[i_end])
        od       = float(ods[i_start])
        wt       = float(wts[i_start])
        geom.segments.append((mp_start, mp_end, od, wt))

    return geom


def _detect_infrastructure(
    df: pd.DataFrame,
    mp_col: str,
    elev_col: str,
    site_col: Optional[str],
    desc_col: Optional[str],
) -> tuple[List[PumpStationRecord], List[CheckValveRecord], Optional[BPCVRecord]]:
    """
    Detect pump stations, standalone check valves, and BPCV from ILI Additional description.
    """
    pump_stations: List[PumpStationRecord] = []
    standalone_cvs: List[CheckValveRecord] = []
    bpcv_suction_rows = []
    bpcv_discharge_rows = []

    if desc_col is None:
        return pump_stations, standalone_cvs, None

    for _, row in df.iterrows():
        desc = str(row.get(desc_col, '') or '').lower().strip()
        mp   = float(row[mp_col]) if pd.notna(row[mp_col]) else None
        if mp is None:
            continue
        elev = float(row[elev_col]) if pd.notna(row.get(elev_col)) else 0.0
        site = str(row.get(site_col, '') or '').strip() if site_col else ''

        if 'backpressure' in desc or 'back pressure' in desc:
            if 'suction' in desc:
                bpcv_suction_rows.append({'mp': mp, 'elev': elev, 'site': site})
            elif 'discharge' in desc:
                bpcv_discharge_rows.append({'mp': mp, 'elev': elev, 'site': site})
            else:
                bpcv_suction_rows.append({'mp': mp, 'elev': elev, 'site': site})

        elif 'station check valve' in desc or ('pump station' in desc and 'check' in desc):
            ps_name = site if site else f"Station MP{mp:.2f}"
            pump_stations.append(PumpStationRecord(
                mp=mp,
                name=ps_name,
                site=site,
                check_valve_mp=mp,
            ))

        elif 'check valve' in desc:
            cv_name = f"Check Valve MP{mp:.2f}"
            standalone_cvs.append(CheckValveRecord(
                mp=mp,
                name=cv_name,
                is_pump_station=False,
            ))

    # Deduplicate pump stations (keep one per unique site/mp cluster)
    pump_stations = _deduplicate_by_proximity(pump_stations, tol_mi=0.5)
    standalone_cvs = _deduplicate_cv_by_proximity(standalone_cvs, tol_mi=0.1)

    # Build BPCV record
    bpcv_rec: Optional[BPCVRecord] = None
    if bpcv_suction_rows:
        row_s = bpcv_suction_rows[0]
        row_d = bpcv_discharge_rows[0] if bpcv_discharge_rows else row_s
        nominal_mp   = 0.5 * (float(row_s['mp']) + float(row_d['mp']))
        nominal_elev = float(row_s['elev'])
        bpcv_rec = BPCVRecord(
            suction_mp=float(row_s['mp']),
            discharge_mp=float(row_d['mp']),
            mp=nominal_mp,
            elevation_ft=nominal_elev,
            site=str(row_s.get('site', '')),
        )

    return pump_stations, standalone_cvs, bpcv_rec


def _deduplicate_by_proximity(
    records: List[PumpStationRecord], tol_mi: float
) -> List[PumpStationRecord]:
    """Keep one pump station record per milepost cluster."""
    if not records:
        return records
    records = sorted(records, key=lambda r: r.mp)
    out = [records[0]]
    for r in records[1:]:
        if abs(r.mp - out[-1].mp) > tol_mi:
            out.append(r)
    return out


def _deduplicate_cv_by_proximity(
    records: List[CheckValveRecord], tol_mi: float
) -> List[CheckValveRecord]:
    if not records:
        return records
    records = sorted(records, key=lambda r: r.mp)
    out = [records[0]]
    for r in records[1:]:
        if abs(r.mp - out[-1].mp) > tol_mi:
            out.append(r)
    return out
