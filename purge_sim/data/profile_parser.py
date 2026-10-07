"""
Pipeline profile parsers for non-ILI data sources.

Supports four input paths:
  1. KMZ / KML (Google Earth format), with or without altitudes
  2. KMZ → GPSVisualizer TXT export (tab-delimited with lat/lon/altitude columns)
  3. Client Excel / CSV with lat/lon/elevation or milepost/elevation
  4. Simple milepost+elevation text file

All parsers return a ProfileData with:
  - mileposts (miles, relative to start)
  - elevations (feet)
  - lat/lon arrays (if available)
  - elevation_status: 'full', 'partial' or 'none'

A file without elevations (most pipeline KMZs are 2D) is NOT given a flat 0 ft
profile: its missing elevations are NaN and elevation_status says so, so the
caller can look them up (purge_sim/data/elevation.py) or refuse the import.

Ported and cleaned from Purge_Modeling_Program_v29.py.
"""

from __future__ import annotations

import math
import os
import zipfile
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import pandas as pd

try:
    from lxml import etree as ET
    _LXML_AVAILABLE = True
except ImportError:
    import xml.etree.ElementTree as ET
    _LXML_AVAILABLE = False

from ..engine.physics import haversine_miles
from ..engine.constants import COL_LAT, COL_LON, COL_ELEV, COL_MP


@dataclass
class ProfileData:
    """Parsed elevation profile — all units converted to miles and feet."""
    mileposts: np.ndarray       # relative to start, in miles
    elevations_ft: np.ndarray
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    source: str = ""
    elevation_status: str = "full"   # 'full' | 'partial' (some NaN) | 'none' (all NaN)

    @property
    def needs_elevation(self) -> bool:
        return self.elevation_status != "full"

    @property
    def start_mp(self) -> float:
        return float(self.mileposts[0])

    @property
    def end_mp(self) -> float:
        return float(self.mileposts[-1])

    @property
    def total_length_mi(self) -> float:
        return float(self.mileposts[-1] - self.mileposts[0])

    def elevation_profile_array(self) -> np.ndarray:
        """Return [[mp, elev_ft], ...] array for SimConfig.elevation_profile."""
        return np.column_stack([self.mileposts, self.elevations_ft])


# ---------------------------------------------------------------------------
# KML/KMZ helpers
# ---------------------------------------------------------------------------

def _lname(tag) -> str:
    if isinstance(tag, str) and '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def _parse_kml_coordinates(coord_text: str) -> List[tuple]:
    """Parse 'lon,lat,alt' tokens from KML <coordinates>."""
    pts = []
    for token in (coord_text or '').strip().split():
        try:
            parts = token.split(',')
            if len(parts) < 2:
                continue
            lon  = float(parts[0])
            lat  = float(parts[1])
            elev = float(parts[2]) if len(parts) > 2 and parts[2] else None
            pts.append((lat, lon, elev))
        except (ValueError, IndexError):
            continue
    return pts


def _parse_gx_coord(text: str) -> Optional[tuple]:
    """Parse 'lon lat alt' from gx:coord."""
    try:
        parts = (text or '').strip().split()
        if len(parts) < 2:
            return None
        return (float(parts[1]), float(parts[0]), float(parts[2]) if len(parts) > 2 else None)
    except ValueError:
        return None


def _extract_coords_from_kml_root(root) -> List[tuple]:
    """Extract (lat, lon, elev_m) from KML root — longest LineString wins."""
    candidates = []

    for ls in root.iter():
        if _lname(ls.tag) == 'LineString':
            for ch in ls.iter():
                if _lname(ch.tag) == 'coordinates' and getattr(ch, 'text', None):
                    pts = _parse_kml_coordinates(ch.text)
                    if pts:
                        candidates.append(pts)

    if candidates:
        return max(candidates, key=len)

    # Fall back to gx:Track
    for tr in root.iter():
        if _lname(tr.tag) == 'Track':
            pts = []
            for ch in tr.iter():
                if _lname(ch.tag) == 'coord' and getattr(ch, 'text', None):
                    p = _parse_gx_coord(ch.text)
                    if p:
                        pts.append(p)
            if pts:
                candidates.append(pts)

    if candidates:
        return max(candidates, key=len)

    # Fall back to any <coordinates> not in a Polygon
    for ch in root.iter():
        if _lname(ch.tag) == 'coordinates' and getattr(ch, 'text', None):
            pts = _parse_kml_coordinates(ch.text)
            if pts:
                candidates.append(pts)

    return max(candidates, key=len) if candidates else []


def elevation_status(elevs: np.ndarray) -> str:
    """'none' when nothing usable is there (no values, or all exactly 0: KML's clamp-to-ground
    placeholder), 'partial' when some points are missing, else 'full'."""
    finite = np.isfinite(elevs)
    if not finite.any() or not np.any(elevs[finite] != 0.0):
        return "none"
    return "full" if finite.all() else "partial"


def _build_profile_from_lat_lon_elev(
    pts: List[tuple],   # (lat, lon, elev_m)
    elev_in_meters: bool = True,
) -> ProfileData:
    """Build ProfileData from lat/lon/elev points."""
    lats  = np.array([p[0] for p in pts], dtype=float)
    lons  = np.array([p[1] for p in pts], dtype=float)
    elevs = np.array([np.nan if p[2] is None else p[2] for p in pts], dtype=float)
    status = elevation_status(elevs)
    if status == "none":
        elevs = np.full(len(pts), np.nan)

    if elev_in_meters:
        elevs = elevs * 3.28084  # m → ft

    # Compute mileposts from lat/lon
    mps = [0.0]
    for i in range(1, len(lats)):
        d = haversine_miles(lats[i-1], lons[i-1], lats[i], lons[i])
        mps.append(mps[-1] + d)
    mps = np.array(mps, dtype=float)

    return ProfileData(
        mileposts=mps,
        elevations_ft=elevs,
        lat=lats,
        lon=lons,
        elevation_status=status,
    )


def parse_kmz(file_path: str) -> ProfileData:
    """
    Parse a KMZ file (zipped KML). Extracts the pipeline centerline and elevation profile.
    Elevation data is in meters in KML (converted to feet).
    """
    with zipfile.ZipFile(file_path, 'r') as z:
        kml_names = [n for n in z.namelist() if n.lower().endswith('.kml')]
        if not kml_names:
            raise ValueError(f"No KML file found inside KMZ: {file_path}")
        # Prefer doc.kml or the largest KML
        main_kml = next((n for n in kml_names if 'doc' in n.lower()), kml_names[0])
        kml_data = z.read(main_kml)

    if _LXML_AVAILABLE:
        parser = ET.XMLParser(resolve_entities=False, no_network=True, recover=True)
        root = ET.fromstring(kml_data, parser)
    else:
        root = ET.fromstring(kml_data.decode('utf-8', errors='replace'))

    pts = _extract_coords_from_kml_root(root)
    if not pts:
        raise ValueError(f"No pipeline coordinates found in KMZ: {file_path}")

    profile = _build_profile_from_lat_lon_elev(pts, elev_in_meters=True)
    profile.source = f"KMZ:{os.path.basename(file_path)}"
    return profile


def parse_kml(file_path: str) -> ProfileData:
    """Parse a KML file directly."""
    with open(file_path, 'rb') as f:
        kml_data = f.read()

    if _LXML_AVAILABLE:
        parser = ET.XMLParser(resolve_entities=False, no_network=True, recover=True)
        root = ET.fromstring(kml_data, parser)
    else:
        root = ET.fromstring(kml_data.decode('utf-8', errors='replace'))

    pts = _extract_coords_from_kml_root(root)
    if not pts:
        raise ValueError(f"No pipeline coordinates found in KML: {file_path}")

    profile = _build_profile_from_lat_lon_elev(pts, elev_in_meters=True)
    profile.source = f"KML:{os.path.basename(file_path)}"
    return profile


# ---------------------------------------------------------------------------
# Flexible TXT/CSV/Excel parser (GPSVisualizer output or client spreadsheet)
# ---------------------------------------------------------------------------

def _detect_col(cols_lower: dict[str, str], aliases: List[str]) -> Optional[str]:
    for a in aliases:
        if a in cols_lower:
            return cols_lower[a]
    return None


def parse_txt_csv(file_path: str, elevation_units: str = 'auto') -> ProfileData:
    """
    Parse a TXT or CSV file with flexible column detection.
    Handles GPSVisualizer output (lat/lon/altitude) and simple milepost+elevation files.

    elevation_units: 'Feet', 'Meters', or 'auto' (auto-detect from column name)
    """
    df = None
    for sep in [None, '\t', ',', ';', '|']:
        try:
            kw = {'sep': sep, 'engine': 'python', 'on_bad_lines': 'skip'} if sep is None else \
                 {'sep': sep, 'on_bad_lines': 'skip'}
            df = pd.read_csv(file_path, **kw)
            if df is not None and len(df.columns) >= 2:
                break
        except Exception:
            continue

    if df is None or len(df.columns) < 2:
        raise ValueError(f"Could not parse TXT/CSV: {file_path}")

    cols_lower = {str(c).strip().lower(): str(c) for c in df.columns}
    lat_col  = _detect_col(cols_lower, [a.lower() for a in COL_LAT])
    lon_col  = _detect_col(cols_lower, [a.lower() for a in COL_LON])
    elev_col = _detect_col(cols_lower, [a.lower() for a in COL_ELEV])
    mp_col   = _detect_col(cols_lower, [a.lower() for a in COL_MP])

    has_gps  = lat_col is not None and lon_col is not None
    has_elev = elev_col is not None
    has_mp   = mp_col is not None

    if not has_gps and not has_mp:
        raise ValueError(f"No lat/lon or milepost column in {file_path}. Columns: {list(df.columns)}")
    if not has_elev and not has_gps:
        raise ValueError(f"No elevation column in {file_path}. Columns: {list(df.columns)}")

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
        raise ValueError(f"Fewer than 2 valid rows in {file_path}")

    # Compute mileposts
    if has_gps:
        lats = df[lat_col].to_numpy(dtype=float)
        lons = df[lon_col].to_numpy(dtype=float)
        mps  = [0.0]
        for i in range(1, len(lats)):
            mps.append(mps[-1] + haversine_miles(lats[i-1], lons[i-1], lats[i], lons[i]))
        mps_arr = np.array(mps, dtype=float)
    else:
        mps_raw = df[mp_col].to_numpy(dtype=float)
        rng = float(np.nanmax(mps_raw) - np.nanmin(mps_raw))
        if rng > 500:          # likely feet
            mps_arr = (mps_raw - mps_raw[0]) / 5280.0
        elif 50 < rng < 500:   # likely km
            mps_arr = (mps_raw - mps_raw[0]) * 0.621371
        else:                  # already miles
            mps_arr = mps_raw - mps_raw[0]

    # Elevations
    if has_elev:
        elevs = df[elev_col].to_numpy(dtype=float)
        valid = np.isfinite(elevs)
        if valid.sum() >= 2:
            elevs = np.where(valid, elevs,
                             np.interp(np.arange(len(elevs)), np.where(valid)[0], elevs[valid]))

        col_lower = elev_col.lower()
        units_in_name = '(m)' in col_lower or '_m' in col_lower or col_lower.endswith(' m')
        if units_in_name or elevation_units == 'Meters':
            elevs = elevs * 3.28084
    else:
        elevs = np.full(len(mps_arr), np.nan)   # lat/lon only: elevation still to be looked up

    n = min(len(mps_arr), len(elevs))
    profile = ProfileData(
        mileposts=mps_arr[:n],
        elevations_ft=elevs[:n],
        lat=df[lat_col].to_numpy(dtype=float)[:n] if has_gps else None,
        lon=df[lon_col].to_numpy(dtype=float)[:n] if has_gps else None,
        source=f"TXT:{os.path.basename(file_path)}",
        elevation_status=elevation_status(elevs[:n]),
    )
    return profile


def parse_excel_profile(file_path: str, sheet_index: int = 0,
                         elevation_units: str = 'auto') -> ProfileData:
    """
    Parse a client-supplied Excel with elevation/milepost data (not a Rosen ILI file).
    Tries to auto-detect headers in the first 10 rows.
    """
    # Try to find header row
    probe = pd.read_excel(file_path, sheet_name=sheet_index, header=None, nrows=15)
    header_row = 0
    for i, row in probe.iterrows():
        row_text = ' '.join(str(v).lower() for v in row if pd.notna(v))
        if any(k in row_text for k in ('elevation', 'milepost', 'latitude', 'altitude', 'elev')):
            header_row = i
            break

    df = pd.read_excel(file_path, sheet_name=sheet_index, header=header_row)
    cols_lower = {str(c).strip().lower(): str(c) for c in df.columns}

    lat_col  = _detect_col(cols_lower, [a.lower() for a in COL_LAT])
    lon_col  = _detect_col(cols_lower, [a.lower() for a in COL_LON])
    elev_col = _detect_col(cols_lower, [a.lower() for a in COL_ELEV])
    mp_col   = _detect_col(cols_lower, [a.lower() for a in COL_MP])

    has_gps  = lat_col is not None and lon_col is not None
    has_elev = elev_col is not None
    has_mp   = mp_col is not None

    if not has_gps and not has_mp:
        raise ValueError(f"No usable columns found in {file_path}. Columns: {list(df.columns)}")

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
        raise ValueError(f"Fewer than 2 valid rows in {file_path}")

    if has_gps:
        lats = df[lat_col].to_numpy(dtype=float)
        lons = df[lon_col].to_numpy(dtype=float)
        mps = [0.0]
        for i in range(1, len(lats)):
            mps.append(mps[-1] + haversine_miles(lats[i-1], lons[i-1], lats[i], lons[i]))
        mps_arr = np.array(mps, dtype=float)
    else:
        mps_raw = df[mp_col].to_numpy(dtype=float)
        rng = float(np.nanmax(mps_raw) - np.nanmin(mps_raw))
        if rng > 500:
            mps_arr = (mps_raw - mps_raw[0]) / 5280.0
        elif 50 < rng < 500:
            mps_arr = (mps_raw - mps_raw[0]) * 0.621371
        else:
            mps_arr = mps_raw - mps_raw[0]

    if has_elev:
        elevs = df[elev_col].to_numpy(dtype=float)
        valid = np.isfinite(elevs)
        if valid.sum() >= 2:
            elevs = np.where(valid, elevs,
                             np.interp(np.arange(len(elevs)), np.where(valid)[0], elevs[valid]))
        col_lower = elev_col.lower()
        if '(m)' in col_lower or elevation_units == 'Meters':
            elevs = elevs * 3.28084
    else:
        elevs = np.full(len(mps_arr), np.nan)   # lat/lon only: elevation still to be looked up

    n = min(len(mps_arr), len(elevs))
    return ProfileData(
        mileposts=mps_arr[:n],
        elevations_ft=elevs[:n],
        lat=df[lat_col].to_numpy(dtype=float)[:n] if has_gps else None,
        lon=df[lon_col].to_numpy(dtype=float)[:n] if has_gps else None,
        source=f"Excel:{os.path.basename(file_path)}",
        elevation_status=elevation_status(elevs[:n]),
    )


def parse_profile(file_path: str, elevation_units: str = 'auto') -> ProfileData:
    """
    Auto-dispatch parser based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.kmz':
        return parse_kmz(file_path)
    elif ext == '.kml':
        return parse_kml(file_path)
    elif ext in ('.xlsx', '.xls', '.xlsm'):
        return parse_excel_profile(file_path, elevation_units=elevation_units)
    elif ext in ('.txt', '.csv', '.tsv'):
        return parse_txt_csv(file_path, elevation_units=elevation_units)
    else:
        # Try TXT parser as fallback
        return parse_txt_csv(file_path, elevation_units=elevation_units)
