"""
kmz_elevations.py — Fetch elevation data for pipeline KMZ routes via USGS 3DEP.

The USGS National Elevation Dataset (3DEP) point-query API is authoritative,
free, requires no API key, and works for all US locations.

Usage
-----
    python tools/kmz_elevations.py pipeline.kmz

    python tools/kmz_elevations.py pipeline.kmz \\
        --out elev_profile.json \\
        --max-points 500 \\
        --delay 0.05

Output is a JSON file: [[milepost_mi, elevation_ft], ...]
Paste or import this directly into the scenario editor as elevation_profile.

Options
-------
--out         Output JSON path  (default: <kmz_base>_elev.json)
--max-points  Subsample route to at most N points  (default: 500)
--delay       Seconds between API requests  (default: 0.05)
--units       Elevation units: Feet or Meters  (default: Feet)

API
---
Endpoint: https://epqs.nationalmap.gov/v1/points
  ?x={lon}&y={lat}&units=Feet&output=json
Returns JSON with "value" key containing the elevation as a string.
Rate limit: ~20 requests/second sustained; use --delay if you hit errors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Tuple


# KML namespace
_KML_NS = {
    'kml': 'http://www.opengis.net/kml/2.2',
    'kml22': 'http://earth.google.com/kml/2.2',
    'kml21': 'http://earth.google.com/kml/2.1',
}


def _haversine_mi(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in miles between two WGS-84 points."""
    R = 3958.8  # Earth radius, miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_kmz(path: str) -> List[Tuple[float, float]]:
    """
    Extract (lat, lon) coordinate pairs from a KMZ (or KML) file.
    Searches all LineString, LinearRing, and Point elements for coordinates.
    Returns a flat list in route order.
    """
    coords: List[Tuple[float, float]] = []

    if path.lower().endswith('.kmz'):
        with zipfile.ZipFile(path) as zf:
            # Find the main KML entry (typically doc.kml or the first .kml)
            kml_names = [n for n in zf.namelist() if n.lower().endswith('.kml')]
            if not kml_names:
                raise ValueError(f"No .kml file found inside {path}")
            kml_text = zf.read(kml_names[0]).decode('utf-8', errors='replace')
    else:
        with open(path, encoding='utf-8', errors='replace') as fh:
            kml_text = fh.read()

    # Strip the namespace prefix to simplify XPath queries
    kml_text = kml_text.replace('xmlns=', 'xmlns:kml_default=')
    for ns_uri in list(_KML_NS.values()):
        kml_text = kml_text.replace(f' xmlns="{ns_uri}"', '')
        kml_text = kml_text.replace(f'xmlns="{ns_uri}"', '')

    root = ET.fromstring(kml_text)

    def _iter_coords(element):
        for child in element.iter():
            tag = child.tag.split('}')[-1]  # strip any remaining namespace
            if tag == 'coordinates' and child.text:
                for triplet in child.text.strip().split():
                    parts = triplet.split(',')
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            yield lat, lon
                        except ValueError:
                            pass

    coords = list(_iter_coords(root))
    if not coords:
        raise ValueError(f"No coordinate data found in {path}. "
                         "Make sure the file contains a LineString or track.")
    return coords


def _subsample(coords: List[Tuple[float, float]], max_n: int) -> List[Tuple[float, float]]:
    """Evenly subsample a coordinate list to at most max_n points."""
    if len(coords) <= max_n:
        return coords
    indices = [round(i * (len(coords) - 1) / (max_n - 1)) for i in range(max_n)]
    return [coords[i] for i in indices]


def _build_milepost_series(coords: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
    """
    Compute cumulative milepost for each coordinate.
    Returns [(lat, lon, milepost_mi), ...] with milepost starting at 0.
    """
    result = [(coords[0][0], coords[0][1], 0.0)]
    cumulative = 0.0
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i - 1]
        lat2, lon2 = coords[i]
        cumulative += _haversine_mi(lat1, lon1, lat2, lon2)
        result.append((lat2, lon2, cumulative))
    return result


def _query_usgs_elevation(lat: float, lon: float, units: str = 'Feet') -> float:
    """
    Query the USGS 3DEP Elevation Point Query Service.
    Returns elevation as a float in the requested units.
    Raises RuntimeError if the query fails or returns a no-data value.
    """
    params = urllib.parse.urlencode({'x': lon, 'y': lat, 'units': units, 'output': 'json'})
    url = f"https://epqs.nationalmap.gov/v1/points?{params}"
    req = urllib.request.Request(url, headers={'User-Agent': 'purge-simulator/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        val = data.get('value', '-1000000')
        if val is None or str(val).strip() in ('-1000000', '-1000000.0', 'None', ''):
            raise RuntimeError(f"No elevation data at ({lat:.5f}, {lon:.5f})")
        return float(val)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"USGS API request failed: {exc}") from exc


def fetch_elevations(
    kmz_path: str,
    out_path: str,
    max_points: int = 500,
    delay_s: float = 0.05,
    units: str = 'Feet',
) -> List[List[float]]:
    """
    Main pipeline: parse KMZ → subsample → query USGS → write JSON.

    Returns the [[milepost, elevation], ...] list.
    """
    print(f"Parsing:  {kmz_path}")
    raw_coords = _parse_kmz(kmz_path)
    print(f"  {len(raw_coords)} coordinate points found in KMZ")

    coords = _subsample(raw_coords, max_points)
    print(f"  Subsampled to {len(coords)} points")

    pts = _build_milepost_series(coords)
    total_mi = pts[-1][2]
    print(f"  Route length: {total_mi:.2f} miles")

    print(f"Querying USGS 3DEP ({units})...")
    profile: List[List[float]] = []
    errors = 0
    for i, (lat, lon, mp) in enumerate(pts):
        try:
            elev = _query_usgs_elevation(lat, lon, units)
            profile.append([round(mp, 4), round(elev, 1)])
        except RuntimeError as exc:
            errors += 1
            print(f"  WARNING: {exc} — interpolating later")
            profile.append([round(mp, 4), None])

        # Progress every 50 points
        if (i + 1) % 50 == 0 or i + 1 == len(pts):
            pct = (i + 1) / len(pts) * 100
            print(f"  {i + 1}/{len(pts)}  ({pct:.0f}%)  MP {mp:.2f}", flush=True)

        if delay_s > 0 and i < len(pts) - 1:
            time.sleep(delay_s)

    # Interpolate any failed lookups
    if errors:
        mps  = [p[0] for p in profile]
        elvs = [p[1] for p in profile]
        good_mps  = [m for m, e in zip(mps, elvs) if e is not None]
        good_elvs = [e for e in elvs if e is not None]
        for i, (m, e) in enumerate(zip(mps, elvs)):
            if e is None:
                if good_mps:
                    import bisect
                    idx = bisect.bisect_left(good_mps, m)
                    if idx == 0:
                        profile[i][1] = good_elvs[0]
                    elif idx >= len(good_mps):
                        profile[i][1] = good_elvs[-1]
                    else:
                        frac = (m - good_mps[idx-1]) / (good_mps[idx] - good_mps[idx-1])
                        profile[i][1] = good_elvs[idx-1] + frac * (good_elvs[idx] - good_elvs[idx-1])
                else:
                    profile[i][1] = 0.0
        print(f"  {errors} failed point(s) interpolated from neighbors.")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(profile, fh, indent=2)

    print(f"\nElevation profile written -> {out_path}")
    print(f"  {len(profile)} points  |  {profile[0][1]:.0f} {units} → {profile[-1][1]:.0f} {units}")
    print(f"  Paste the contents of this file into the scenario 'elevation_profile' field,")
    print(f"  or use scenario_editor to import it.")

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Fetch pipeline elevations from a KMZ file via USGS 3DEP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('kmz', help='Input .kmz or .kml file')
    parser.add_argument('--out', default=None,
                        help='Output JSON path (default: <kmz_base>_elev.json)')
    parser.add_argument('--max-points', type=int, default=500,
                        help='Max sample points along route (default: 500)')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Seconds between API requests (default: 0.05)')
    parser.add_argument('--units', choices=['Feet', 'Meters'], default='Feet',
                        help='Elevation units (default: Feet)')
    args = parser.parse_args()

    if not os.path.exists(args.kmz):
        print(f"ERROR: file not found: {args.kmz}", file=sys.stderr)
        sys.exit(1)

    base = os.path.splitext(args.kmz)[0]
    out_path = args.out or (base + '_elev.json')

    fetch_elevations(
        kmz_path=args.kmz,
        out_path=out_path,
        max_points=args.max_points,
        delay_s=args.delay,
        units=args.units,
    )


if __name__ == '__main__':
    main()
