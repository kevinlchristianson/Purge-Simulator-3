"""
kmz_elevations.py — Look up a route's elevation profile in USGS 3DEP.

Command-line front end to purge_sim/data/elevation.py, the same lookup the app
runs when it imports a KMZ that has no elevations. Use it to produce a profile
outside the app, or to compare spacings.

Usage
-----
    python tools/kmz_elevations.py pipeline.kmz
    python tools/kmz_elevations.py pipeline.kmz --spacing 100 --reverse --out elev_profile.json

Works on anything the profile parser reads with lat/lon: KMZ, KML, GPS TXT/CSV, Excel.
Output is a JSON file: [[milepost_mi, elevation_ft], ...], ready for a scenario's
elevation_profile. Lookups are cached in ~/.purge_sim/elevation_cache.sqlite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purge_sim.data.elevation import (DEFAULT_SPACING_FT, ElevationError,  # noqa: E402
                                      describe_source, fetch_route_elevation)
from purge_sim.data.profile_parser import parse_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Look up a route's elevation profile in USGS 3DEP",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("route", help="Input .kmz / .kml / GPS .txt/.csv / .xlsx file")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <input_base>_elev.json)")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_FT,
                        help=f"Sample spacing along the route, ft (default {DEFAULT_SPACING_FT:g})")
    parser.add_argument("--reverse", action="store_true", help="Run mileposts from the other end of the route")
    parser.add_argument("--no-cache", action="store_true", help="Ignore and don't write the lookup cache")
    args = parser.parse_args()

    if not os.path.exists(args.route):
        print(f"ERROR: file not found: {args.route}", file=sys.stderr)
        return 1
    prof = parse_profile(args.route)
    if prof.lat is None or prof.lon is None:
        print("ERROR: the file has no lat/lon coordinates to look up", file=sys.stderr)
        return 1
    if not prof.needs_elevation:
        print("Note: the file already has elevations; looking them up in 3DEP anyway.")
    lat, lon = list(prof.lat), list(prof.lon)
    if args.reverse:
        lat, lon = lat[::-1], lon[::-1]

    def progress(frac, msg):
        print(f"\r  {msg} ({frac * 100:.0f}%)   ", end="", flush=True)

    print(f"Route: {len(lat)} vertices, {prof.total_length_mi:.2f} mi")
    try:
        res = fetch_route_elevation(lat, lon, spacing_ft=args.spacing, use_cache=not args.no_cache,
                                    progress_cb=progress)
    except ElevationError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    print()
    out_path = args.out or (os.path.splitext(args.route)[0] + "_elev.json")
    with open(out_path, "w") as fh:
        json.dump(res.profile(), fh)
    print(describe_source(res.source))
    for f in res.flags:
        print("  FLAG:", f)
    print(f"Written -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
