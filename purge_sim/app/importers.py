"""
Build a new scenario from a raw data file (ILI Excel, KMZ/KML, TXT/CSV, Excel profile).

This is the first step of every new job. The parsers already exist in
purge_sim/data; this turns their output into a ScenarioInputs the app can edit,
save and run. Everything the parser can't know (fluid, drive limits, which
detected stations really pump) is left at defaults for the user, or the
assistant, to set and justify.

A KMZ (or GPS file) without elevations gets them from USGS 3DEP during import
(purge_sim/data/elevation.py), unless the user's settings say to ask first.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from ..data.elevation import DEFAULT_SPACING_FT, describe_source, fetch_route_elevation, thin_route
from ..data.scenario import Scenario, ScenarioInputs, ScenarioMeta
from . import settings


def _from_ili(path: str) -> ScenarioInputs:
    from ..data.ili_parser import parse_ili
    d = parse_ili(path)
    elev = [[float(m), float(e)] for m, e in d.elevation_profile]
    inp = ScenarioInputs(
        purge_start_mp=round(float(elev[0][0]), 3),
        purge_end_mp=round(float(elev[-1][0]), 3),
        elevation_profile=elev,
        pipe_segments=[{"start_mp": float(s), "end_mp": float(e), "od_in": float(od), "wt_in": float(wt)}
                       for s, e, od, wt in d.pipe_geometry.segments],
        mop_joints=[{"mp": float(j.mp), "mop_psig": float(j.mop_psig), "elevation_ft": float(j.elevation_ft),
                     "od_in": float(j.od_in), "wt_in": float(j.wt_in)} for j in d.mop_joints],
        check_valves=[{"mp": float(cv.mp), "name": cv.name, "is_pump_station": bool(cv.is_pump_station)}
                      for cv in d.check_valves()],
        # 30 psig suction matches the desktop app's ILI import default.
        pump_stations=[{"mp": float(ps.mp), "name": ps.name, "suction_psig": 30.0}
                       for ps in d.pump_station_configs()],
        data_source="ILI",
        data_file=os.path.basename(path),
        elevation_source={"provider": "file", "file": os.path.basename(path)},
    )
    if d.bpcv_record is not None:
        b = d.bpcv_record
        inp.bpcv = {"mp": float(b.mp), "elevation_ft": float(b.elevation_ft), "name": f"{b.site} BPCV"}
    return inp


class ElevationConfirmNeeded(ValueError):
    """The file has no elevation and the user asked to confirm before a USGS lookup."""

    def __init__(self, points: int, length_mi: float):
        self.points, self.length_mi = points, length_mi
        super().__init__(f"This file has no elevation data. Look up its {length_mi:.1f} mi route in "
                         "USGS 3DEP? That sends the route's coordinates to the USGS service.")


def _from_profile(path: str, fetch_elevation: Optional[bool] = None,
                  spacing_ft: float = DEFAULT_SPACING_FT, progress_cb=None) -> Tuple[ScenarioInputs, List[str]]:
    """fetch_elevation: None follows the ask-first setting, True looks up, False refuses."""
    from ..data.profile_parser import parse_profile
    p = parse_profile(path)
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    data_source = {"kmz": "KMZ", "kml": "KMZ", "xlsx": "Excel", "xls": "Excel", "xlsm": "Excel"}.get(ext, "TXT")
    has_route = p.lat is not None and p.lon is not None
    notes: List[str] = []
    if p.needs_elevation:
        if not has_route:
            raise ValueError(f"{os.path.basename(path)} has no usable elevations and no coordinates to "
                             "look them up from")
        if fetch_elevation is None:
            fetch_elevation = not settings.elevation_ask_first()
            if not fetch_elevation:
                raise ElevationConfirmNeeded(len(p.lat), p.total_length_mi)
        if not fetch_elevation:
            raise ValueError(f"{os.path.basename(path)} has no elevation data; import a profile with "
                             "elevations (for example a GPSVisualizer TXT export) or allow the USGS lookup")
        res = fetch_route_elevation(p.lat, p.lon, spacing_ft=spacing_ft, progress_cb=progress_cb)
        elev = res.profile()
        source = res.source
        if p.elevation_status == "partial":
            notes.append("The file had elevations on only some points; all were replaced by the USGS lookup.")
        data_source += "+USGS"
    else:
        elev = [[float(m), float(e)] for m, e in p.elevation_profile_array()]
        source = {"provider": "file", "file": os.path.basename(path)}
    start, end = float(elev[0][0]), float(elev[-1][0])
    inputs = ScenarioInputs(
        purge_start_mp=round(start, 3),
        purge_end_mp=round(end, 3),
        elevation_profile=elev,
        elevation_source=source,
        route_latlon=thin_route(p.lat, p.lon) if has_route else [],
        # Placeholder geometry: a profile file carries no pipe data. Must be set per job.
        pipe_segments=[{"start_mp": start, "end_mp": end, "od_in": 24.0, "wt_in": 0.313}],
        data_source=data_source,
        data_file=os.path.basename(path),
    )
    return inputs, notes + list(source.get("flags", []))


def scenario_from_file(path: str, kind: str, name: str, fetch_elevation: Optional[bool] = None,
                       spacing_ft: float = DEFAULT_SPACING_FT, progress_cb=None) -> Tuple[Scenario, List[str]]:
    """Returns the scenario and any warnings about its data (shown to the user)."""
    kind = (kind or "").lower()
    warnings: List[str] = []
    if kind == "ili":
        inputs = _from_ili(path)
    elif kind == "profile":
        inputs, warnings = _from_profile(path, fetch_elevation, spacing_ft, progress_cb)
    else:
        raise ValueError("kind must be 'ili' or 'profile'")
    meta = ScenarioMeta(name=name or os.path.splitext(os.path.basename(path))[0],
                        source_files=[os.path.basename(path)],
                        notes=describe_source(inputs.elevation_source)
                        if inputs.elevation_source.get("provider") != "file" else "")
    return Scenario(meta=meta, inputs=inputs), warnings
