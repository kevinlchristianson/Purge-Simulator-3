"""
Build a new scenario from a raw data file (ILI Excel, KMZ/KML, TXT/CSV, Excel profile).

This is the first step of every new job. The parsers already exist in
purge_sim/data; this turns their output into a ScenarioInputs the app can edit,
save and run. Everything the parser can't know (fluid, drive limits, which
detected stations really pump) is left at defaults for the user, or the
assistant, to set and justify.
"""

from __future__ import annotations

import os

from ..data.scenario import Scenario, ScenarioInputs, ScenarioMeta


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
    )
    if d.bpcv_record is not None:
        b = d.bpcv_record
        inp.bpcv = {"mp": float(b.mp), "elevation_ft": float(b.elevation_ft), "name": f"{b.site} BPCV"}
    return inp


def _from_profile(path: str) -> ScenarioInputs:
    from ..data.profile_parser import parse_profile
    p = parse_profile(path)
    elev = [[float(m), float(e)] for m, e in p.elevation_profile_array()]
    start, end = float(elev[0][0]), float(elev[-1][0])
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ScenarioInputs(
        purge_start_mp=round(start, 3),
        purge_end_mp=round(end, 3),
        elevation_profile=elev,
        # Placeholder geometry: a profile file carries no pipe data. Must be set per job.
        pipe_segments=[{"start_mp": start, "end_mp": end, "od_in": 24.0, "wt_in": 0.313}],
        data_source={"kmz": "KMZ", "kml": "KMZ", "xlsx": "Excel", "xls": "Excel", "xlsm": "Excel"}.get(ext, "TXT"),
        data_file=os.path.basename(path),
    )


def scenario_from_file(path: str, kind: str, name: str) -> Scenario:
    kind = (kind or "").lower()
    if kind == "ili":
        inputs = _from_ili(path)
    elif kind == "profile":
        inputs = _from_profile(path)
    else:
        raise ValueError("kind must be 'ili' or 'profile'")
    meta = ScenarioMeta(name=name or os.path.splitext(os.path.basename(path))[0],
                        source_files=[os.path.basename(path)])
    return Scenario(meta=meta, inputs=inputs)
