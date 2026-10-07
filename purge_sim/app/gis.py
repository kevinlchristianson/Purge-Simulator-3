"""
Route geometry for the app's Map tab: where each milepost is on the ground.

A scenario stores its route as bare [lat, lon] vertices (route_latlon, from a KMZ,
GPS or PxP import). Mileposts come from the elevation profile, so a vertex's
milepost is its distance along the route, scaled so the route's ends land on the
profile's first and last mileposts. For a KMZ import those agree to rounding; for
an ILI or PxP scenario with a KMZ added later, the scale absorbs the difference
between surveyed stationing and the KMZ's map length (reported, and flagged when
it is large, because station markers will then sit slightly off).

attach_route() adds a route to a scenario that has none (every bundled client job),
without touching its elevation profile or mileposts.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from ..data.elevation import interpolate_route, route_mileposts, thin_route
from ..data.scenario import ScenarioInputs

# Route vertices sent to the browser. Plenty for a pipeline at map scale.
MAX_VERTICES = 3000
# Above this the KMZ length and the profile's mileposts disagree enough to flag.
SCALE_WARN = 0.03


class RouteError(ValueError):
    """The route file can't be used. The message is shown to the user as-is."""


def _downsample(n: int, max_points: int) -> List[int]:
    if n <= max_points:
        return list(range(n))
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return sorted(set(int(i) for i in idx))


def _profile_span(inp: ScenarioInputs) -> Optional[Tuple[float, float]]:
    prof = inp.elevation_profile
    if not prof or len(prof) < 2:
        return None
    return float(prof[0][0]), float(prof[-1][0])


def vertex_mileposts(inp: ScenarioInputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(milepost, lat, lon) per route vertex, plus the route's map length (mi) and the scale
    applied to it to fit the profile's milepost span (1.0 when there is no profile)."""
    a = np.asarray(inp.route_latlon, dtype=float)
    lat, lon = a[:, 0], a[:, 1]
    d = route_mileposts(lat, lon)
    length = float(d[-1])
    span = _profile_span(inp)
    if span is None or length <= 0:
        return d, lat, lon, length, 1.0
    scale = (span[1] - span[0]) / length
    return span[0] + d * scale, lat, lon, length, scale


def _round(v, nd=6):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def map_view(inp: ScenarioInputs) -> dict:
    """Everything the Map tab draws that doesn't depend on a run: the route with a milepost
    and ground elevation per vertex, and the stations, check valves, boosters and BPCV
    placed on it."""
    span = _profile_span(inp)
    out = {"has_route": len(inp.route_latlon) >= 2,
           "purge_start_mp": inp.purge_start_mp, "purge_end_mp": inp.purge_end_mp,
           "profile_mp_range": list(span) if span else None,
           "warnings": []}
    if not out["has_route"]:
        return out
    vmp, lat, lon, length, scale = vertex_mileposts(inp)
    idx = _downsample(len(vmp), MAX_VERTICES)
    # keep the purge ends exact: insert interpolated vertices there
    mp_s = np.asarray([vmp[i] for i in idx])
    extra = [m for m in (inp.purge_start_mp, inp.purge_end_mp) if mp_s[0] < m < mp_s[-1] and not np.isclose(mp_s, m).any()]
    mp_all = np.sort(np.concatenate([mp_s, extra])) if extra else mp_s
    plat, plon = interpolate_route(vmp, lat, lon, mp_all)
    prof = np.asarray(inp.elevation_profile, dtype=float) if span else None
    elev = np.interp(mp_all, prof[:, 0], prof[:, 1]) if prof is not None else np.full(len(mp_all), np.nan)

    def place(mp: float) -> Tuple[Optional[float], Optional[float]]:
        if not (vmp[0] - 1e-6 <= mp <= vmp[-1] + 1e-6):
            return None, None
        la, lo = interpolate_route(vmp, lat, lon, mp)
        return _round(la), _round(lo)

    feats = []

    def add(kind: str, name: str, mp, **extra_props):
        if mp is None:
            return
        la, lo = place(float(mp))
        feats.append({"kind": kind, "name": name, "mp": _round(mp, 3), "lat": la, "lon": lo, **extra_props})

    add("purge_start", "Launch (purge start)", inp.purge_start_mp)
    add("purge_end", "Pig stop (purge end)", inp.purge_end_mp)
    pump_mps = [float(p["mp"]) for p in inp.pump_stations]
    for p in inp.pump_stations:
        add("pump", p.get("name", "Pump station"), p["mp"], suction_psig=p.get("suction_psig"))
    for cv in inp.check_valves:
        if any(abs(float(cv["mp"]) - m) < 0.01 for m in pump_mps):
            continue   # the check valve at a pump station: one marker is enough
        add("check_valve", cv.get("name", "Check valve"), cv["mp"])
    for b in inp.booster_stations:
        add("booster", b.get("name", "Booster site"), b["mp"])
    if inp.bpcv:
        add("bpcv", inp.bpcv.get("name") or "BPCV", inp.bpcv["mp"])
    unplaced = [f["name"] for f in feats if f["lat"] is None]
    if unplaced:
        out["warnings"].append("Not on the mapped route (milepost outside it): " + ", ".join(unplaced) + ".")
    if abs(scale - 1.0) > SCALE_WARN:
        out["warnings"].append(
            f"The route's map length ({length:.2f} mi) differs from the profile's milepost span "
            f"({span[1] - span[0]:.2f} mi) by {abs(scale - 1) * 100:.0f}%. Mileposts are spread evenly along "
            "the route to fit, so markers can sit off their true place. Check the route file covers the "
            "same stretch of pipe, in the same direction.")
    lats, lons = np.asarray(plat), np.asarray(plon)
    out.update({
        "mp": [_round(m, 4) for m in mp_all],
        "lat": [_round(v) for v in lats],
        "lon": [_round(v) for v in lons],
        "elevation_ft": [_round(e, 1) for e in elev],
        "bounds": [[_round(lats.min()), _round(lons.min())], [_round(lats.max()), _round(lons.max())]],
        "route_length_mi": _round(length, 3),
        "mp_scale": _round(scale, 4),
        "features": feats,
    })
    return out


# ---------------------------------------------------------------------------
# Adding a route to a scenario that has none
# ---------------------------------------------------------------------------

def _direction_fit(file_mp, file_elev, prof: np.ndarray) -> Optional[dict]:
    """How well the file's own elevations match the scenario's profile, run forward and
    reversed (both scaled to the profile's span). None when the file has no elevations."""
    ok = np.isfinite(file_elev)
    if ok.sum() < 5 or len(prof) < 5:
        return None
    fm, fe = np.asarray(file_mp)[ok], np.asarray(file_elev)[ok]
    total = fm[-1] - fm[0]
    if total <= 0:
        return None
    a, b = prof[0, 0], prof[-1, 0]
    x = a + (fm - fm[0]) / total * (b - a)
    xr = a + (fm[-1] - fm) / total * (b - a)
    pe = np.interp(x, prof[:, 0], prof[:, 1])
    pr = np.interp(xr, prof[:, 0], prof[:, 1])
    if np.std(fe) < 1 or np.std(pe) < 1:
        return None

    def corr(u, v):
        return float(np.corrcoef(u, v)[0, 1]) if np.std(v) > 0 else 0.0
    return {"forward": corr(fe, pe), "reversed": corr(fe, pr)}


def route_from_file(path: str, inp: ScenarioInputs, direction: str = "auto") -> Tuple[List[list], List[str]]:
    """Read route coordinates from a KMZ / KML / GPS TXT / CSV / Excel file to sit on `inp`'s
    existing mileposts. direction: 'auto' (from the file's elevations when it has any,
    else as drawn), 'as_is' or 'reverse'. Returns (route_latlon, notes)."""
    from ..data.profile_parser import parse_profile
    if direction not in ("auto", "as_is", "reverse"):
        raise RouteError("direction must be 'auto', 'as_is' or 'reverse'")
    try:
        p = parse_profile(path)
    except Exception as e:
        raise RouteError(f"Couldn't read {os.path.basename(path)}: {e}") from e
    if p.lat is None or p.lon is None or len(p.lat) < 2:
        raise RouteError(f"{os.path.basename(path)} has no route coordinates (latitude/longitude); "
                         "use the pipeline's KMZ or a GPS export")
    lat, lon = np.asarray(p.lat, dtype=float), np.asarray(p.lon, dtype=float)
    notes: List[str] = []
    flip = direction == "reverse"
    if direction == "auto":
        prof = np.asarray(inp.elevation_profile, dtype=float) if inp.elevation_profile else np.empty((0, 2))
        fit = _direction_fit(p.mileposts, np.asarray(p.elevations_ft, dtype=float), prof) if len(prof) else None
        if fit is None:
            notes.append("Used the route in the direction it is drawn in the file (no elevations in the file to "
                         "check it against). If the launch marker is at the wrong end, add it again reversed.")
        else:
            flip = fit["reversed"] > fit["forward"]
            notes.append(f"Direction picked from the file's elevations: {'reversed' if flip else 'as drawn'} "
                         f"(profile match {max(fit.values()):.2f} vs {min(fit.values()):.2f}).")
            if max(fit.values()) < 0.5:
                notes.append("The file's elevations match this scenario's profile poorly in either direction; "
                             "check it is the same pipeline.")
    if flip:
        lat, lon = lat[::-1], lon[::-1]
    try:
        route = thin_route(lat, lon)
    except Exception as e:
        raise RouteError(str(e)) from e
    return route, notes
