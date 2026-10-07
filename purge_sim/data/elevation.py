"""
Elevation lookup for routes that carry no elevation of their own.

Most pipeline KMZs are 2D: the centerline has lat/lon but no altitude (or a
placeholder 0). Without this module those imports came in as a flat 0 ft profile,
which hides every peak and understates static head. Here the route is sampled at
a fixed spacing, each point is looked up in USGS 3DEP (the National Map's
Elevation Point Query Service, US only, no API key), and the result is returned
as a [milepost, elevation_ft] profile plus a provenance record for the scenario.

Sampling:
  - points every `spacing_ft` along the true route length, plus the route's own
    vertices (thinned where they're denser than half the spacing),
  - then up to two refinement passes that bisect any gap where the ground moves
    more than REFINE_STEP_FT between neighbours, so crests aren't missed,
  - no-data points are filled by linear interpolation and reported.

Lookups go through a small sqlite cache (~/.purge_sim/elevation_cache.sqlite), so
re-imports and re-fetches at another spacing reuse earlier answers. Standard
library only (urllib, sqlite3, concurrent.futures), so the packaged app needs
nothing extra.

A DEM gives the ground (or water) surface, not the pipe. Burial depth is
negligible; HDD crossings and river bottoms are not, so long dead-flat runs are
flagged for the engineer to check.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..engine.physics import haversine_miles

FT_PER_MI = 5280.0
DEFAULT_SPACING_FT = 250.0
MIN_SPACING_FT = 25.0
REFINE_STEP_FT = 20.0        # bisect a gap when neighbours differ by more than this
REFINE_MIN_GAP_FT = 30.0     # ...unless they are already this close
MAX_REFINE_PASSES = 2
MAX_POINTS = 25000
FILL_WARN_FRACTION = 0.02
FILL_WARN_RUN_MI = 0.25
FLAT_RUN_MI = 0.25           # a run this long within FLAT_RANGE_FT looks like water
FLAT_RANGE_FT = 0.5
ROUTE_STORE_MIN_FT = 100.0   # route vertices kept in the scenario for re-fetching

ProgressCb = Optional[Callable[[float, str], None]]


class ElevationError(RuntimeError):
    """The lookup could not produce a usable profile. The message is shown as-is."""


# ---------------------------------------------------------------------------
# Provider: USGS 3DEP Elevation Point Query Service
# ---------------------------------------------------------------------------

_NODATA = -1000000.0


def _parse_epqs_value(data) -> Optional[float]:
    """Pull the elevation out of an EPQS response; None for no data."""
    if isinstance(data, dict) and "value" in data:
        val = data["value"]
    elif isinstance(data, dict) and "USGS_Elevation_Point_Query_Service" in data:   # legacy shape
        val = data["USGS_Elevation_Point_Query_Service"].get("Elevation_Query", {}).get("Elevation")
    else:
        raise ValueError(f"unexpected EPQS response: {str(data)[:120]}")
    if val is None or str(val).strip() in ("", "None"):
        return None
    v = float(val)
    return None if v <= _NODATA + 1 else v


class USGS3DEP:
    """One point per request, run on a small thread pool with retry and backoff."""

    name = "usgs_3dep"
    label = "USGS 3DEP (National Map Elevation Point Query Service)"
    # The documented v1 endpoint first; the older form the original tool used as a fallback.
    URLS = (
        "https://epqs.nationalmap.gov/v1/json?x={lon:.7f}&y={lat:.7f}&units=Feet&wkid=4326&includeDate=false",
        "https://epqs.nationalmap.gov/v1/points?x={lon:.7f}&y={lat:.7f}&units=Feet&output=json",
    )

    def __init__(self, workers: int = 8, timeout_s: float = 20.0, retries: int = 3,
                 urlopen: Optional[Callable] = None):
        self.workers = workers
        self.timeout_s = timeout_s
        self.retries = retries
        self._urlopen = urlopen or urllib.request.urlopen
        self._url_idx = 0
        self._lock = threading.Lock()

    def lookup_one(self, lat: float, lon: float) -> Optional[float]:
        """Elevation in feet, None where 3DEP has no data. Raises ElevationError if unreachable."""
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            idx = self._url_idx
            url = self.URLS[idx].format(lat=lat, lon=lon)
            req = urllib.request.Request(url, headers={"User-Agent": "purge-simulator/1.0"})
            try:
                with self._urlopen(req, timeout=self.timeout_s) as resp:
                    return _parse_epqs_value(json.loads(resp.read().decode("utf-8")))
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (400, 404) and idx + 1 < len(self.URLS):
                    with self._lock:   # endpoint moved: switch for every later request
                        if self._url_idx == idx:
                            self._url_idx = idx + 1
                    continue
            except (urllib.error.URLError, OSError, ValueError) as e:
                last = e
            if attempt < self.retries:
                time.sleep(0.5 * 2 ** attempt)
        raise ElevationError(f"USGS 3DEP lookup failed at ({lat:.5f}, {lon:.5f}): {last}")

    def lookup(self, points: Sequence[Tuple[float, float]], progress_cb: ProgressCb = None) -> List[Optional[float]]:
        """Progress is reported from the calling thread, so a Tk callback is safe."""
        out: List[Optional[float]] = [None] * len(points)
        errors: List[str] = []
        abort = threading.Event()

        def one(i: int) -> bool:
            if abort.is_set():
                return False
            try:
                out[i] = self.lookup_one(*points[i])
                return True
            except ElevationError as e:
                errors.append(str(e))
                return False

        ok = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(one, i) for i in range(len(points))]
            for n, fut in enumerate(as_completed(futures), 1):
                ok += bool(fut.result())
                if not ok and len(errors) >= min(8, len(points)):
                    abort.set()   # nothing has worked yet: the service is unreachable, stop now
                if progress_cb:
                    progress_cb(n / len(points), f"{n}/{len(points)} points")
        if abort.is_set() or len(errors) > 0.25 * len(points):
            raise ElevationError(
                "Couldn't reach the USGS elevation service (check the internet connection). "
                "Without it, import a profile that already has elevations, such as a GPSVisualizer "
                f"TXT export. First error: {errors[0] if errors else 'aborted'}")
        return out   # the few points that failed stay None and get filled like no-data


def default_provider():
    """The provider imports use. Tests swap this out."""
    return USGS3DEP()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def default_cache_path() -> str:
    return os.environ.get("PURGE_SIM_ELEV_CACHE") or os.path.join(
        os.path.expanduser("~"), ".purge_sim", "elevation_cache.sqlite")


class ElevationCache:
    """provider + lat/lon (rounded to 1e-5 deg, about 1 m) -> elevation ft (NULL = no data)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_cache_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS elev (provider TEXT, lat INTEGER, lon INTEGER, ft REAL, "
                      "PRIMARY KEY (provider, lat, lon))")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def _key(lat: float, lon: float) -> Tuple[int, int]:
        return int(round(lat * 1e5)), int(round(lon * 1e5))

    def get_many(self, provider: str, points: Sequence[Tuple[float, float]]) -> dict:
        """{index: elevation or None} for every point already cached."""
        found = {}
        with self._conn() as c:
            for i, (lat, lon) in enumerate(points):
                row = c.execute("SELECT ft FROM elev WHERE provider=? AND lat=? AND lon=?",
                                (provider, *self._key(lat, lon))).fetchone()
                if row is not None:
                    found[i] = row[0]
        return found

    def put_many(self, provider: str, items: Sequence[Tuple[float, float, Optional[float]]]) -> None:
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO elev VALUES (?, ?, ?, ?)",
                          [(provider, *self._key(lat, lon), ft) for lat, lon, ft in items])


# ---------------------------------------------------------------------------
# Route geometry
# ---------------------------------------------------------------------------

def valid_coords(lat, lon) -> np.ndarray:
    """Mask of usable coordinates: finite, in range, and not 0,0 (a blank cell some
    spreadsheets export as zero, which would send the route to the Gulf of Guinea)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    with np.errstate(invalid="ignore"):
        return (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
                & ~((lat == 0) & (lon == 0)))


def _clean_route(lat, lon) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = valid_coords(lat, lon)
    lat, lon = lat[ok], lon[ok]
    if len(lat) < 2:
        raise ElevationError("the route needs at least two coordinates to look up elevation")
    keep = [0] + [i for i in range(1, len(lat)) if (lat[i], lon[i]) != (lat[i - 1], lon[i - 1])]
    return lat[keep], lon[keep]


def route_mileposts(lat, lon) -> np.ndarray:
    mp = [0.0]
    for i in range(1, len(lat)):
        mp.append(mp[-1] + haversine_miles(lat[i - 1], lon[i - 1], lat[i], lon[i]))
    return np.asarray(mp)


def sample_mileposts(vertex_mp: np.ndarray, spacing_ft: float) -> np.ndarray:
    """A uniform grid every spacing_ft plus the route vertices, never closer than half a spacing."""
    total = float(vertex_mp[-1])
    step = spacing_ft / FT_PER_MI
    grid = np.arange(0.0, total, step)
    cand = sorted([(m, 0) for m in vertex_mp] + [(m, 1) for m in grid] + [(total, 0)])   # vertices win ties
    out: List[float] = []
    half = 0.5 * step
    for m, kind in cand:
        if not out:
            out.append(m)
        elif m - out[-1] >= half:
            out.append(m)
        elif kind == 0 and len(out) > 1 and m - out[-2] >= half:
            out[-1] = m   # swap a grid point for the vertex right next to it
    if total - out[-1] > 1e-9:
        if total - out[-1] < half and len(out) > 1:
            out[-1] = total
        else:
            out.append(total)
    return np.asarray(out)


def interpolate_route(vertex_mp, lat, lon, mp) -> Tuple[np.ndarray, np.ndarray]:
    return np.interp(mp, vertex_mp, lat), np.interp(mp, vertex_mp, lon)


def thin_route(lat, lon, min_ft: float = ROUTE_STORE_MIN_FT) -> List[list]:
    """Route vertices at least min_ft apart (ends kept), rounded for the scenario JSON."""
    lat, lon = _clean_route(lat, lon)
    keep = [0]
    for i in range(1, len(lat) - 1):
        if haversine_miles(lat[keep[-1]], lon[keep[-1]], lat[i], lon[i]) * FT_PER_MI >= min_ft:
            keep.append(i)
    keep.append(len(lat) - 1)
    return [[round(float(lat[i]), 6), round(float(lon[i]), 6)] for i in keep]


# ---------------------------------------------------------------------------
# The lookup
# ---------------------------------------------------------------------------

@dataclass
class ElevationResult:
    mileposts: np.ndarray
    elevations_ft: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    filled_points: int = 0
    flags: List[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    @property
    def length_mi(self) -> float:
        return float(self.mileposts[-1])

    def profile(self) -> List[list]:
        return [[round(float(m), 5), round(float(e), 1)] for m, e in zip(self.mileposts, self.elevations_ft)]


def _flat_runs(mp: np.ndarray, elev: np.ndarray) -> List[Tuple[float, float]]:
    runs, i, n = [], 0, len(mp)
    while i < n:
        j, lo, hi = i, elev[i], elev[i]
        while j + 1 < n and max(hi, elev[j + 1]) - min(lo, elev[j + 1]) <= FLAT_RANGE_FT:
            j += 1
            lo, hi = min(lo, elev[j]), max(hi, elev[j])
        if mp[j] - mp[i] >= FLAT_RUN_MI:
            runs.append((float(mp[i]), float(mp[j])))
        i = j + 1
    return runs


def fetch_route_elevation(lat, lon, spacing_ft: float = DEFAULT_SPACING_FT, provider=None,
                          cache: Optional[ElevationCache] = None, use_cache: bool = True,
                          progress_cb: ProgressCb = None) -> ElevationResult:
    """Sample the route, look every point up (cache first), refine peaks, fill gaps."""
    spacing_ft = float(spacing_ft)
    if not (MIN_SPACING_FT <= spacing_ft <= 5280.0):
        raise ElevationError(f"spacing must be between {MIN_SPACING_FT:g} and 5280 ft")
    provider = provider or default_provider()
    if use_cache and cache is None:
        cache = ElevationCache()
    lat, lon = _clean_route(lat, lon)
    vmp = route_mileposts(lat, lon)
    if vmp[-1] <= 0:
        raise ElevationError("the route has zero length")
    mp = sample_mileposts(vmp, spacing_ft)
    if len(mp) > MAX_POINTS:
        raise ElevationError(f"{len(mp):,} sample points at {spacing_ft:g} ft is too many; use a wider spacing")

    known: dict = {}   # milepost -> elevation or None

    def look(mps: np.ndarray, base: float, span: float) -> None:
        plat, plon = interpolate_route(vmp, lat, lon, mps)
        pts = list(zip(plat.tolist(), plon.tolist()))
        got = cache.get_many(provider.name, pts) if cache else {}
        todo = [i for i in range(len(pts)) if i not in got]
        if todo:
            def cb(frac, msg):
                if progress_cb:
                    progress_cb(base + span * frac, f"Looking up elevation: {msg}")
            vals = provider.lookup([pts[i] for i in todo], cb)
            for i, v in zip(todo, vals):
                got[i] = v
            if cache:
                cache.put_many(provider.name, [(pts[i][0], pts[i][1], got[i]) for i in todo])
        for i, m in enumerate(mps.tolist()):
            known[m] = got[i]

    look(mp, 0.0, 0.85)
    for p in range(MAX_REFINE_PASSES):
        ms = sorted(known)
        new = []
        for a, b in zip(ms, ms[1:]):
            ea, eb = known[a], known[b]
            if ea is None or eb is None:
                continue
            if abs(eb - ea) > REFINE_STEP_FT and (b - a) * FT_PER_MI > REFINE_MIN_GAP_FT:
                new.append(0.5 * (a + b))
        if not new or len(known) + len(new) > MAX_POINTS:
            break
        look(np.asarray(new), 0.85 + 0.07 * p, 0.07)

    ms = np.asarray(sorted(known))
    raw = np.asarray([np.nan if known[m] is None else known[m] for m in ms], dtype=float)
    good = np.isfinite(raw)
    if good.sum() < 2:
        raise ElevationError("USGS 3DEP has no elevation for this route. It covers the US only; for other "
                             "routes import a profile that already has elevations.")
    elev = np.where(good, raw, np.interp(ms, ms[good], raw[good]))

    flags: List[str] = []
    filled = int((~good).sum())
    if filled:
        longest, k, n = 0.0, 0, len(ms)
        while k < n:
            if good[k]:
                k += 1
                continue
            j = k
            while j + 1 < n and not good[j + 1]:
                j += 1
            # the gap spans from the last good point before it to the first good one after
            longest = max(longest, float(ms[min(j + 1, n - 1)] - ms[max(k - 1, 0)]))
            k = j + 1
        if filled > FILL_WARN_FRACTION * len(ms) or longest > FILL_WARN_RUN_MI:
            flags.append(f"{filled} of {len(ms)} points had no elevation data and were interpolated "
                         f"(longest gap {longest:.2f} mi); check those stretches.")
    for a, b in _flat_runs(ms, elev):
        flags.append(f"Dead-flat ground MP {a:.2f}-{b:.2f}: likely a water surface. The DEM gives the "
                     "surface, not the pipe; check for a river or HDD crossing there.")

    if progress_cb:
        progress_cb(1.0, "Elevation lookup complete")
    plat, plon = interpolate_route(vmp, lat, lon, ms)
    source = {
        "provider": provider.name,
        "dataset": getattr(provider, "label", provider.name),
        "spacing_ft": spacing_ft,
        "refine_step_ft": REFINE_STEP_FT,
        "points": int(len(ms)),
        "filled_points": filled,
        "length_mi": round(float(ms[-1]), 4),
        "fetched": date.today().isoformat(),
        "flags": flags,
    }
    return ElevationResult(mileposts=ms, elevations_ft=elev, lat=plat, lon=plon,
                           filled_points=filled, flags=flags, source=source)


def describe_source(source: dict) -> str:
    """One line for scenario notes and the UI."""
    if not source:
        return ""
    if source.get("provider") == "file":
        return f"Elevation from the imported file ({source.get('file', '')})."
    return (f"Elevation: {source.get('dataset', source.get('provider'))}, {source.get('points')} points at "
            f"{source.get('spacing_ft', 0):g} ft spacing with peak refinement, fetched {source.get('fetched')}"
            + (f"; {source['filled_points']} points interpolated" if source.get("filled_points") else "") + ".")
