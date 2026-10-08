"""
client_report.py  —  self-contained HTML page of one run, for the client.

One file, no external assets: styles, data and the small chart script are all inline,
so it opens offline in any browser, prints cleanly, and can be emailed as is.

It reuses the app's own result views (results_summary / results_series), the FILL REPORT
rows of the xlsx deliverable (purge_report._build_rows) and the engine's HGL
(hgl.compute_hgl) rather than recomputing anything. The page layout and chart script live
in static/report_template.html; this module only fills in the data.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
from datetime import datetime
from typing import List, Optional

import numpy as np

from ..engine.hgl import compute_hgl
from ..engine.purge_report import _build_rows, _deployed_boosters, _select_n
from ..engine.simulator import SimResults
from .. import branding
from . import paths

GRID_POINTS = 1000     # milepost grid the profile frames share
PROFILE_FRAMES = 60    # pipeline-profile snapshots on the page's time slider
FILL_ROWS = 50         # same 50-point table as the xlsx FILL REPORT

_FILL = [   # (column, row key, decimals): the xlsx FILL REPORT columns
    ("Miles", "miles", 3), ("Elevation (ft)", "elevation_ft", 1), ("Elapsed Time (hr)", "elapsed_hr", 2),
    ("Drive Pressure (psi)", "drive_psi", 1), ("Friction Loss (psi)", "friction_psi", 1),
    ("Injection Rate (SCFM)", "inj_scfm", 0), ("Cumulative N2 (SCF)", "cum_scf", 0),
    ("Pig Speed (mph)", "speed_mph", 2),
]


def _r(v, nd=1):
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd else int(round(v))


def _fluid_line(inputs, cfg) -> str:
    name = getattr(inputs, "fluid_name", "") if inputs is not None else ""
    return f"{name or 'Liquid'}, SG {cfg.fluid_sg:.3f}, {cfg.fluid_viscosity_cst:g} cSt"


def _pipe_line(inputs, cfg) -> str:
    segs = list(getattr(inputs, "pipe_segments", None) or [])
    if not segs:
        od, wt = cfg.pipe_geometry.od_wt_at(cfg.purge_start_mp)
        return f'{od:.3f}" OD × {wt:.3f}" WT'
    sizes = []
    for s in segs:
        txt = f'{s.get("od_in", 0):.3f}" OD × {s.get("wt_in", 0):.3f}" WT'
        if txt not in sizes:
            sizes.append(txt)
    return "; ".join(sizes) if len(sizes) <= 3 else f"{sizes[0]} (+{len(sizes) - 1} other wall thicknesses)"


def _basis(res: SimResults, inputs) -> List[list]:
    cfg = res.config
    span = cfg.purge_end_mp - cfg.purge_start_mp
    rows = [
        ["Route", f"MP {cfg.purge_start_mp:.2f} to MP {cfg.purge_end_mp:.2f} ({span:.2f} mi)"],
        ["Pipe", _pipe_line(inputs, cfg)],
        ["Product displaced", _fluid_line(inputs, cfg)],
        ["Pig speed", f"target {cfg.target_speed_mph:g} mph (min {cfg.min_speed_mph:g}, max {cfg.max_speed_mph:g})"],
        ["Max N2 drive pressure", f"{cfg.max_drive_psig:,.0f} psig" if math.isfinite(cfg.max_drive_psig) else "not limited"],
        ["Max injection", (f"{cfg.max_injection_psig:,.0f} psig, " if math.isfinite(cfg.max_injection_psig) else "")
         + f"{cfg.max_injection_scfm:,.0f} SCFM"],
        ["Exit pressure", f"{cfg.exit_pressure_run_psig:g} psig while running, {cfg.exit_pressure_end_psig:g} psig at the end"],
        ["N2 temperature", f"{cfg.n2_temperature_f:g} °F"],
    ]
    if cfg.mop_joints:
        j = min(cfg.mop_joints, key=lambda m: m.mop_psig)
        rows.append(["MOP", f"{len(cfg.mop_joints):,} joints, lowest {j.mop_psig:,.0f} psig at MP {j.mp:.2f}"])
    if cfg.bpcv is not None:
        rows.append(["Back-pressure control valve", f"MP {getattr(cfg.bpcv, 'mp', cfg.purge_end_mp):.2f}"])
    return rows


def _facilities(res: SimResults) -> List[dict]:
    cfg = res.config
    out = [{"type": "Pump station", "name": p.name, "mp": _r(p.mp, 2)} for p in cfg.pump_stations]
    out += [{"type": "Check valve", "name": c.name, "mp": _r(c.mp, 2)}
            for c in cfg.check_valves if not c.is_pump_station]
    out += [{"type": "N2 booster (ran)", "name": n, "mp": _r(mp, 2)} for n, mp in _deployed_boosters(res)]
    return sorted(out, key=lambda f: f["mp"] if f["mp"] is not None else 0)


def _fill_table(res: SimResults) -> dict:
    rows = _select_n(_build_rows(res), FILL_ROWS)
    boosters = [n for n, _ in _deployed_boosters(res)]
    cols = [c for c, _, _ in _FILL] + [f"{n} booster flow (SCFM)" for n in boosters]
    out = []
    for row in rows:
        vals = [_r(row.get(k), nd) for _, k, nd in _FILL]
        vals += [_r(row.get("booster_flow", {}).get(n, 0.0), 0) for n in boosters]
        flag = "viol" if row.get("mop_viol") else ("slack" if row.get("slack") else "")
        out.append({"v": vals, "flag": flag})
    return {"cols": cols, "rows": out}


def _elevation_lookup(ep: np.ndarray):
    """Scalar elevation interpolator for compute_hgl. Same values as its default np.interp
    lambda, but ~20x faster per call on a 30k-point profile (no array conversion per point),
    which turns a long-route export from minutes into seconds."""
    xs, ys = ep[:, 0].tolist(), ep[:, 1].tolist()
    n = len(xs)

    def at(x: float) -> float:
        i = bisect.bisect_right(xs, x)
        if i <= 0:
            return ys[0]
        if i >= n:
            return ys[-1]
        x0, x1 = xs[i - 1], xs[i]
        return ys[i - 1] + (ys[i] - ys[i - 1]) * ((x - x0) / (x1 - x0) if x1 > x0 else 0.0)
    return at


def _profile(res: SimResults) -> dict:
    """Pipeline profile on one shared milepost grid: elevation and MOP once, then the
    pressure along the line at evenly spaced moments of the run."""
    cfg = res.config
    start, end = cfg.purge_start_mp, cfg.purge_end_mp
    grid = np.linspace(start, end, GRID_POINTS)
    ep = np.asarray(cfg.elevation_profile, dtype=float)
    elev = np.interp(grid, ep[:, 0], ep[:, 1]) if ep.size else np.zeros_like(grid)
    mop = None
    if cfg.mop_joints:
        jm = np.array([j.mp for j in cfg.mop_joints], dtype=float)
        jp = np.array([j.mop_psig for j in cfg.mop_joints], dtype=float)
        order = np.argsort(jm)
        mop = [_r(v, 0) for v in np.interp(grid, jm[order], jp[order])]
    elev_at = _elevation_lookup(ep) if ep.size else None
    steps = res.steps
    idx = sorted(set(np.round(np.linspace(0, len(steps) - 1, PROFILE_FRAMES)).astype(int))) if steps else []
    frames = []
    for i in idx:
        s = steps[i]
        h = compute_hgl(s, cfg, elevation_at=elev_at)
        gas = h.is_gas
        pig = float(s.pig_mp)
        p = np.full(grid.shape, np.nan)
        if gas.any():
            m = grid <= pig
            p[m] = np.interp(grid[m], h.mp[gas], h.pressure_psig[gas])
        if (~gas).any():
            m = grid > pig
            p[m] = np.interp(grid[m], h.mp[~gas], h.pressure_psig[~gas])
        frames.append({
            "t": _r(s.t_hr, 2), "pig": _r(pig, 3), "speed": _r(s.pig_speed_mph, 2),
            "face": _r(s.pig_face_psig, 0), "exit_mp": _r(s.exit_mp, 2), "exit_psig": _r(s.exit_psig, 0),
            "slack": bool(s.slack_line_risk),
            "gas_end": _r(h.pressure_psig[gas][-1], 0) if gas.any() else None,
            "liq_start": _r(h.pressure_psig[~gas][0], 0) if (~gas).any() else None,
            "p": [_r(v, 0) for v in p],
            "stations": [{"name": st.get("name", ""), "mp": _r(st.get("mp"), 2), "status": st.get("status", "")}
                         for st in (s.station_pressures or [])],
        })
    return {"mp": [_r(v, 3) for v in grid], "elev": [_r(v, 0) for v in elev], "mop": mop, "frames": frames}


def build_data(res: SimResults, scenario_name: str, inputs=None, notes: str = "",
               date: Optional[str] = None) -> dict:
    from .workspace import results_series, results_summary   # lazy: workspace imports this module
    return {
        "title": scenario_name,
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "notes": notes or "",
        "brand": branding.public(),     # company name, colors, logo (Settings > Branding)
        "basis": _basis(res, inputs),
        "facilities": _facilities(res),
        "summary": results_summary(res),
        "series": results_series(res, max_points=800),
        "profile": _profile(res),
        "fill": _fill_table(res),
    }


def render_html(data: dict) -> str:
    with open(os.path.join(paths.static_dir(), "report_template.html"), "r", encoding="utf-8") as f:
        tpl = f.read()
    # No "<" inside the inline <script>, whatever the scenario name or notes say ("<" only
    # occurs inside JSON strings, where \u003c reads back as the same character).
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    title = (data.get("title") or "Purge simulation").replace("&", "&amp;").replace("<", "&lt;")
    fill = {"__REPORT_TITLE__": title, "__REPORT_DATA__": blob}
    return re.sub("__REPORT_(?:TITLE|DATA)__", lambda m: fill[m.group(0)], tpl)   # one pass


def export_client_html(res: SimResults, path: str, scenario_name: str, inputs=None,
                       notes: str = "", date: Optional[str] = None) -> None:
    html = render_html(build_data(res, scenario_name, inputs=inputs, notes=notes, date=date))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
