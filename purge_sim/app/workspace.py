"""
The app's working state and every operation on it.

One Workspace holds the scenario being edited, the last simulation results and
the run job. Both the web UI (through server.py) and the Claude assistant
(through assistant.py) act only through these methods, so the assistant can
never do anything the user couldn't do by hand, and the same validation and
hard-rule checks apply to both.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import re
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..data.elevation import DEFAULT_SPACING_FT, ElevationError, describe_source, fetch_route_elevation
from ..data.scenario import Scenario, ScenarioInputs, load_scenario, save_scenario
from ..engine.config_builder import build_sim_config
from ..engine.simulator import SimResults, simulate
from . import paths


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

# Hard engineering rule 4 (CLAUDE.md): boosters never draw below 100 psi without
# Kevin's explicit sign-off for that job. No edit through the app (UI or assistant) can
# set a value below the floor; a job with the sign-off carries it in its scenario JSON.
BOOSTER_SUCTION_FLOOR_PSIG = 100.0

EXIT_BEHAVIORS = ("taper_last_n_miles", "step_last_n_miles", "linear_ramp",
                  "constant_run", "constant_end")

# Large arrays that come from data import, not from hand edits.
_IMPORT_ONLY_FIELDS = {"elevation_profile", "mop_joints", "elevation_source", "route_latlon", "landmarks"}

_LIST_ITEM_KEYS = {
    "pipe_segments":    {"required": {"start_mp", "end_mp", "od_in", "wt_in"}, "numeric": {"start_mp", "end_mp", "od_in", "wt_in"}},
    "check_valves":     {"required": {"mp", "name"}, "numeric": {"mp"}},
    "pump_stations":    {"required": {"mp", "name", "suction_psig"}, "numeric": {"mp", "suction_psig"}},
    "booster_stations": {"required": {"mp", "name"}, "numeric": {"mp", "discharge_psig", "suction_min_psig", "max_flow_scfm"}},
}

_INPUT_FIELDS = {f.name: f for f in dataclasses.fields(ScenarioInputs)}


class InputError(ValueError):
    """A rejected edit. The message is shown to the user / assistant as-is."""


def _num(name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise InputError(f"{name} must be a number, got {v!r}")
    v = float(v)
    if not np.isfinite(v):
        raise InputError(f"{name} must be finite")
    return v


def _coerce(name: str, value: Any, current: Any) -> Any:
    f = _INPUT_FIELDS[name]
    default = f.default if f.default is not dataclasses.MISSING else None
    type_str = str(f.type)

    if name in _LIST_ITEM_KEYS:
        if not isinstance(value, list):
            raise InputError(f"{name} must be a list of objects")
        spec = _LIST_ITEM_KEYS[name]
        out = []
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise InputError(f"{name}[{i}] must be an object")
            missing = spec["required"] - set(item)
            if missing:
                raise InputError(f"{name}[{i}] is missing {sorted(missing)}")
            clean = dict(item)
            for k in spec["numeric"]:
                if k in clean and clean[k] is not None:
                    clean[k] = _num(f"{name}[{i}].{k}", clean[k])
            if "name" in clean:
                clean["name"] = str(clean["name"])
            out.append(clean)
        return out

    if name == "deployed_booster_mps":
        if value is None:
            return None
        if not isinstance(value, list):
            raise InputError("deployed_booster_mps must be a list of mileposts or null")
        return [_num("deployed_booster_mps[]", v) for v in value]

    if name == "bpcv":
        if value is None:
            return None
        if not isinstance(value, dict) or not {"mp", "elevation_ft"} <= set(value):
            raise InputError("bpcv must be null or an object with mp and elevation_ft (name optional)")
        clean = dict(value)
        clean["mp"] = _num("bpcv.mp", clean["mp"])
        clean["elevation_ft"] = _num("bpcv.elevation_ft", clean["elevation_ft"])
        return clean

    if name == "exit_pressure_behavior":
        if value not in EXIT_BEHAVIORS:
            raise InputError(f"exit_pressure_behavior must be one of {list(EXIT_BEHAVIORS)}")
        return value

    if isinstance(default, bool) or "bool" in type_str:
        if not isinstance(value, bool):
            raise InputError(f"{name} must be true or false")
        return value
    if isinstance(default, str):
        return str(value)
    if isinstance(default, int) and not isinstance(default, bool):
        v = _num(name, value)
        if v != int(v):
            raise InputError(f"{name} must be a whole number")
        return int(v)
    if "Optional" in type_str or default is None:
        return None if value is None else _num(name, value)
    return _num(name, value)


def _check_rules(old: ScenarioInputs, new: ScenarioInputs, changed: set) -> None:
    """Invariants an edit must not break. Only the fields being changed are checked, so a
    scenario whose JSON already carries a signed-off exception (e.g. a lower booster suction
    floor for that job) stays editable without the app re-litigating that decision."""
    def frac_ok(v):
        return v is None or 0.0 < v <= 1.0

    if changed & {"purge_start_mp", "purge_end_mp"} and new.purge_end_mp <= new.purge_start_mp:
        raise InputError("purge_end_mp must be greater than purge_start_mp")
    if "drive_ceiling_fraction" in changed and not frac_ok(new.drive_ceiling_fraction):
        raise InputError("drive_ceiling_fraction must be in (0, 1]")
    if "drive_mop_fraction" in changed and not frac_ok(new.drive_mop_fraction):
        raise InputError("drive_mop_fraction must be in (0, 1]")
    if changed & {"min_speed_mph", "max_speed_mph", "target_speed_mph"}:
        if new.min_speed_mph <= 0 or new.max_speed_mph < new.min_speed_mph:
            raise InputError("speeds must satisfy 0 < min_speed_mph <= max_speed_mph")
    if "dt_hr" in changed and new.dt_hr <= 0:
        raise InputError("dt_hr must be positive")

    floor = BOOSTER_SUCTION_FLOOR_PSIG
    if "spread_suction_min_psig" in changed and new.spread_suction_min_psig < floor:
        raise InputError(
            f"spread_suction_min_psig {new.spread_suction_min_psig:g} is below the "
            f"{floor:g} psi booster suction floor (hard rule 4). Going lower needs explicit "
            f"engineering sign-off for this job and is not something the app will do on its own.")
    if "booster_stations" in changed:
        before = {(b.get("name"), b.get("suction_min_psig")) for b in old.booster_stations}
        for b in new.booster_stations:
            s = b.get("suction_min_psig")
            if s is not None and s < floor and (b.get("name"), s) not in before:
                raise InputError(
                    f"booster {b.get('name')} suction_min_psig {s:g} is below the {floor:g} psi "
                    f"floor (hard rule 4); needs explicit engineering sign-off for this job.")


def apply_changes(inputs: ScenarioInputs, changes: Dict[str, Any]) -> tuple[ScenarioInputs, List[dict]]:
    """Validate `changes` against `inputs`; return (new inputs, diff). Raises InputError."""
    if not isinstance(changes, dict) or not changes:
        raise InputError("changes must be a non-empty object of field -> value")
    data = dataclasses.asdict(inputs)
    diff = []
    for name, value in changes.items():
        if name not in _INPUT_FIELDS:
            raise InputError(f"unknown input field {name!r}")
        if name in _IMPORT_ONLY_FIELDS:
            raise InputError(f"{name} comes from data import and can't be edited here")
        new = _coerce(name, value, data[name])
        if new != data[name]:
            diff.append({"field": name, "old": _brief(data[name]), "new": _brief(new)})
        data[name] = new
    updated = ScenarioInputs(**data)
    _check_rules(inputs, updated, set(changes))
    return updated, diff


def _brief(v: Any) -> Any:
    if isinstance(v, list) and len(v) > 12:
        return f"[{len(v)} items]"
    return v


# ---------------------------------------------------------------------------
# Summaries (compact, JSON-safe views for the UI and the assistant)
# ---------------------------------------------------------------------------

def _f(v: Any, nd: int = 3) -> Any:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    if not np.isfinite(x):
        return None
    return round(x, nd)


def downsample_idx(n: int, max_points: int) -> List[int]:
    if n <= max_points:
        return list(range(n))
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return sorted(set(int(i) for i in idx))


def profile_stats(elev: List[list], start_mp: float, end_mp: float) -> dict:
    if not elev:
        return {"points": 0}
    a = np.asarray(elev, dtype=float)
    span = a[(a[:, 0] >= start_mp) & (a[:, 0] <= end_mp)]
    if len(span) == 0:
        span = a
    i_hi, i_lo = int(np.argmax(span[:, 1])), int(np.argmin(span[:, 1]))
    return {
        "points": int(len(a)),
        "mp_range": [_f(a[0, 0]), _f(a[-1, 0])],
        "elev_start_ft": _f(np.interp(start_mp, a[:, 0], a[:, 1]), 1),
        "elev_end_ft": _f(np.interp(end_mp, a[:, 0], a[:, 1]), 1),
        "elev_max_ft": _f(span[i_hi, 1], 1), "elev_max_mp": _f(span[i_hi, 0]),
        "elev_min_ft": _f(span[i_lo, 1], 1), "elev_min_mp": _f(span[i_lo, 0]),
    }


def mop_stats(joints: List[dict]) -> dict:
    if not joints:
        return {"joints": 0}
    m = np.array([j["mop_psig"] for j in joints], dtype=float)
    i = int(np.argmin(m))
    return {"joints": len(joints), "mop_min_psig": _f(m.min(), 1), "mop_min_mp": _f(joints[i]["mp"]),
            "mop_max_psig": _f(m.max(), 1), "mop_median_psig": _f(np.median(m), 1)}


def _setup_status(sc: Scenario) -> dict:
    from . import intake
    try:
        v = intake.view(sc)
    except Exception as e:   # a summary must never fail because of the setup view
        return {"error": f"{type(e).__name__}: {e}"}
    return {"answered": sum(q["status"] == "answered" for q in v["questions"]), "missing": v["missing"],
            "fresh_import": v["fresh_import"]}


def inputs_summary(sc: Scenario) -> dict:
    inp = sc.inputs
    scalars = {k: v for k, v in dataclasses.asdict(inp).items()
               if k not in _IMPORT_ONLY_FIELDS and k not in _LIST_ITEM_KEYS
               and k not in ("bpcv", "deployed_booster_mps")}
    return {
        "meta": {"name": sc.meta.name, "notes": sc.meta.notes, "source_files": sc.meta.source_files,
                 "modified_at": sc.meta.modified_at},
        "job_setup": _setup_status(sc),
        "scalars": scalars,
        "pipe_segments": inp.pipe_segments,
        "check_valves": inp.check_valves,
        "pump_stations": inp.pump_stations,
        "booster_stations": inp.booster_stations,
        "deployed_booster_mps": inp.deployed_booster_mps,
        "bpcv": inp.bpcv,
        "elevation_profile": profile_stats(inp.elevation_profile, inp.purge_start_mp, inp.purge_end_mp),
        "elevation_source": inp.elevation_source,
        "route_points": len(inp.route_latlon),
        "mop_joints": mop_stats(inp.mop_joints),
    }


def results_summary(res: SimResults) -> dict:
    steps = res.steps
    out = {
        "completed": res.completed,
        "abort_reason": res.abort_reason,
        "steps": len(steps),
        "wall_time_s": _f(res.wall_time_s, 1),
        "n2_injected_scf": _f(res.total_scf_injected, 0),
        "n2_booster_fresh_scf": _f(res.total_scf_booster_fresh, 0),
        "n2_total_fresh_scf": _f(res.total_scf_n2, 0),
        "n2_vented_scf": _f(res.total_scf_vented, 0),
    }
    if steps:
        faces = np.array([s.pig_face_psig for s in steps])
        speeds = np.array([s.pig_speed_mph for s in steps])
        inj_p = np.array([s.injection_psig for s in steps])
        slack = [s for s in steps if s.slack_line_risk]
        cfg = res.config
        span = cfg.purge_end_mp - cfg.purge_start_mp
        worst = min((s for s in steps if s.worst_mop_margin is not None),
                    key=lambda s: s.worst_mop_margin, default=None)
        out.update({
            "duration_hr": _f(steps[-1].t_hr, 2),
            "final_mp": _f(steps[-1].pig_mp),
            "percent_of_route": _f(100 * (steps[-1].pig_mp - cfg.purge_start_mp) / span if span else 0, 1),
            "pig_face_psig": {"avg": _f(faces.mean(), 1), "max": _f(faces.max(), 1), "min": _f(faces.min(), 1)},
            "pig_speed_mph": {"avg": _f(speeds.mean()), "max": _f(speeds.max()), "min": _f(speeds.min())},
            "injection_psig_max": _f(inj_p.max(), 1),
            "mop_violation_steps": int(sum(1 for s in steps if s.mop_violations)),
            "mop_violations_total": int(sum(s.mop_violations for s in steps)),
            "mop_warning_steps": int(sum(1 for s in steps if s.mop_warnings)),
            "worst_mop_margin_psig": _f(worst.worst_mop_margin, 1) if worst else None,
            "worst_mop_mp": _f(worst.worst_mop_mp) if worst else None,
            "slack_line_risk_steps": len(slack),
            "slack_line_risk_mp_range": [_f(slack[0].pig_mp), _f(slack[-1].pig_mp)] if slack else None,
            "meter_valve_steps": int(sum(1 for s in steps if s.meter_valve_active)),
        })
        flags = []
        if res.total_scf_vented > 0:
            flags.append(f"N2 vented ({res.total_scf_vented:,.0f} SCF): red flag under hard rule 1, fix by control")
        if slack:
            flags.append(f"slack-line risk on {len(slack)} steps (hard rule 2)")
        if out["mop_violation_steps"]:
            flags.append(f"MOP violations on {out['mop_violation_steps']} steps")
        if not res.completed:
            flags.append(f"run did not complete: {res.abort_reason}")
        out["flags"] = flags
    if res.booster_plan is not None:
        bp = res.booster_plan
        out["booster_plan"] = {"sites_mp": [_f(m) for m in bp.sites_mp], "notes": list(bp.notes),
                               "reasons": {str(_f(k)): v for k, v in bp.reasons.items()}}
    if res.spread_events:
        out["spread_events"] = [{k: _f(v, 2) if isinstance(v, float) else v for k, v in e.items()}
                                for e in res.spread_events]
    if res.roadmap is not None:
        try:
            out["roadmap_report"] = res.roadmap.report()
        except Exception:
            pass
    return out


def results_series(res: SimResults, max_points: int = 1500) -> dict:
    steps = res.steps
    idx = downsample_idx(len(steps), max_points)
    pick = [steps[i] for i in idx]
    return {
        "step_index": idx,
        "t_hr": [_f(s.t_hr, 4) for s in pick],
        "pig_mp": [_f(s.pig_mp) for s in pick],
        "pig_speed_mph": [_f(s.pig_speed_mph) for s in pick],
        "pig_face_psig": [_f(s.pig_face_psig, 1) for s in pick],
        "injection_psig": [_f(s.injection_psig, 1) for s in pick],
        "injection_scfm": [_f(s.injection_scfm, 0) for s in pick],
        "total_scf": [_f(s.total_scf, 0) for s in pick],
        "exit_psig": [_f(s.exit_psig, 1) for s in pick],
        "mop_violations": [int(s.mop_violations) for s in pick],
        "slack_line_risk": [bool(s.slack_line_risk) for s in pick],
        "boosters_active": [sum(1 for b in s.booster_states if b.get("running"))
                            for s in pick],
    }


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_name(name: str) -> str:
    name = _SAFE_NAME.sub("_", (name or "").strip()).strip(" .")
    if not name:
        raise InputError("a name is required")
    return name[:80]


class Workspace:
    def __init__(self):
        self.lock = threading.RLock()
        self.scenario: Optional[Scenario] = None
        self.scenario_id: Optional[str] = None
        self.dirty = False
        self.results: Optional[SimResults] = None
        self.results_for: Optional[str] = None   # scenario name the results belong to
        self.results_stale = False
        self.job = {"state": "idle", "progress": 0.0, "message": "", "error": None, "started": None}
        self.revision = 0   # bumps on every change, so the UI knows to refresh

    # ---- scenario library -------------------------------------------------

    def _roots(self) -> Dict[str, str]:
        return {"bundled": paths.bundled_scenarios_dir(), "user": paths.user_scenarios_dir()}

    def list_scenarios(self) -> List[dict]:
        out = []
        for origin, root in self._roots().items():
            if not os.path.isdir(root):
                continue
            for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                except (OSError, ValueError):
                    continue
                if not (isinstance(raw, dict) and "inputs" in raw):
                    continue   # loose elevation / route-meta files, not scenarios
                rel = os.path.relpath(path, root).replace("\\", "/")
                meta = raw.get("meta", {})
                inp = raw.get("inputs", {})
                out.append({
                    "id": f"{origin}:{rel}",
                    "origin": origin,
                    "job": rel.split("/")[0] if "/" in rel else "",
                    "name": meta.get("name") or os.path.splitext(os.path.basename(rel))[0],
                    "file": rel,
                    "notes": (meta.get("notes") or "")[:400],
                    "modified_at": meta.get("modified_at", ""),
                    "span_mi": _f((inp.get("purge_end_mp") or 0) - (inp.get("purge_start_mp") or 0), 2),
                    "fluid": inp.get("fluid_name", ""),
                })
        return out

    def _resolve(self, scenario_id: str) -> str:
        origin, _, rel = (scenario_id or "").partition(":")
        root = self._roots().get(origin)
        if not root or not rel:
            raise InputError(f"unknown scenario id {scenario_id!r}")
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(path):
            raise InputError(f"scenario {scenario_id!r} not found")
        return path

    def scenario_path(self, scenario_id: str) -> str:
        """The library file behind a scenario id; raises InputError if it isn't one."""
        return self._resolve(scenario_id)

    def load(self, scenario_id: str) -> dict:
        if self.job["state"] == "running":
            raise InputError("wait for the running simulation to finish before opening another scenario")
        sc = load_scenario(self._resolve(scenario_id))
        with self.lock:
            self.scenario, self.scenario_id, self.dirty = sc, scenario_id, False
            self.results, self.results_for, self.results_stale = None, None, False
            self.revision += 1
        return inputs_summary(sc)

    def require(self) -> Scenario:
        if self.scenario is None:
            raise InputError("no scenario is open; load one from the library or import a data file first")
        return self.scenario

    def update_inputs(self, changes: Dict[str, Any]) -> List[dict]:
        with self.lock:
            sc = self.require()
            new_inputs, diff = apply_changes(sc.inputs, changes)
            if diff:
                sc.inputs = new_inputs
                self.dirty = True
                self.results_stale = self.results is not None
                self.revision += 1
            return diff

    def update_meta(self, name: Optional[str] = None, notes: Optional[str] = None) -> None:
        with self.lock:
            sc = self.require()
            if name is not None:
                sc.meta.name = name.strip() or sc.meta.name
            if notes is not None:
                sc.meta.notes = notes
            self.dirty = True
            self.revision += 1

    def save_as(self, name: str, notes: Optional[str] = None, overwrite: bool = False) -> dict:
        """Save to the user's scenario folder. Bundled scenarios are never written."""
        with self.lock:
            sc = self.require()
            fname = safe_name(name)
            if not fname.lower().endswith(".json"):
                fname += ".json"
            root = paths.user_scenarios_dir()
            path = os.path.join(root, fname)
            if os.path.exists(path) and not overwrite:
                raise InputError(f"{fname} already exists in your scenarios folder; pick a new name")
            sc.meta.name = os.path.splitext(fname)[0]
            if notes is not None:
                sc.meta.notes = notes
            save_scenario(sc, path)
            self.scenario_id, self.dirty = f"user:{fname}", False
            self.revision += 1
            return {"id": self.scenario_id, "path": path}

    def _busy_job(self, message: str) -> None:
        with self.lock:
            if self.job["state"] == "running":
                raise InputError("wait for the running job to finish first")
            self.job = {"state": "running", "progress": 0.0, "message": message, "error": None,
                        "started": time.time()}
            self.revision += 1

    def _job_progress(self, frac: float, message: str) -> None:
        self.job["progress"] = float(frac)
        self.job["message"] = message

    def _idle_job(self) -> None:
        with self.lock:
            self.job = {"state": "idle", "progress": 0.0, "message": "", "error": None, "started": None}
            self.revision += 1

    def new_from_import(self, file_path: str, kind: str = "auto", name: str = "",
                        fetch_elevation: Optional[bool] = None,
                        spacing_ft: float = DEFAULT_SPACING_FT) -> dict:
        """Build a new scenario from a data file. A file without elevations gets them from
        USGS 3DEP here (progress shows as the job), unless settings say to ask first."""
        from .importers import scenario_from_file
        self._busy_job("Importing " + os.path.basename(file_path))
        try:
            sc, warnings = scenario_from_file(file_path, kind, name, fetch_elevation, spacing_ft,
                                              progress_cb=self._job_progress)
        except ElevationError as e:
            raise InputError(str(e)) from e
        finally:
            self._idle_job()
        with self.lock:
            self.scenario, self.scenario_id, self.dirty = sc, None, True
            self.results, self.results_for, self.results_stale = None, None, False
            self.revision += 1
        from ..data.formats import FORMAT_LABELS
        fmt = (sc.meta.intake or {}).get("format", "")
        return {**inputs_summary(sc), "warnings": warnings, "format": fmt,
                "format_label": FORMAT_LABELS.get(fmt, "")}

    def fetch_elevation(self, spacing_ft: float = DEFAULT_SPACING_FT, reverse: bool = False) -> dict:
        """Look the open scenario's route up in USGS 3DEP again and replace its elevation profile.
        reverse=True flips the route so mileposts run from the other end."""
        sc = self.require()
        route = sc.inputs.route_latlon
        if len(route) < 2:
            raise InputError("this scenario has no stored route coordinates (only KMZ and GPS imports "
                             "carry them); re-import the KMZ to look its elevation up")
        if reverse:
            route = route[::-1]
        lat = [p[0] for p in route]
        lon = [p[1] for p in route]
        self._busy_job("Looking up elevation")
        try:
            res = fetch_route_elevation(lat, lon, spacing_ft=spacing_ft, progress_cb=self._job_progress)
        except ElevationError as e:
            raise InputError(str(e)) from e
        finally:
            self._idle_job()
        warnings = list(res.flags)
        with self.lock:
            inp = sc.inputs
            old_len = float(inp.elevation_profile[-1][0]) if inp.elevation_profile else None
            inp.elevation_profile = res.profile()
            inp.elevation_source = res.source
            if reverse:
                inp.route_latlon = route
                inp.purge_start_mp, inp.purge_end_mp = 0.0, round(res.length_mi, 3)
                if inp.check_valves or inp.pump_stations or inp.booster_stations or inp.bpcv:
                    warnings.append("Mileposts now run from the other end. Station, check valve, booster and "
                                    "BPCV mileposts were not changed; re-check them.")
            else:
                inp.purge_start_mp = max(0.0, min(inp.purge_start_mp, res.length_mi))
                inp.purge_end_mp = min(inp.purge_end_mp, round(res.length_mi, 3))
            if "+USGS" not in inp.data_source and inp.data_source:
                inp.data_source += "+USGS"
            line = describe_source(res.source)
            sc.meta.notes = (sc.meta.notes.rstrip() + "\n" + line).strip() if sc.meta.notes else line
            self.dirty = True
            self.results_stale = self.results is not None
            self.revision += 1
        return {"points": len(res.mileposts), "length_mi": _f(res.length_mi, 3),
                "previous_length_mi": _f(old_len, 3) if old_len is not None else None,
                "filled_points": res.filled_points, "source": res.source, "warnings": warnings,
                "purge_start_mp": sc.inputs.purge_start_mp, "purge_end_mp": sc.inputs.purge_end_mp}

    # ---- job intake ------------------------------------------------------------

    def intake_view(self) -> dict:
        from . import intake
        with self.lock:
            return intake.view(self.require())

    def apply_intake(self, answers: Dict[str, Any]) -> dict:
        """Apply job intake answers (intake.py) to the open scenario: builds the inputs they
        describe, records the answers and assumptions on the scenario and refreshes the
        Job setup block in its notes. Returns what changed, assumptions, warnings, the
        questions still missing and the pre-run check."""
        from . import intake
        with self.lock:
            sc = self.require()
            if self.job["state"] == "running":
                raise InputError("wait for the running job to finish first")
            try:
                new_inputs, meta_intake, report = intake.apply(sc, answers)
            except intake.IntakeError as e:
                raise InputError(str(e)) from e
            old = dataclasses.asdict(sc.inputs)
            new = dataclasses.asdict(new_inputs)
            _check_rules(sc.inputs, new_inputs, {k for k in new if new[k] != old[k]})
            diff = [{"field": k, "old": _brief(old[k]), "new": _brief(new[k])}
                    for k in new if new[k] != old[k]]
            sc.inputs = new_inputs
            sc.meta.intake = meta_intake
            sc.meta.notes = intake.merge_notes(sc.meta.notes, intake.notes_block(new_inputs, report))
            self.dirty = True
            self.results_stale = self.results is not None
            self.revision += 1
            return {"changed": diff, **report}

    def precheck(self) -> dict:
        from ..data import fluids
        from ..engine.precheck import precheck
        with self.lock:
            sc = self.require()
            key = ((sc.meta.intake or {}).get("answers") or {}).get("fluid") or fluids.match(sc.inputs.fluid_name)
            vmin = fluids.FLUIDS[key].min_liquid_psig if key in fluids.FLUIDS else None
            try:
                return precheck(sc.inputs, vmin)
            except ValueError as e:
                raise InputError(str(e)) from e

    # ---- simulation ------------------------------------------------------------

    def run(self, progress_cb: Optional[Callable[[float], None]] = None) -> SimResults:
        """Run synchronously. Used by the assistant and by start_run's worker."""
        with self.lock:
            sc = self.require()
            if self.job["state"] == "running" and threading.current_thread().name != "sim-run":
                raise InputError("a simulation is already running")
            cfg = build_sim_config(sc.inputs)
            name = sc.meta.name
            self.job = {"state": "running", "progress": 0.0, "message": f"Simulating {name}",
                        "error": None, "started": time.time()}
            self.revision += 1

        def _cb(frac: float):
            self.job["progress"] = float(frac)
            if progress_cb:
                progress_cb(frac)

        try:
            res = simulate(cfg, progress_cb=_cb)
        except Exception as e:
            with self.lock:
                self.job.update(state="error", error=f"{type(e).__name__}: {e}",
                                message=traceback.format_exc(limit=3))
                self.revision += 1
            raise
        with self.lock:
            self.results, self.results_for, self.results_stale = res, name, False
            self.job.update(state="done", progress=1.0,
                            message="Complete" if res.completed else f"Aborted: {res.abort_reason}")
            self.revision += 1
        return res

    def start_run(self) -> None:
        with self.lock:
            self.require()
            if self.job["state"] == "running":
                raise InputError("a simulation is already running")
            self.job = {"state": "running", "progress": 0.0, "message": "Starting", "error": None,
                        "started": time.time()}

        def _worker():
            try:
                self.run()
            except Exception:
                pass   # recorded in self.job

        threading.Thread(target=_worker, name="sim-run", daemon=True).start()

    def require_results(self) -> SimResults:
        if self.results is None:
            raise InputError("no results yet; run the simulation first")
        return self.results

    # ---- result views ------------------------------------------------------------

    def profile_at(self, step_index: int, max_points: int = 1200) -> dict:
        """Full-route HGL at one timestep: the live Pipeline Profile view."""
        from ..engine.hgl import compute_hgl
        res = self.require_results()
        cfg = res.config
        step_index = max(0, min(int(step_index), len(res.steps) - 1))
        step = res.steps[step_index]
        h = compute_hgl(step, cfg)
        idx = downsample_idx(len(h.mp), max_points)
        mop_mp, mop_p = [], []
        if cfg.mop_joints:
            j_idx = downsample_idx(len(cfg.mop_joints), max_points)
            mop_mp = [_f(cfg.mop_joints[i].mp) for i in j_idx]
            mop_p = [_f(cfg.mop_joints[i].mop_psig, 1) for i in j_idx]
        return {
            "step_index": step_index,
            "t_hr": _f(step.t_hr, 3),
            "pig_mp": _f(step.pig_mp),
            "pig_speed_mph": _f(step.pig_speed_mph),
            "pig_face_psig": _f(step.pig_face_psig, 1),
            "exit_mp": _f(step.exit_mp), "exit_psig": _f(step.exit_psig, 1),
            "exit_description": step.exit_description,
            "slack_line_risk": bool(step.slack_line_risk),
            "mp": [_f(h.mp[i]) for i in idx],
            "elevation_ft": [_f(h.elevation_ft[i], 1) for i in idx],
            "pressure_psig": [_f(h.pressure_psig[i], 1) for i in idx],
            "is_gas": [bool(h.is_gas[i]) for i in idx],
            "mop_mp": mop_mp, "mop_psig": mop_p,
            "stations": [{"mp": _f(s["mp"]), "name": s.get("name", ""), "status": s.get("status", ""),
                          "discharge_psig": _f(s.get("discharge_psig"), 1)}
                         for s in step.station_pressures],
            "boosters": [{"mp": _f(b["mp"]), "name": b.get("name", ""),
                          **{k: (_f(v, 1) if isinstance(v, float) else v) for k, v in b.items()
                             if k not in ("mp", "name") and isinstance(v, (int, float, bool, str))}}
                         for b in step.booster_states],
        }

    def map_view(self) -> dict:
        """Route geometry, mileposts and infrastructure placed on the map (gis.py)."""
        from . import gis
        with self.lock:
            return gis.map_view(self.require().inputs)

    def attach_route(self, file_path: str, direction: str = "auto") -> dict:
        """Add (or replace) the open scenario's route coordinates from a KMZ / KML / GPS file,
        for the Map tab. Elevation profile, mileposts and stations are not changed."""
        from . import gis
        with self.lock:
            sc = self.require()
            try:
                route, notes = gis.route_from_file(file_path, sc.inputs, direction)
            except gis.RouteError as e:
                raise InputError(str(e)) from e
            replaced = len(sc.inputs.route_latlon) >= 2
            sc.inputs.route_latlon = route
            fname = os.path.basename(file_path)
            if fname not in sc.meta.source_files:
                sc.meta.source_files.append(fname)
            line = f"Route coordinates for the map from {fname}" + (" (replaced the earlier route)." if replaced else ".")
            sc.meta.notes = (sc.meta.notes.rstrip() + "\n" + line).strip() if sc.meta.notes else line
            self.dirty = True
            self.revision += 1
            view = gis.map_view(sc.inputs)
        return {"route_points": len(route), "notes": notes + view["warnings"],
                "route_length_mi": view.get("route_length_mi"), "mp_scale": view.get("mp_scale")}

    def elevation_view(self, start_mp: Optional[float] = None, end_mp: Optional[float] = None,
                       max_points: int = 400) -> dict:
        sc = self.require()
        a = np.asarray(sc.inputs.elevation_profile, dtype=float)
        if a.size == 0:
            return {"mp": [], "elevation_ft": [], "source": sc.inputs.elevation_source}
        lo = sc.inputs.purge_start_mp if start_mp is None else start_mp
        hi = sc.inputs.purge_end_mp if end_mp is None else end_mp
        a = a[(a[:, 0] >= lo) & (a[:, 0] <= hi)]
        idx = downsample_idx(len(a), max_points)
        return {"mp": [_f(a[i, 0]) for i in idx], "elevation_ft": [_f(a[i, 1], 1) for i in idx],
                "source": sc.inputs.elevation_source}

    def state(self) -> dict:
        with self.lock:
            sc = self.scenario
            return {
                "revision": self.revision,
                "scenario_id": self.scenario_id,
                "scenario_name": sc.meta.name if sc else None,
                "dirty": self.dirty,
                "has_results": self.results is not None,
                "results_for": self.results_for,
                "results_stale": self.results_stale,
                "job": {k: v for k, v in self.job.items() if k != "started"},
            }

    # ---- exports ------------------------------------------------------------------

    def export(self, kind: str) -> str:
        """Write a deliverable to a fresh, timestamped folder; return its path."""
        res = self.require_results()
        name = safe_name(self.results_for or "run")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(paths.outputs_dir(), name, stamp)
        os.makedirs(out_dir, exist_ok=True)
        sc = self.scenario
        pi = {"project": name, "notes": sc.meta.notes if sc else "",
              "date": datetime.now().strftime("%Y-%m-%d")}
        if kind == "formatted":
            from ..engine.formatted_report import export_formatted_report
            path = os.path.join(out_dir, f"{name}_formatted.xlsx")
            export_formatted_report(res, path, scenario_name=name, project_info=pi)
        elif kind == "full":
            from ..engine.purge_report import export_purge_report
            path = os.path.join(out_dir, f"{name}_report.xlsx")
            export_purge_report(res, path, scenario_name=name, project_info=pi)
        elif kind == "log":
            from ..engine.log_export import export_run_log
            path = os.path.join(out_dir, f"{name}_log.txt")
            export_run_log(res, path, scenario_name=name)
        elif kind == "gif":
            from ..engine.animation import generate_animation
            path = os.path.join(out_dir, f"{name}_profile.gif")
            generate_animation(res, res.config, out_gif=path, out_mp4=None, scenario_name=name)
        else:
            raise InputError(f"unknown export kind {kind!r}")
        return path
