"""
Job intake: the questions every new purge has to answer, and turning the answers
into simulator inputs.

Until now this was done in a Claude Code session for every job: read the request
("purging this diesel line from MP 0 to MP 35 to tankage, it's 6 inch, drive
capped at 500, MOP 900 at all points"), look up what wasn't said (6" -> 6.625" OD,
a wall thickness, diesel's SG and viscosity), decide which way the pig runs, where
the liquid really leaves the line, how MOP is defined, and write all of it into
the scenario JSON with the assumptions in the notes. This module is that step as
code, shared by the Job setup form in the app and the assistant:

    view(sc)            -> every question with its current answer, the default that
                           applies if it stays blank, and which required ones are missing
    apply(sc, answers)  -> new ScenarioInputs + a report (what changed, every
                           assumption made, warnings, the pre-run check)

Answers live in scenario.meta.intake["answers"] so a saved scenario carries its
setup. A blank answer means "use the default"; every default that isn't grounded
in the imported data is listed as an assumption to confirm, and lands in the
scenario notes under a "Job setup" block.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data import fluids as fluid_lib
from ..data import pipe_catalog as pipes
from ..data.scenario import Scenario, ScenarioInputs
from ..engine.constants import NPS_OD_IN
from ..engine.precheck import exit_pressure_at_stop, mop_at, precheck

NOTES_BEGIN = "--- Job setup ---"
NOTES_END = "--- end job setup ---"

# NPS 2 and up (the smallest line pipe any job has used is 3.5" OD)
_NPS_CHOICES = [k for k in NPS_OD_IN if NPS_OD_IN[k] >= 2.375]

# kind: number | select | bool. required: no default can stand in for it on a new job.
QUESTIONS: List[dict] = [
    # ---- route
    {"id": "reverse_route", "section": "Route", "kind": "bool",
     "label": "Pig runs from the file's far end back toward its start",
     "help": "KMZ routes are often drawn the other way (Tipville: inverted so MP increases southbound). "
             "Flips the profile so MP 0 is the launch; every milepost below is then in the new direction."},
    {"id": "purge_start_mp", "section": "Route", "kind": "number", "unit": "MP", "label": "Pig launch milepost"},
    {"id": "purge_end_mp", "section": "Route", "kind": "number", "unit": "MP", "label": "Pig stop milepost (purge end)"},
    {"id": "fluid_exit_mp", "section": "Route", "kind": "number", "unit": "MP",
     "label": "Where the liquid actually leaves the line",
     "help": "Leave blank if it's the pig stop. If the product goes on past the pig stop (Laurel: pig stops at "
             "the Allendale BV, diesel goes on to the Billings tank farm), the stretch beyond is still pushed "
             "through, so the pig stop sees the delivery pressure plus that stretch's friction and head."},
    # ---- exit
    {"id": "exit_type", "section": "Exit", "kind": "select", "label": "What receives the liquid",
     "options": [{"value": "tankage", "label": "Tankage"},
                 {"value": "pressurized", "label": "Pressurized receipt (sphere, pipeline, flare header)"}]},
    {"id": "exit_pressure_psig", "section": "Exit", "kind": "number", "unit": "psig",
     "label": "Delivery pressure held at the exit",
     "help": "Past tankage jobs held 50 psig. Volatile products need vapor pressure + 100 psi."},
    # ---- pipe
    {"id": "nps", "section": "Pipe", "kind": "select", "label": "Nominal pipe size (in)", "required": True,
     "options": [{"value": k, "label": f'{k}" ({NPS_OD_IN[k]:.3f}" OD)'} for k in _NPS_CHOICES],
     "help": "Sets one uniform pipe size over the whole route. For a line that changes size, set the size "
             "here, then edit the pipe segments on the Inputs tab."},
    {"id": "wt_in", "section": "Pipe", "kind": "number", "unit": "in", "label": "Wall thickness",
     "help": "Blank uses the standard (STD) wall for the size, which is an assumption to confirm."},
    {"id": "grade", "section": "Pipe", "kind": "select", "label": "Pipe grade (API 5L)",
     "options": [{"value": g, "label": g} for g in pipes.GRADE_SMYS_PSI],
     "help": "Used for the Barlow check of the MOP. Blank reports the minimum grade the MOP needs."},
    # ---- fluid
    {"id": "fluid", "section": "Fluid", "kind": "select", "label": "Product", "required": True,
     "options": fluid_lib.options()},
    {"id": "api_gravity", "section": "Fluid", "kind": "number", "unit": "API", "label": "API gravity (crude)",
     "help": "Sets the specific gravity. Leave blank to use the library value."},
    {"id": "fluid_sg", "section": "Fluid", "kind": "number", "label": "Specific gravity",
     "help": "Blank uses the product library value (or the API gravity)."},
    {"id": "fluid_viscosity_cst", "section": "Fluid", "kind": "number", "unit": "cSt", "label": "Viscosity",
     "help": "Blank uses the product library value."},
    # ---- MOP
    {"id": "mop_basis", "section": "MOP", "kind": "select", "label": "How MOP is defined",
     "options": [{"value": "flat", "label": "One MOP for the whole line"},
                 {"value": "flange_class", "label": "Flange class rating"},
                 {"value": "from_data", "label": "Point by point from the imported data"}]},
    {"id": "mop_psig", "section": "MOP", "kind": "number", "unit": "psig", "label": "MOP", "required": True},
    {"id": "flange_class", "section": "MOP", "kind": "select", "label": "Flange class",
     "options": [{"value": k, "label": f"ANSI {k} ({v:,.0f} psig)"} for k, v in pipes.FLANGE_CLASS_PSIG.items()]},
    {"id": "design_factor", "section": "MOP", "kind": "number", "label": "Design factor for the Barlow check"},
    # ---- drive and N2
    {"id": "max_drive_psig", "section": "Drive and N2", "kind": "number", "unit": "psig",
     "label": "Maximum drive pressure", "help": "Blank uses the MOP."},
    {"id": "n2_supply_psig", "section": "Drive and N2", "kind": "number", "unit": "psig",
     "label": "N2 spread maximum discharge", "help": "Blank uses the maximum drive."},
    {"id": "n2_supply_scfm", "section": "Drive and N2", "kind": "number", "unit": "SCFM",
     "label": "N2 spread maximum rate"},
    {"id": "drive_margin_pct", "section": "Drive and N2", "kind": "number", "unit": "%",
     "label": "Hold the drive this far below MOP", "help": "CHS jobs held 10%. Blank keeps the current margin."},
    {"id": "n2_temperature_f", "section": "Drive and N2", "kind": "number", "unit": "F", "label": "N2 temperature"},
    # ---- speed
    {"id": "target_speed_mph", "section": "Pig speed", "kind": "number", "unit": "mph", "label": "Target pig speed"},
    {"id": "max_speed_mph", "section": "Pig speed", "kind": "number", "unit": "mph", "label": "Hard maximum pig speed",
     "help": "Blank uses the target speed as the cap."},
    {"id": "min_speed_mph", "section": "Pig speed", "kind": "number", "unit": "mph", "label": "Minimum pig speed",
     "help": "Blank on a new job uses target - 0.25 mph so the controller defends the target."},
    # ---- strategy
    {"id": "strategy", "section": "Strategy", "kind": "select", "label": "Injection strategy",
     "options": [{"value": "lean", "label": "Lean: inject as needed to hold speed"},
                 {"value": "pack_and_coast", "label": "Pack and coast: inject a budget, then coast"}]},
    {"id": "n2_budget_scf", "section": "Strategy", "kind": "number", "unit": "SCF", "label": "N2 budget (pack and coast)",
     "help": "Blank uses the pre-run check's minimum to hold target speed, which is a starting point to sweep."},
    {"id": "pack_pressure_psig", "section": "Strategy", "kind": "number", "unit": "psig",
     "label": "Pack pressure (pack and coast)",
     "help": "Blank uses the highest flat pack that keeps every joint at or under MOP with the column at rest."},
]
_Q = {q["id"]: q for q in QUESTIONS}

FRESH_DEFAULT_SPEED_MPH = 3.0


class IntakeError(ValueError):
    """A rejected answer. The message is shown as-is."""


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

def _clean(qid: str, value: Any) -> Any:
    if qid not in _Q:
        raise IntakeError(f"unknown question {qid!r}; one of {list(_Q)}")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    q = _Q[qid]
    if q["kind"] == "bool":
        if isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "y")
        return bool(value)
    if q["kind"] == "select":
        v = str(value).strip()
        if qid == "nps":
            try:
                return pipes.nps_key(v)
            except KeyError:
                raise IntakeError(f"nps {value!r} is not a standard pipe size")
        allowed = [o["value"] for o in q["options"]]
        match = next((a for a in allowed if a.lower() == v.lower()), None)
        if match is None:
            raise IntakeError(f"{qid} must be one of {allowed}")
        return match
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise IntakeError(f"{qid} must be a number, got {value!r}")
    if not math.isfinite(x):
        raise IntakeError(f"{qid} must be finite")
    if qid in ("purge_start_mp", "purge_end_mp", "fluid_exit_mp"):
        return x
    if x < 0 or (x == 0 and qid not in ("drive_margin_pct", "n2_temperature_f", "exit_pressure_psig")):
        raise IntakeError(f"{qid} must be positive")
    if qid == "drive_margin_pct" and x >= 100:
        raise IntakeError("drive_margin_pct must be below 100")
    if qid == "design_factor" and x > 1:
        raise IntakeError("design_factor must be at most 1")
    return x


def stored_answers(sc: Scenario) -> Dict[str, Any]:
    return dict((sc.meta.intake or {}).get("answers") or {})


def _is_fresh(sc: Scenario) -> bool:
    return bool((sc.meta.intake or {}).get("fresh_import"))


def _pipe_from_data(inp: ScenarioInputs) -> bool:
    """The pipe geometry came from the data file: a Rosen ILI tally or a PxP pressure sheet."""
    return inp.data_source.startswith(("ILI", "PxP"))


def _pipe_is_placeholder(inp: ScenarioInputs) -> bool:
    s = inp.pipe_segments
    return (len(s) == 1 and abs(s[0]["od_in"] - 24.0) < 1e-9 and abs(s[0]["wt_in"] - 0.313) < 1e-9
            and not _pipe_from_data(inp))


def _joints_uniform(inp: ScenarioInputs) -> Optional[float]:
    if not inp.mop_joints:
        return None
    vals = {round(float(j["mop_psig"]), 3) for j in inp.mop_joints}
    return vals.pop() if len(vals) == 1 else None


# ---------------------------------------------------------------------------
# Defaults: what a blank answer means for this scenario
# ---------------------------------------------------------------------------

def _defaults(sc: Scenario, a: Dict[str, Any]) -> Dict[str, Tuple[Any, str]]:
    """{question id: (value, source)} for every question, given the answers so far.
    source is 'data' (grounded in the scenario or its import), 'derived' (computed from
    other answers), 'standard' (a modeling convention that rarely changes) or 'assumed'
    (a typical value the job may not match: listed for the user to confirm). A value
    of None means there is no usable default."""
    inp, fresh = sc.inputs, _is_fresh(sc)
    prof = inp.elevation_profile
    d: Dict[str, Tuple[Any, str]] = {}
    d["reverse_route"] = (bool((sc.meta.intake or {}).get("reversed", False)), "data")
    d["purge_start_mp"] = (inp.purge_start_mp, "data")
    d["purge_end_mp"] = (inp.purge_end_mp, "data")
    d["fluid_exit_mp"] = (None, "data")

    # fluid
    key = fluid_lib.match(inp.fluid_name)
    d["fluid"] = (None, "data") if fresh else (key, "data")
    f = fluid_lib.FLUIDS.get(a.get("fluid") or d["fluid"][0] or "")
    d["api_gravity"] = (None, "data")
    d["fluid_sg"] = ((round(fluid_lib.api_to_sg(a["api_gravity"]), 4), "derived") if a.get("api_gravity")
                     else (f.sg, "assumed") if f and (a.get("fluid") or fresh) else (inp.fluid_sg, "data"))
    d["fluid_viscosity_cst"] = ((f.viscosity_cst, "assumed") if f and (a.get("fluid") or fresh)
                                else (inp.fluid_viscosity_cst, "data"))

    # exit
    d["exit_type"] = ("pressurized" if f and f.vapor_psia and f.min_liquid_psig and f.min_liquid_psig > 75
                      else "tankage", "assumed")
    vmin = f.min_liquid_psig if f else None
    if vmin and vmin > 50:
        d["exit_pressure_psig"] = (float(5 * math.ceil(vmin / 5)), "derived")
    elif fresh or a.get("fluid_exit_mp"):
        d["exit_pressure_psig"] = (50.0, "assumed")
    else:
        d["exit_pressure_psig"] = (inp.exit_pressure_run_psig, "data")

    # pipe
    placeholder = _pipe_is_placeholder(inp)
    cur_nps = pipes.nps_for_od(inp.pipe_segments[0]["od_in"]) if inp.pipe_segments else None
    d["nps"] = (None, "data") if placeholder or fresh and not _pipe_from_data(inp) \
        else (cur_nps, "data")
    nps = a.get("nps") or d["nps"][0]
    pipe_from_intake = bool((sc.meta.intake or {}).get("pipe_set"))
    if a.get("nps") and (a["nps"] != cur_nps or placeholder or pipe_from_intake):
        std = pipes.standard_wt_in(nps)
        d["wt_in"] = (std, "assumed") if std else (None, "assumed")
    elif inp.pipe_segments and not placeholder:
        d["wt_in"] = (inp.pipe_segments[0]["wt_in"], "data")
    else:
        d["wt_in"] = (pipes.standard_wt_in(nps) if nps else None, "assumed")
    d["grade"] = (None, "data")

    # MOP
    uniform = _joints_uniform(inp)
    if a.get("flange_class"):
        d["mop_basis"] = ("flange_class", "derived")
    elif a.get("mop_psig"):
        d["mop_basis"] = ("flat", "derived")
    elif inp.mop_joints and uniform is None:
        d["mop_basis"] = ("from_data", "data")
    else:
        d["mop_basis"] = ("flat", "assumed" if fresh else "data")
    basis = a.get("mop_basis") or d["mop_basis"][0]
    if basis == "flange_class" and a.get("flange_class"):
        d["mop_psig"] = (pipes.FLANGE_CLASS_PSIG[a["flange_class"]], "derived")
    elif basis == "from_data" and inp.mop_joints:
        d["mop_psig"] = (min(j["mop_psig"] for j in inp.mop_joints), "data")
    elif not fresh and (inp.maop_psig or uniform):
        d["mop_psig"] = (inp.maop_psig or uniform, "data")
    else:
        d["mop_psig"] = (None, "data")
    d["flange_class"] = (None, "data")
    d["design_factor"] = (pipes.DEFAULT_DESIGN_FACTOR, "standard")
    mop = a.get("mop_psig") or d["mop_psig"][0]

    # drive and N2
    if fresh or a.get("mop_psig") or a.get("flange_class"):
        d["max_drive_psig"] = (mop, "derived")
    else:
        d["max_drive_psig"] = (inp.max_drive_psig, "data")
    drive = a.get("max_drive_psig") or d["max_drive_psig"][0]
    d["n2_supply_psig"] = ((drive, "derived") if fresh or a.get("max_drive_psig") or a.get("mop_psig")
                           else (inp.max_injection_psig, "data"))
    d["n2_supply_scfm"] = (inp.max_injection_scfm, "assumed" if fresh else "data")
    d["drive_margin_pct"] = (round((1.0 - inp.drive_ceiling_fraction) * 100.0, 3), "standard" if fresh else "data")
    d["n2_temperature_f"] = (inp.n2_temperature_f, "standard" if fresh else "data")

    # speed
    d["target_speed_mph"] = ((FRESH_DEFAULT_SPEED_MPH, "assumed") if fresh else (inp.target_speed_mph, "data"))
    tgt = a.get("target_speed_mph") or d["target_speed_mph"][0]
    d["max_speed_mph"] = ((tgt, "derived") if fresh or a.get("target_speed_mph") else (inp.max_speed_mph, "data"))
    d["min_speed_mph"] = ((max(0.5, round(tgt - 0.25, 3)), "derived") if fresh or a.get("target_speed_mph")
                          else (inp.min_speed_mph, "data"))

    # strategy (budget and pack pressure are filled from the pre-run check in apply())
    d["strategy"] = ("pack_and_coast" if inp.n2_budget_scf else "lean", "standard" if fresh else "data")
    d["n2_budget_scf"] = (None if fresh else inp.n2_budget_scf, "data")
    pack = None
    if inp.n2_budget_scf and not fresh:
        pack = inp.drive_mop_fraction * (inp.maop_psig or uniform or 0) or None
    d["pack_pressure_psig"] = (pack, "data")
    # a default that was an assumption when first applied stays one until it's answered,
    # even though the scenario now carries the value
    for qid in (sc.meta.intake or {}).get("assumed_ids", []):
        if qid in d and d[qid][0] is not None and d[qid][1] == "data":
            d[qid] = (d[qid][0], "assumed")
    return d


def view(sc: Scenario) -> dict:
    """Every question with its answer, the default that applies when blank, and status."""
    a = stored_answers(sc)
    d = _defaults(sc, a)
    basis = a.get("mop_basis") or d["mop_basis"][0]
    strategy = a.get("strategy") or d["strategy"][0]
    out, missing = [], []
    for q in QUESTIONS:
        qid = q["id"]
        dv, src = d.get(qid, (None, "data"))
        relevant = True
        if qid == "flange_class":
            relevant = basis == "flange_class"
        elif qid == "mop_psig":
            relevant = basis == "flat"
        elif qid in ("n2_budget_scf", "pack_pressure_psig"):
            relevant = strategy == "pack_and_coast"
        elif qid == "api_gravity":
            relevant = "crude" in (a.get("fluid") or d["fluid"][0] or "")
        # Product is only unknown on a fresh import; a saved scenario's SG and viscosity are data.
        required = (bool(q.get("required")) and (qid != "fluid" or _is_fresh(sc))) \
            or (qid == "flange_class" and basis == "flange_class")
        if qid == "mop_psig":
            required = basis == "flat"
        if qid in a and a[qid] is not None:
            status = "answered"
        elif dv is not None:
            status = "default"
        else:
            status = "missing" if required and relevant else "blank"
        if status == "missing":
            missing.append(qid)
        out.append({**q, "answer": a.get(qid), "default": dv, "default_source": src,
                    "status": status, "relevant": relevant, "required": required})
    return {"questions": out, "missing": missing, "fresh_import": _is_fresh(sc),
            "fluids": {k: f.as_dict() for k, f in fluid_lib.FLUIDS.items()}}


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _mirror(mp: float, a: float, b: float) -> float:
    return round(a + b - float(mp), 6)


def _reverse_inputs(inp: ScenarioInputs) -> ScenarioInputs:
    """Flip the route so mileposts run from the other end. Everything carrying a
    milepost moves with it, so stations stay at the same physical place."""
    new = copy.deepcopy(inp)
    prof = inp.elevation_profile
    a, b = float(prof[0][0]), float(prof[-1][0])
    new.elevation_profile = [[_mirror(m, a, b), e] for m, e in reversed(prof)]
    new.route_latlon = list(reversed(inp.route_latlon))
    new.purge_start_mp = _mirror(inp.purge_end_mp, a, b)
    new.purge_end_mp = _mirror(inp.purge_start_mp, a, b)
    new.pipe_segments = sorted(({**s, "start_mp": _mirror(s["end_mp"], a, b), "end_mp": _mirror(s["start_mp"], a, b)}
                                for s in inp.pipe_segments), key=lambda s: s["start_mp"])
    for name in ("check_valves", "pump_stations", "booster_stations", "mop_joints", "landmarks"):
        setattr(new, name, sorted(({**x, "mp": _mirror(x["mp"], a, b)} for x in getattr(inp, name)),
                                  key=lambda x: x["mp"]))
    if inp.bpcv:
        new.bpcv = {**inp.bpcv, "mp": _mirror(inp.bpcv["mp"], a, b)}
    if inp.deployed_booster_mps:
        new.deployed_booster_mps = sorted(_mirror(m, a, b) for m in inp.deployed_booster_mps)
    return new


def _flat_joints(inp: ScenarioInputs, mop: float) -> List[dict]:
    """One MOP joint per profile point, as past flat-MOP jobs were built."""
    geom = [(s["start_mp"], s["end_mp"], s["od_in"], s["wt_in"]) for s in inp.pipe_segments]

    def odwt(mp):
        for s0, s1, od, wt in geom:
            if s0 - 1e-9 <= mp <= s1 + 1e-9:
                return od, wt
        return geom[-1][2], geom[-1][3]
    out = []
    for mp, e in inp.elevation_profile:
        od, wt = odwt(mp)
        out.append({"mp": float(mp), "mop_psig": float(mop), "elevation_ft": float(e), "od_in": od, "wt_in": wt})
    return out


def apply(sc: Scenario, answers: Dict[str, Any]) -> Tuple[ScenarioInputs, Dict[str, Any], dict]:
    """Merge `answers` into the stored ones and build the scenario inputs they describe.

    Returns (new inputs, new meta.intake, report). Nothing is changed on `sc`; the
    caller (Workspace) swaps them in under its lock. Raises IntakeError on a bad answer
    or on answers that contradict each other."""
    if not isinstance(answers, dict):
        raise IntakeError("answers must be an object of question id -> value")
    a = stored_answers(sc)
    for k, v in answers.items():
        cv = _clean(k, v)
        if cv is None:
            a.pop(k, None)
        else:
            a[k] = cv
    if not sc.inputs.elevation_profile:
        raise IntakeError("the scenario has no elevation profile yet; import a route first")

    intake_meta = dict(sc.meta.intake or {})
    inp = copy.deepcopy(sc.inputs)
    assumptions: List[str] = []
    warnings: List[str] = []

    # route direction first: every milepost answer is in the launch-to-stop direction
    want_rev = bool(a.get("reverse_route", intake_meta.get("reversed", False)))
    if want_rev != bool(intake_meta.get("reversed", False)):
        pa, pb = float(inp.elevation_profile[0][0]), float(inp.elevation_profile[-1][0])
        inp = _reverse_inputs(inp)
        intake_meta["reversed"] = want_rev
        # milepost answers from earlier calls were in the old direction; this call's are in the new one
        old_mp = {k: a.pop(k) for k in ("purge_start_mp", "purge_end_mp", "fluid_exit_mp")
                  if k in a and k not in answers}
        if "purge_end_mp" in old_mp:
            a["purge_start_mp"] = _mirror(old_mp["purge_end_mp"], pa, pb)
        if "purge_start_mp" in old_mp:
            a["purge_end_mp"] = _mirror(old_mp["purge_start_mp"], pa, pb)
        if "fluid_exit_mp" in old_mp:
            warnings.append("The liquid exit milepost was answered in the old direction; it was cleared, "
                            "answer it again.")
        warnings.append("Route reversed: mileposts now run from the other end of the file, and every station "
                        "and joint milepost moved with it.")
    d = _defaults(Scenario(meta=dataclasses.replace(sc.meta, intake=intake_meta), inputs=inp), a)

    assumed_ids: List[str] = []

    def val(qid):
        if a.get(qid) is not None:
            return a[qid]
        v, src = d.get(qid, (None, "data"))
        if v is not None and src == "assumed":
            assumed_ids.append(qid)
            q = _Q[qid]
            shown = f"{v:g}" if isinstance(v, float) else str(v)
            assumptions.append(f"{q['label']}: {shown}{(' ' + q['unit']) if q.get('unit') else ''}")
        return v

    prof = np.asarray(inp.elevation_profile, dtype=float)
    p0, p1 = float(prof[0, 0]), float(prof[-1, 0])

    # ---- route
    start, end = float(val("purge_start_mp")), float(val("purge_end_mp"))
    if not (p0 - 1e-3 <= start < end <= p1 + 1e-3):
        raise IntakeError(f"pig launch {start:g} and stop {end:g} must lie on the profile (MP {p0:g} to {p1:g}) "
                          "with the stop past the launch")
    inp.purge_start_mp, inp.purge_end_mp = start, end
    fluid_exit = a.get("fluid_exit_mp")
    if fluid_exit is not None and fluid_exit < end - 1e-6:
        raise IntakeError(f"the liquid exit (MP {fluid_exit:g}) can't be before the pig stop (MP {end:g})")

    # ---- fluid
    fkey = val("fluid")
    if fkey:
        f = fluid_lib.get(fkey)
        if a.get("fluid") or _is_fresh(sc):
            inp.fluid_name = f.name
    else:
        f = None
    inp.fluid_sg = float(a.get("fluid_sg") or val("fluid_sg"))
    inp.fluid_viscosity_cst = float(val("fluid_viscosity_cst"))
    if a.get("api_gravity") and a.get("fluid_sg"):
        warnings.append("Both API gravity and SG were given; the SG override wins.")
    if a.get("api_gravity") and f and "API" in inp.fluid_name:
        inp.fluid_name = f"{inp.fluid_name.split('(')[0].strip()} (API {a['api_gravity']:g})"

    # ---- pipe
    nps = val("nps")
    wt = val("wt_in")
    od = pipes.od_in(nps) if nps else None
    pipe_changed = bool(a.get("nps") or a.get("wt_in")) and od and wt
    if pipe_changed:
        if len(inp.pipe_segments) > 1:
            warnings.append(f"Replaced {len(inp.pipe_segments)} pipe segments with one uniform {od:.3f}\" x {wt:.3f}\" "
                            "segment. Edit the pipe segments if the size changes along the route.")
        inp.pipe_segments = [{"start_mp": p0, "end_mp": p1, "od_in": od, "wt_in": float(wt)}]
        intake_meta["pipe_set"] = True
    elif nps and not wt:
        warnings.append(f"No standard wall thickness on file for NPS {nps}; give the wall thickness.")
    if _pipe_is_placeholder(inp):
        warnings.append("Pipe size is still the import placeholder (24\" x 0.313\"); answer the pipe size.")

    # ---- MOP
    basis = val("mop_basis")
    mop = None
    if basis == "flange_class":
        fc = val("flange_class")
        if not fc:
            raise IntakeError("mop_basis is flange_class: give the flange_class")
        mop = pipes.FLANGE_CLASS_PSIG[fc]
    elif basis == "flat":
        mop = val("mop_psig")
    fresh = _is_fresh(sc)
    mop_answered = any(a.get(k) is not None for k in ("mop_basis", "mop_psig", "flange_class"))
    if basis in ("flat", "flange_class") and mop and (fresh or mop_answered):
        if inp.mop_joints and _joints_uniform(inp) is None:
            warnings.append(f"Replaced {len(inp.mop_joints)} point-by-point MOP joints from the data with a flat "
                            f"{mop:g} psig.")
        inp.maop_psig = float(mop)
        inp.mop_joints = _flat_joints(inp, mop)
    elif basis == "from_data":
        if not inp.mop_joints:
            raise IntakeError("mop_basis is from_data but this scenario has no point-by-point MOP data")
        mop = min(j["mop_psig"] for j in inp.mop_joints)
    mop_min = float(np.min(mop_at(inp, [j[0] for j in inp.elevation_profile]))) if (inp.mop_joints or inp.maop_psig) \
        else None
    if mop_min is not None and not math.isfinite(mop_min):
        mop_min = None

    barlow = None
    if mop and inp.pipe_segments and not _pipe_is_placeholder(inp):
        seg = min(inp.pipe_segments, key=lambda s: s["wt_in"] / s["od_in"])
        barlow = pipes.barlow_check(seg["od_in"], seg["wt_in"], float(mop_min or mop), a.get("grade"),
                                    float(val("design_factor")))
        if barlow.get("ok") is False:
            warnings.append(f"Barlow check: {seg['od_in']:.3f}\" x {seg['wt_in']:.3f}\" {barlow['grade']} at "
                            f"{barlow['design_factor']:g} design factor is good for {barlow['barlow_psig']:,.0f} psig, "
                            f"below the {mop_min or mop:,.0f} psig MOP.")
        elif barlow.get("ok") is None and barlow["min_grade"] != "B":
            g = barlow["min_grade"]
            assumptions.append(f"Pipe grade unknown: the MOP needs at least {g} at a {barlow['design_factor']:g} design factor"
                               if g else "Pipe grade unknown, and no API 5L grade justifies this MOP at this wall")

    # ---- drive and N2
    drive = val("max_drive_psig")
    inp.max_drive_psig = float(drive) if drive else None
    supply = val("n2_supply_psig")
    inp.max_injection_psig = float(supply) if supply else None
    inp.max_injection_scfm = float(val("n2_supply_scfm"))
    inp.drive_ceiling_fraction = round(1.0 - float(val("drive_margin_pct")) / 100.0, 6)
    inp.n2_temperature_f = float(val("n2_temperature_f"))
    if drive and mop_min and drive > mop_min + 1e-6:
        warnings.append(f"Maximum drive {drive:g} psig is above the lowest MOP ({mop_min:g} psig); the simulator "
                        "caps the drive at MOP anyway.")
    if supply and drive and supply < drive:
        warnings.append(f"The N2 spread ({supply:g} psig) can't reach the {drive:g} psig drive cap; the spread is "
                        "the real limit.")

    # ---- speed
    tgt = float(val("target_speed_mph"))
    inp.target_speed_mph = tgt
    inp.max_speed_mph = float(val("max_speed_mph"))
    inp.min_speed_mph = float(val("min_speed_mph"))
    if not (0 < inp.min_speed_mph <= tgt <= inp.max_speed_mph):
        raise IntakeError(f"speeds must satisfy 0 < min ({inp.min_speed_mph:g}) <= target ({tgt:g}) <= "
                          f"max ({inp.max_speed_mph:g})")

    # ---- exit (needs fluid, pipe and speed for the stretch past the pig stop)
    delivery = float(val("exit_pressure_psig"))
    vmin = f.min_liquid_psig if f else None
    if vmin and delivery < vmin:
        warnings.append(f"{inp.fluid_name} needs at least {vmin:g} psig everywhere (vapor pressure + "
                        f"{fluid_lib.VAPOR_MARGIN_PSI:g} psi); the {delivery:g} psig delivery pressure is below that.")
    at_stop = delivery
    if fluid_exit is not None and fluid_exit > end + 1e-6:
        try:
            at_stop = round(exit_pressure_at_stop(inp, end, float(fluid_exit), delivery), 1)
        except ValueError as e:
            raise IntakeError(str(e))
        warnings.append(f"The liquid leaves at MP {fluid_exit:g}, past the pig stop at MP {end:g}. The simulator "
                        f"puts the exit at the pig stop, so it is set to {at_stop:g} psig: {delivery:g} psig delivery "
                        f"plus that stretch's friction at {tgt:g} mph and the head to clear its terrain.")
    exit_answered = any(a.get(k) is not None for k in ("exit_pressure_psig", "exit_type", "fluid_exit_mp", "fluid"))
    if fresh or exit_answered:
        # past jobs all held a constant delivery pressure to the end of the run
        inp.exit_pressure_run_psig = inp.exit_pressure_end_psig = float(at_stop)
        inp.exit_pressure_behavior = "constant_run"
        inp.throttle_down_miles = 0.0

    # ---- pre-run check, then strategy (it needs the check's pack limit and budget)
    try:
        pre = precheck(inp, vmin)
    except ValueError as e:
        pre = {"error": str(e), "findings": [], "job_type": []}
    strategy = val("strategy")
    if strategy == "pack_and_coast":
        prior = set((sc.meta.intake or {}).get("assumed_ids", []))
        # an assumed pack pressure or budget is re-estimated from the current inputs each time
        pack = a.get("pack_pressure_psig")
        keep_fraction = pack is None and sc.inputs.n2_budget_scf is not None and "pack_pressure_psig" not in prior
        if pack is None and not keep_fraction:
            pack = pre.get("static_pack_limit_psig")
        if a.get("pack_pressure_psig") is None and pack and (not keep_fraction or "pack_pressure_psig" in prior):
            assumed_ids.append("pack_pressure_psig")
            assumptions.append(f"Pack pressure: {pack:g} psig (highest flat pack with every joint at or under "
                               "MOP, column at rest)")
        budget = a.get("n2_budget_scf") or (None if "n2_budget_scf" in prior else d["n2_budget_scf"][0])
        if budget is None:
            budget = pre.get("n2_min_coast_budget_scf")
        if a.get("n2_budget_scf") is None and budget and (d["n2_budget_scf"][0] is None or "n2_budget_scf" in prior):
            assumed_ids.append("n2_budget_scf")
            assumptions.append(f"N2 budget: {budget:,.0f} SCF (minimum to hold target speed on the coast; sweep it)")
        if not budget:
            raise IntakeError("pack and coast needs an N2 budget, and the pre-run check couldn't estimate one")
        inp.n2_budget_scf = float(budget)
        if pack and mop_min and not keep_fraction:
            inp.drive_mop_fraction = round(min(1.0, float(pack) / mop_min), 4)
    else:
        inp.n2_budget_scf = None

    # ---- record
    intake_meta["answers"] = a
    intake_meta.pop("fresh_import", None)
    if fluid_exit is not None:
        intake_meta["exit_pressure_at_stop_psig"] = at_stop
    missing = view(Scenario(meta=dataclasses.replace(sc.meta, intake=intake_meta), inputs=inp))["missing"]
    report = {"assumptions": assumptions, "warnings": warnings, "missing": missing,
              "barlow": barlow, "precheck": pre}
    intake_meta["assumptions"] = assumptions
    intake_meta["assumed_ids"] = sorted(set(assumed_ids))
    return inp, intake_meta, report


def notes_block(inp: ScenarioInputs, report: dict) -> str:
    """The 'Job setup' block written into the scenario notes."""
    seg = inp.pipe_segments
    pipe = ", ".join(f'{s["od_in"]:.3f}" x {s["wt_in"]:.3f}" (MP {s["start_mp"]:g}-{s["end_mp"]:g})' for s in seg[:4])
    pre = report.get("precheck") or {}
    lines = [NOTES_BEGIN,
             f"Pig MP {inp.purge_start_mp:g} -> {inp.purge_end_mp:g}; exit {inp.exit_pressure_run_psig:g} psig. "
             f"{inp.fluid_name}, SG {inp.fluid_sg:g}, {inp.fluid_viscosity_cst:g} cSt. Pipe {pipe}.",
             f"MOP {inp.maop_psig or '-'} psig; max drive {inp.max_drive_psig or '-'} psig; N2 spread "
             f"{inp.max_injection_psig or '-'} psig / {inp.max_injection_scfm:,.0f} SCFM. Speed "
             f"{inp.min_speed_mph:g}/{inp.target_speed_mph:g}/{inp.max_speed_mph:g} mph (min/target/max). "
             + (f"Pack and coast: {inp.n2_budget_scf:,.0f} SCF at {inp.drive_mop_fraction:.0%} of MOP."
                if inp.n2_budget_scf else "Lean injection.")]
    if pre.get("job_type"):
        lines.append("Job type: " + ", ".join(pre["job_type"]) + ".")
    if report.get("assumptions"):
        lines.append("ASSUMED - CONFIRM: " + "; ".join(report["assumptions"]) + ".")
    for w in report.get("warnings", []):
        lines.append("NOTE: " + w)
    lines.append(NOTES_END)
    return "\n".join(lines)


def merge_notes(notes: str, block: str) -> str:
    """Replace the previous Job setup block in the notes, or append one."""
    notes = notes or ""
    if NOTES_BEGIN in notes and NOTES_END in notes:
        i, j = notes.index(NOTES_BEGIN), notes.index(NOTES_END) + len(NOTES_END)
        return (notes[:i] + block + notes[j:]).strip()
    return (notes.rstrip() + "\n\n" + block).strip() if notes.strip() else block
