"""
In-app engineering assistant, backed by the Claude API.

The assistant works on the same Workspace the user sees: it can browse and load
scenarios, read inputs and results, change inputs (through the same validation
and hard-rule checks as the UI), run the simulator, sweep a parameter, and save
a new scenario, and look a route's elevation up in USGS 3DEP again. It cannot
overwrite bundled or existing scenarios, cannot type in elevation or MOP arrays,
and cannot lower the booster suction floor.

The API key comes from settings.py (env var or the user's settings file).
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from ..data.elevation import DEFAULT_SPACING_FT
from . import settings
from .workspace import InputError, Workspace, inputs_summary, results_series, results_summary, _f

MAX_TOOL_ROUNDS = 25
MAX_SWEEP_VALUES = 8

# Models that accept the server-side refusal fallback ("default" routing).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

SYSTEM_PROMPT = """\
You are the engineering assistant built into Purge Simulator, a desktop tool that models \
nitrogen-displacement ("purge") operations on liquid pipelines: a pig driven by N2 gas \
pressure pushes the liquid product (crude, diesel, butane, NGL) ahead of it to an exit \
(tankage, a pump suction, or a back-pressure control valve). The person you're talking to \
is a pipeline purge engineer preparing bids and execution plans for real client jobs. \
Treat them as the expert: be direct, quantitative and brief.

## What the engine models
- N2 column behind the pig, split into isolated segments by mainline check valves and \
booster stations; pressure via Peng-Robinson.
- Pig speed solved so N2 drive pressure balances liquid resistance ahead (friction, \
static head, exit backpressure).
- Pump stations ahead of the pig hold the liquid HGL and shut down when no longer \
hydraulically needed (hard 1-mile safety limit ahead of the pig).
- BPCV sets exit backpressure = min over downstream joints of (MOP - static head - friction).
- Booster spreads recompress upstream N2 and discharge downstream; n_spreads and \
mob_time_hr govern how many spreads exist and how long moves take.
- A pre-run pressure "roadmap" (ceiling/floor corridor per milepost) caps the drive and \
sites boosters.
- Strategies: the default lean floor-defending controller, or pack-and-coast \
(n2_budget_scf set: inject while holding drive near drive_mop_fraction x MOP, then coast \
on the stored column).
- Typical job classes: speed-capped (max_speed_mph binds), drive-capped (max_drive_psig / \
MOP binds), friction-dominated (small diameter, long), pump-and-BPCV jobs.

## Hard engineering rules (never break these; flag any result that does)
1. No N2 venting as a relied-upon mechanism. Any vented SCF > 0 is a red flag to fix by \
control (drive ceiling, booster settings, budget), never to accept.
2. Zero slack line is the default target. Any step with slack_line_risk is reported, never \
hidden or reinterpreted. Accepting slack is the engineer's explicit call, not yours.
3. The BPCV only drops the HGL; it never adds head.
4. Booster suction floor is 200 psi hard / 100 psi willing. Never set a booster to draw \
below 100 psi; the app rejects it, and you should not try to work around that. A few saved \
scenarios carry a lower floor the engineer signed off for that job; leave those as they are \
and point them out when relevant.

## How to work
- Use the tools. Ground every number you state in a tool result from this conversation, \
and say which run or scenario it came from. If you're estimating or inferring, say so.
- Before changing inputs, say what you'll change and why in one line; after a run, lead \
with the outcome (completed or not, N2 total, flags), then the one or two things that \
matter.
- Scenario edits change the user's open scenario immediately (they see it in the app). \
Edits are not saved to disk until saved; save only when asked, always under a new name.
- When asked to compare options (drive pressures, speeds, budgets, booster sets), prefer \
run_sweep over editing the open scenario repeatedly.
- New data imports leave pipe geometry, fluid, drive limits and pump-vs-BPCV roles at \
defaults. Help classify them: which detected stations really pump, whether there's a \
BPCV, and which job class this is, and state the assumptions explicitly.
- Elevation comes from data, never from you. Don't type in, estimate or "fill" elevations. \
A KMZ without altitudes is looked up in USGS 3DEP on import; get_scenario shows the \
profile's elevation_source (provider, spacing, interpolated points, flags). If a profile \
is missing, flat, or from too coarse a spacing, use fetch_elevation. Mention any flags \
(interpolated gaps, dead-flat stretches that may be river or HDD crossings) when they \
touch the analysis.
- Keep replies short. Use a small table when comparing runs. No preamble.
"""

TOOLS: List[dict] = [
    {
        "name": "list_scenarios",
        "description": "List every scenario in the library: bundled client-job scenarios and the user's own saved ones. Returns id, job folder, name, span, fluid and the start of each scenario's notes.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "load_scenario",
        "description": "Open a scenario by id (from list_scenarios) as the working scenario. Replaces what the user has open and clears previous results. If the open scenario has unsaved changes this is refused; set discard_unsaved only after the user agrees to lose them.",
        "input_schema": {"type": "object", "properties": {
            "scenario_id": {"type": "string"}, "discard_unsaved": {"type": "boolean"}},
            "required": ["scenario_id"], "additionalProperties": False},
    },
    {
        "name": "get_scenario",
        "description": "Read the open scenario: meta and notes, every scalar input, infrastructure lists (pipe segments, check valves, pump stations, boosters, BPCV) and statistics for the elevation profile and MOP joints.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_elevation_profile",
        "description": "Sample the open scenario's elevation profile (milepost, elevation ft) over a milepost range, downsampled.",
        "input_schema": {"type": "object", "properties": {
            "start_mp": {"type": "number"}, "end_mp": {"type": "number"},
            "max_points": {"type": "integer", "minimum": 10, "maximum": 1000}},
            "additionalProperties": False},
    },
    {
        "name": "update_inputs",
        "description": (
            "Change inputs on the open scenario. `changes` maps ScenarioInputs field names to new values, e.g. "
            "{\"max_drive_psig\": 450, \"target_speed_mph\": 2.5}. List fields (pipe_segments, check_valves, "
            "pump_stations, booster_stations) are replaced whole. bpcv is {mp, elevation_ft, name} or null. "
            "elevation_profile and mop_joints cannot be edited. Edits are validated and hard rules enforced; a "
            "rejected edit changes nothing and returns the reason. Returns the old/new diff."),
        "input_schema": {"type": "object", "properties": {
            "changes": {"type": "object"},
            "reason": {"type": "string", "description": "One line on why, shown to the user."}},
            "required": ["changes", "reason"], "additionalProperties": False},
    },
    {
        "name": "run_simulation",
        "description": "Run the simulator on the open scenario (can take from seconds to a few minutes on long routes). Returns the results summary: completion, N2 totals, vented N2, pressures, speeds, MOP and slack-line flags, booster plan, roadmap report.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_results",
        "description": "Read the latest run in more detail. view='summary' (same as run_simulation), 'timeseries' (downsampled per-step series: time, pig MP, speed, face/injection psig, injection SCFM, N2 inventory, exit psig, flags), or 'profile' (full-route pressure vs elevation and MOP at one moment, chosen by pig_mp or step_index).",
        "input_schema": {"type": "object", "properties": {
            "view": {"type": "string", "enum": ["summary", "timeseries", "profile"]},
            "max_points": {"type": "integer", "minimum": 10, "maximum": 600},
            "pig_mp": {"type": "number"}, "step_index": {"type": "integer"}},
            "required": ["view"], "additionalProperties": False},
    },
    {
        "name": "run_sweep",
        "description": f"Run the open scenario once per value of one input (up to {MAX_SWEEP_VALUES} values) without changing the open scenario, and return a comparison table (completion, N2 totals, vented, max face psig, speeds, duration, MOP and slack flags). Use for comparing drive pressures, speeds, budgets, spreads etc.",
        "input_schema": {"type": "object", "properties": {
            "field": {"type": "string"},
            "values": {"type": "array", "items": {}, "minItems": 1, "maxItems": MAX_SWEEP_VALUES}},
            "required": ["field", "values"], "additionalProperties": False},
    },
    {
        "name": "fetch_elevation",
        "description": (
            "Look the open scenario's route up in USGS 3DEP (US only) and replace its elevation profile. "
            "Needs a scenario imported from a KMZ or GPS file (route_points > 0 in get_scenario). Samples every "
            "spacing_ft (default 250) plus route vertices, refines steep stretches, and interpolates no-data "
            "points. reverse=true flips the route so mileposts run from the other end (station mileposts are "
            "not moved). This sends the route's coordinates to the USGS service; if the user has turned on "
            "'ask before USGS lookups', get their OK first and pass user_confirmed=true. Returns points, "
            "length, interpolated points and flags."),
        "input_schema": {"type": "object", "properties": {
            "spacing_ft": {"type": "number", "minimum": 25, "maximum": 5280},
            "reverse": {"type": "boolean"},
            "user_confirmed": {"type": "boolean"},
            "reason": {"type": "string", "description": "One line on why, shown to the user."}},
            "required": ["reason"], "additionalProperties": False},
    },
    {
        "name": "save_scenario_as",
        "description": "Save the open scenario to the user's scenarios folder under a new name, with notes recording the rationale. Never overwrites an existing file. Only do this when the user asks.",
        "input_schema": {"type": "object", "properties": {
            "name": {"type": "string"}, "notes": {"type": "string"}},
            "required": ["name"], "additionalProperties": False},
    },
]


def _sweep(ws: Workspace, field: str, values: List[Any]) -> dict:
    from ..engine.config_builder import build_sim_config
    from ..engine.simulator import simulate
    from .workspace import apply_changes
    sc = ws.require()
    if len(values) > MAX_SWEEP_VALUES:
        raise InputError(f"at most {MAX_SWEEP_VALUES} values per sweep")
    rows = []
    for v in values:
        try:
            inputs, _ = apply_changes(sc.inputs, {field: v})
        except InputError as e:
            rows.append({field: v, "error": str(e)})
            continue
        res = simulate(build_sim_config(inputs))
        s = results_summary(res)
        rows.append({
            field: v, "completed": s["completed"], "abort_reason": s["abort_reason"] or None,
            "n2_total_fresh_scf": s["n2_total_fresh_scf"], "n2_vented_scf": s["n2_vented_scf"],
            "duration_hr": s.get("duration_hr"),
            "max_face_psig": (s.get("pig_face_psig") or {}).get("max"),
            "avg_speed_mph": (s.get("pig_speed_mph") or {}).get("avg"),
            "min_speed_mph": (s.get("pig_speed_mph") or {}).get("min"),
            "mop_violation_steps": s.get("mop_violation_steps"),
            "slack_line_risk_steps": s.get("slack_line_risk_steps"),
        })
    return {"scenario": sc.meta.name, "field": field, "rows": rows,
            "note": "Sweep runs are not kept; the open scenario and its results are unchanged."}


def execute_tool(ws: Workspace, name: str, args: Dict[str, Any]) -> Any:
    if name == "list_scenarios":
        return ws.list_scenarios()
    if name == "load_scenario":
        if ws.dirty and not args.get("discard_unsaved"):
            raise InputError("the open scenario has unsaved changes; ask the user whether to save "
                             "them first or discard them")
        return ws.load(args["scenario_id"])
    if name == "get_scenario":
        return inputs_summary(ws.require())
    if name == "get_elevation_profile":
        return ws.elevation_view(args.get("start_mp"), args.get("end_mp"), int(args.get("max_points", 200)))
    if name == "update_inputs":
        diff = ws.update_inputs(args.get("changes") or {})
        return {"changed": diff or "no change (values already set)",
                "note": "The open scenario is updated (unsaved). Run the simulation to see the effect."}
    if name == "run_simulation":
        return results_summary(ws.run())
    if name == "get_results":
        res = ws.require_results()
        view = args.get("view", "summary")
        if view == "summary":
            return results_summary(res)
        if view == "timeseries":
            return results_series(res, int(args.get("max_points", 120)))
        if view == "profile":
            if args.get("step_index") is not None:
                i = int(args["step_index"])
            elif args.get("pig_mp") is not None:
                target = float(args["pig_mp"])
                i = min(range(len(res.steps)), key=lambda k: abs(res.steps[k].pig_mp - target))
            else:
                i = len(res.steps) // 2
            return ws.profile_at(i, max_points=int(args.get("max_points", 150)))
        raise InputError(f"unknown view {view!r}")
    if name == "run_sweep":
        return _sweep(ws, args["field"], list(args.get("values") or []))
    if name == "fetch_elevation":
        if settings.elevation_ask_first() and not args.get("user_confirmed"):
            raise InputError("the user's settings say to ask before sending a route to USGS; ask them, "
                             "then call again with user_confirmed=true")
        out = ws.fetch_elevation(float(args.get("spacing_ft") or DEFAULT_SPACING_FT), bool(args.get("reverse")))
        return {**out, "note": "The open scenario's elevation profile is replaced (unsaved). Run again to "
                               "see the effect."}
    if name == "save_scenario_as":
        return ws.save_as(args["name"], args.get("notes"), overwrite=False)
    raise InputError(f"unknown tool {name!r}")


def _tool_label(name: str, args: Dict[str, Any]) -> str:
    if name == "update_inputs":
        keys = ", ".join(f"{k} = {json.dumps(v)[:60]}" for k, v in (args.get("changes") or {}).items())
        reason = args.get("reason")
        return f"Changed {keys}" + (f" ({reason})" if reason else "")
    if name == "load_scenario":
        return f"Opened {args.get('scenario_id')}"
    if name == "run_sweep":
        return f"Swept {args.get('field')} over {args.get('values')}"
    if name == "save_scenario_as":
        return f"Saved as {args.get('name')}"
    if name == "fetch_elevation":
        reason = args.get("reason")
        return (f"Looked up elevation in USGS 3DEP at {args.get('spacing_ft') or DEFAULT_SPACING_FT:g} ft"
                + (", reversed" if args.get("reverse") else "") + (f" ({reason})" if reason else ""))
    return {"list_scenarios": "Listed scenarios", "get_scenario": "Read the scenario",
            "get_elevation_profile": "Read the elevation profile", "run_simulation": "Ran the simulation",
            "get_results": f"Read results ({args.get('view')})"}.get(name, name)


class Assistant:
    """One conversation with Claude over the shared Workspace. Runs turns on a worker thread."""

    def __init__(self, ws: Workspace, client_factory: Optional[Callable[[], Any]] = None):
        self.ws = ws
        self._client_factory = client_factory or self._default_client
        self.messages: List[dict] = []        # API transcript (append-only)
        self.transcript: List[dict] = []      # what the UI shows
        self.busy = False
        self.lock = threading.Lock()

    @staticmethod
    def _default_client():
        import anthropic
        key = settings.api_key()
        if not key:
            raise RuntimeError("No Claude API key is set. Add one in Settings, or set ANTHROPIC_API_KEY.")
        return anthropic.Anthropic(api_key=key)

    def view(self) -> dict:
        return {"busy": self.busy, "transcript": list(self.transcript)}

    def reset(self) -> None:
        with self.lock:
            if self.busy:
                raise InputError("the assistant is still working")
            self.messages, self.transcript = [], []

    def send(self, text: str, background: bool = True) -> None:
        text = (text or "").strip()
        if not text:
            raise InputError("message is empty")
        with self.lock:
            if self.busy:
                raise InputError("the assistant is still working on the last message")
            self.busy = True
            self.transcript.append({"role": "user", "text": text, "t": time.time()})
        if background:
            threading.Thread(target=self._turn, args=(text,), name="assistant", daemon=True).start()
        else:
            self._turn(text)

    def _note(self, role: str, text: str) -> None:
        self.transcript.append({"role": role, "text": text, "t": time.time()})

    def _create(self, client, model: str):
        kwargs = dict(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            cache_control={"type": "ephemeral"},
        )
        if model in _FALLBACK_MODELS:
            return client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return client.beta.messages.create(**kwargs)

    def _turn(self, text: str) -> None:
        start_len = len(self.messages)
        try:
            client = self._client_factory()
            model = settings.model()
            self.messages.append({"role": "user", "content": text})
            for _ in range(MAX_TOOL_ROUNDS):
                resp = self._create(client, model)
                self.messages.append({"role": "assistant", "content": resp.content})
                for block in resp.content:
                    if getattr(block, "type", None) == "text" and block.text.strip():
                        self._note("assistant", block.text)
                if resp.stop_reason == "refusal":
                    self._note("error", "Claude declined this request.")
                    return
                if resp.stop_reason == "max_tokens":
                    self._note("error", "The reply hit its length limit and was cut off.")
                    return
                tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                if resp.stop_reason != "tool_use" or not tool_uses:
                    return
                results = []
                for tu in tool_uses:
                    args = tu.input if isinstance(tu.input, dict) else {}
                    try:
                        out = execute_tool(self.ws, tu.name, args)
                        self._note("tool", _tool_label(tu.name, args))
                        results.append({"type": "tool_result", "tool_use_id": tu.id,
                                        "content": json.dumps(out, default=_json_fallback)})
                    except InputError as e:
                        self._note("tool", f"{_tool_label(tu.name, args)}: rejected ({e})")
                        results.append({"type": "tool_result", "tool_use_id": tu.id,
                                        "content": f"Rejected: {e}", "is_error": True})
                    except Exception as e:
                        self._note("tool", f"{_tool_label(tu.name, args)}: failed ({type(e).__name__}: {e})")
                        results.append({"type": "tool_result", "tool_use_id": tu.id,
                                        "content": f"Error: {type(e).__name__}: {e}\n"
                                                   f"{traceback.format_exc(limit=2)}",
                                        "is_error": True})
                self.messages.append({"role": "user", "content": results})
            self._note("error", f"Stopped after {MAX_TOOL_ROUNDS} tool rounds.")
        except Exception as e:
            self._note("error", _friendly_error(e))
            # Rewind this turn so the next message starts from a valid transcript
            # (no dangling user turn or unanswered tool_use).
            del self.messages[start_len:]
        finally:
            self.busy = False


def _json_fallback(o: Any) -> Any:
    try:
        return _f(o)
    except Exception:
        return str(o)


def _friendly_error(e: Exception) -> str:
    try:
        import anthropic
    except ImportError:
        return "The anthropic package isn't installed, so the assistant can't run (pip install anthropic)."
    if isinstance(e, anthropic.AuthenticationError):
        return "The Claude API key was rejected. Check it in Settings."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "This API key doesn't have access to the selected model."
    if isinstance(e, anthropic.NotFoundError):
        return f"Model {settings.model()!r} wasn't found. Check the model in Settings."
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited by the Claude API. Wait a moment and try again."
    if isinstance(e, anthropic.APIStatusError):
        return f"Claude API error {e.status_code}: {getattr(e, 'message', e)}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Couldn't reach the Claude API. Check the internet connection."
    return str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
