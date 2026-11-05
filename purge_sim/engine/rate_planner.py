"""
Steady injection-rate planner.

The strategy controllers (lean, pack-and-coast) trim the nitrogen pump every step, which
is exact in the model and useless to the people running the job: a pump operator can't
follow a rate that moves every few minutes, and the LN2 supervisor needs to know when the
tankers have to be there. This planner turns a run into a schedule with as few rate
changes as possible:

  1. run the job the strategy's own way (the reference);
  2. try ONE flat rate: the same total N2, pumped over the same injection time;
  3. run that with the pump pinned to the schedule (SimConfig.injection_schedule) and
     check every rule the reference is held to (completes, MOP, no venting, slack line,
     stalls, min / max speed, the pressure cap never has to hold the pump back, not
     much slower than the reference);
  4. if a rule breaks, try the fewest extra steps (two rates, then three...), each
     schedule being the best piecewise-constant fit to how the reference actually pumped,
     with a little more N2 as a fallback, until one passes.

The result is a schedule the crew can hold: "5,700 SCFM for 9.2 h, then stop" or
"6,000 SCFM for 4 h, then 5,000 SCFM for 5.2 h".
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .simulator import SimConfig, SimResults, simulate

Schedule = List[Tuple[float, float]]     # [(hours, scfm), ...] from t = 0, then the pump is off

RATE_ROUND_SCFM = 50.0        # rates are quoted to the nearest 50 SCFM
HOURS_ROUND = 0.1             # ...and durations to the nearest 6 minutes
BREAKPOINT_GRID_HR = 0.25     # candidate rate-change times
N2_SCALES = (1.0, 1.05, 0.95, 1.10)   # fallbacks when the exact-N2 fit breaks a rule
MAX_DURATION_STRETCH = 0.10   # a schedule may take this much longer than the reference
HELD_BACK_TOLERANCE_HR = 0.1  # the pressure cap may clip the pump for at most this long


@dataclass
class RuleResult:
    name: str
    ok: bool
    text: str


@dataclass
class Attempt:
    schedule: Schedule
    label: str
    n2_scf: float
    rules: List[RuleResult]
    summary: dict
    results: Optional[SimResults] = None

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.rules)

    @property
    def violations(self) -> List[str]:
        return [r.text for r in self.rules if not r.ok]


@dataclass
class RatePlan:
    reference: dict                    # summary of the strategy's own run
    reference_results: SimResults
    attempts: List[Attempt] = field(default_factory=list)
    best: Optional[Attempt] = None     # the first schedule that passed (fewest steps)

    @property
    def schedule(self) -> Optional[Schedule]:
        return self.best.schedule if self.best else None

    @property
    def text(self) -> str:
        return plan_text(self)


# ---------------------------------------------------------------------------
# Describing a run
# ---------------------------------------------------------------------------

def run_summary(res: SimResults) -> dict:
    """The numbers the rules are checked on, for the reference and every attempt."""
    cfg = res.config
    steps = res.steps
    if not steps:
        return {"completed": res.completed, "abort_reason": res.abort_reason, "duration_hr": 0.0,
                "n2_scf": res.total_scf_injected, "vented_scf": res.total_scf_vented,
                "mop_violation_steps": 0, "slack_steps": 0, "stall_steps": 0, "overspeed_steps": 0,
                "overspeed_mph_max": 0.0, "below_min_steps": 0, "held_back_hr": 0.0,
                "held_back_mp": None, "max_face_psig": 0.0, "injection_hr": 0.0,
                "avg_speed_mph": 0.0, "min_speed_mph": 0.0}
    t = np.array([s.t_hr for s in steps])
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2] if len(t) > 1 else 0.0))
    inj = np.array([s.injection_scfm for s in steps])
    speed = np.array([s.pig_speed_mph for s in steps])
    held = np.array([s.injection_held_back for s in steps], dtype=bool)
    on = inj > 0.02 * cfg.max_injection_scfm
    injection_hr = float(t[on][-1] + dt[on][-1]) if on.any() else 0.0
    held_hr = float(dt[held].sum()) if held.any() else 0.0
    held_mp = float(steps[int(np.argmax(held))].pig_mp) if held.any() else None
    over = [s for s in steps if s.overspeed]
    return {
        "completed": res.completed,
        "abort_reason": res.abort_reason,
        "duration_hr": float(t[-1]),
        "n2_scf": float(res.total_scf_injected),
        "vented_scf": float(res.total_scf_vented),
        "mop_violation_steps": int(sum(1 for s in steps if s.mop_violations)),
        "slack_steps": int(sum(1 for s in steps if s.slack_line_risk)),
        "stall_steps": int(sum(1 for s in steps[10:] if s.pig_speed_mph <= 0.0)),
        "below_min_steps": int(sum(1 for s in steps[10:] if 0.0 < s.pig_speed_mph < cfg.min_speed_mph)),
        "overspeed_steps": len(over),
        "overspeed_mph_max": float(max((s.overspeed_mph for s in over), default=0.0)),
        "held_back_hr": held_hr,
        "held_back_mp": held_mp,
        "max_face_psig": float(max(s.pig_face_psig for s in steps)),
        "injection_hr": injection_hr,
        "avg_speed_mph": float(speed.mean()),
        "min_speed_mph": float(speed[10:].min()) if len(speed) > 10 else float(speed.min()),
    }


def check_rules(summary: dict, reference: dict, cfg: SimConfig) -> List[RuleResult]:
    """Every rule a schedule must hold. A flag the reference run already carries (its own
    controller's launch ramp, an exit that can't hold the pig on a descent) is not held
    against the schedule, but the schedule may not make it worse."""
    s, r = summary, reference
    out: List[RuleResult] = []

    def rule(name, ok, bad, good=""):
        out.append(RuleResult(name, bool(ok), bad if not ok else good))

    rule("completes", s["completed"], f"the run does not complete: {s['abort_reason']}",
         f"completes in {s['duration_hr']:.1f} h")
    rule("mop", s["mop_violation_steps"] == 0,
         f"MOP exceeded on {s['mop_violation_steps']} steps", "every joint under MOP")
    rule("vent", s["vented_scf"] <= 0.0,
         f"{s['vented_scf']:,.0f} SCF would have to be vented (hard rule 1)", "no venting")
    rule("drive_cap", s["held_back_hr"] <= HELD_BACK_TOLERANCE_HR,
         f"a pressure limit (drive cap / spread pressure / MOP) would hold the pump below the "
         f"scheduled rate for {s['held_back_hr']:.1f} h" + (f" (from MP {s['held_back_mp']:.1f})"
                                                          if s["held_back_mp"] is not None else ""),
         "the pump never has to be held back by a pressure limit")
    rule("slack", s["slack_steps"] <= r["slack_steps"],
         f"slack-line risk on {s['slack_steps']} steps (hard rule 2; the strategy run had "
         f"{r['slack_steps']})",
         "no slack-line risk" if s["slack_steps"] == 0 else
         f"slack-line risk on {s['slack_steps']} steps, no worse than the strategy run")
    rule("stall", s["stall_steps"] <= r["stall_steps"],
         f"the pig stalls on {s['stall_steps']} steps", "the pig never stalls")
    rule("min_speed", s["below_min_steps"] <= r["below_min_steps"],
         f"pig below {cfg.min_speed_mph:g} mph on {s['below_min_steps']} steps (strategy run: "
         f"{r['below_min_steps']})",
         f"pig never below {cfg.min_speed_mph:g} mph" if s["below_min_steps"] == 0 else
         f"pig below {cfg.min_speed_mph:g} mph on {s['below_min_steps']} steps, as in the strategy run")
    rule("max_speed", s["overspeed_steps"] <= r["overspeed_steps"]
         and s["overspeed_mph_max"] <= r["overspeed_mph_max"] + 0.05,
         f"pig over {cfg.max_speed_mph:g} mph on {s['overspeed_steps']} steps, up to "
         f"{s['overspeed_mph_max']:.2f} mph over (strategy run: {r['overspeed_steps']} steps)",
         f"pig never over {cfg.max_speed_mph:g} mph" if s["overspeed_steps"] == 0 else
         f"pig over max speed on {s['overspeed_steps']} steps, no worse than the strategy run")
    limit = r["duration_hr"] * (1.0 + MAX_DURATION_STRETCH) + 0.25
    rule("duration", (not s["completed"]) or s["duration_hr"] <= limit,
         f"takes {s['duration_hr']:.1f} h against {r['duration_hr']:.1f} h for the strategy run",
         f"{s['duration_hr']:.1f} h against {r['duration_hr']:.1f} h for the strategy run")
    return out


# ---------------------------------------------------------------------------
# Fitting a schedule to the reference
# ---------------------------------------------------------------------------

def cumulative_injection(res: SimResults) -> Tuple[np.ndarray, np.ndarray]:
    """(t_hr, SCF injected by t) from the step record, scaled so the end matches the
    run's total exactly."""
    steps = res.steps
    t = np.array([s.t_hr for s in steps], dtype=float)
    t = np.append(t, t[-1] + (t[-1] - t[-2] if len(t) > 1 else 0.0))
    scf = np.array([s.injection_scfm for s in steps], dtype=float) * np.diff(t) * 60.0
    cum = np.concatenate([[0.0], np.cumsum(scf)])
    if cum[-1] > 0 and res.total_scf_injected > 0:
        cum *= res.total_scf_injected / cum[-1]
    return t, cum


def fit_schedule(t: np.ndarray, cum: np.ndarray, end_hr: float, n_steps: int,
                 grid_hr: float = BREAKPOINT_GRID_HR) -> Schedule:
    """The n-piece constant-rate schedule over [0, end_hr] closest (least squares on the
    cumulative N2 curve) to how the reference pumped. Each piece's rate is the chord of
    the cumulative curve between its breakpoints, so the total N2 is the reference's."""
    end_hr = max(end_hr, grid_hr)
    n_grid = max(2, int(round(end_hr / grid_hr)) + 1)
    g = np.linspace(0.0, end_hr, n_grid)
    cg = np.interp(g, t, cum)
    # fine samples for the error integral
    f = np.linspace(0.0, end_hr, max(200, 4 * n_grid))
    cf = np.interp(f, t, cum)
    fi = np.searchsorted(g, f)          # grid index at or after each fine sample

    def cost(a: int, b: int) -> float:
        lo, hi = g[a], g[b]
        m = (fi > a) & (fi <= b)
        if not m.any():
            return 0.0
        x = f[m]
        line = cg[a] + (cg[b] - cg[a]) * (x - lo) / (hi - lo)
        return float(((cf[m] - line) ** 2).sum())

    n = len(g)
    INF = float("inf")
    best = np.full((n_steps + 1, n), INF)
    prev = np.full((n_steps + 1, n), -1, dtype=int)
    best[0, 0] = 0.0
    for k in range(1, n_steps + 1):
        for b in range(k, n):
            for a in range(k - 1, b):
                if best[k - 1, a] == INF:
                    continue
                c = best[k - 1, a] + cost(a, b)
                if c < best[k, b]:
                    best[k, b], prev[k, b] = c, a
    # read the breakpoints back from the end
    pts = [n - 1]
    k = n_steps
    while k > 0:
        pts.append(int(prev[k, pts[-1]]))
        k -= 1
    pts = pts[::-1]
    sched: Schedule = []
    for a, b in zip(pts[:-1], pts[1:]):
        hours = g[b] - g[a]
        if hours <= 0:
            continue
        sched.append((float(hours), float((cg[b] - cg[a]) / hours / 60.0)))
    return sched


def round_schedule(sched: Schedule, max_scfm: float, total_scf: Optional[float] = None) -> Schedule:
    """Quote the rates to RATE_ROUND_SCFM and the durations to HOURS_ROUND; the last step's
    length is trimmed so the total N2 stays within a rounding of total_scf."""
    out: Schedule = []
    for hours, scfm in sched:
        r = min(max_scfm, max(0.0, round(scfm / RATE_ROUND_SCFM) * RATE_ROUND_SCFM))
        h = round(hours / HOURS_ROUND) * HOURS_ROUND
        if h > 0 and r > 0:
            out.append((h, r))
    if out and total_scf:
        have = sum(h * r * 60.0 for h, r in out[:-1])
        h_last, r_last = out[-1]
        if r_last > 0:
            h_last = max(HOURS_ROUND, round((total_scf - have) / (r_last * 60.0) / HOURS_ROUND) * HOURS_ROUND)
        out[-1] = (h_last, r_last)
    return out


def schedule_scf(sched: Schedule) -> float:
    return sum(h * r * 60.0 for h, r in sched)


def schedule_text(sched: Schedule) -> str:
    """'5,700 SCFM for 9.2 h, then stop' / '6,000 SCFM for 4.0 h, then 5,000 SCFM for 5.2 h, then stop'."""
    if not sched:
        return "no injection"
    parts = [f"{r:,.0f} SCFM for {h:.1f} h" for h, r in sched]
    return ", then ".join(parts) + ", then stop"


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

def plan_steady_injection(cfg: SimConfig, reference: Optional[SimResults] = None,
                          max_steps: int = 4, n2_scales: Sequence[float] = N2_SCALES,
                          progress_cb: Optional[Callable[[str, float], None]] = None,
                          keep_results: bool = True) -> RatePlan:
    """Find the constant-rate schedule with the fewest rate changes that holds every rule.

    cfg is the job as the strategy would run it (injection_schedule is ignored); reference
    is that run if the caller already has it. progress_cb(message, fraction) is called
    before each simulation. keep_results keeps the SimResults of the passing attempt on
    RatePlan.best.results (the caller can show / export it as the run)."""
    base = copy.deepcopy(cfg)
    base.injection_schedule = None
    n_total = 1 + max_steps * len(n2_scales)
    done = 0

    def _progress(msg: str):
        if progress_cb:
            progress_cb(msg, done / n_total)

    if reference is None:
        _progress("Running the job the strategy's own way")
        reference = simulate(copy.deepcopy(base))
    done += 1
    ref = run_summary(reference)
    plan = RatePlan(reference=ref, reference_results=reference)
    if not reference.steps or ref["n2_scf"] <= 0:
        return plan

    t, cum = cumulative_injection(reference)
    end_hr = max(ref["injection_hr"], BREAKPOINT_GRID_HR)
    tried = set()
    for n_steps in range(1, max_steps + 1):
        fitted = fit_schedule(t, cum, end_hr, n_steps)
        for scale in n2_scales:
            done += 1
            scaled = [(h, r * scale) for h, r in fitted]
            if any(r > base.max_injection_scfm * 1.001 for _, r in scaled):
                continue    # more than the spread can pump
            sched = round_schedule(scaled, base.max_injection_scfm, ref["n2_scf"] * scale)
            key = tuple(sched)
            if not sched or key in tried:
                continue
            tried.add(key)
            label = (f"{n_steps} rate{'s' if n_steps > 1 else ''}"
                     + (f", {(scale - 1) * 100:+.0f}% N2" if abs(scale - 1) > 1e-9 else ""))
            _progress(f"Trying {label}: {schedule_text(sched)}")
            trial = copy.deepcopy(base)
            trial.injection_schedule = list(sched)
            res = simulate(trial)
            summ = run_summary(res)
            att = Attempt(schedule=list(sched), label=label, n2_scf=summ["n2_scf"],
                          rules=check_rules(summ, ref, base), summary=summ,
                          results=res if keep_results else None)
            plan.attempts.append(att)
            if att.passed:
                plan.best = att
                if not keep_results:
                    att.results = None
                return plan
            if not keep_results:
                att.results = None
    return plan


def plan_text(plan: RatePlan) -> str:
    """The plan in plain words for the pump operator and the LN2 supervisor."""
    r = plan.reference
    lines = []
    if plan.best is None:
        lines.append("No constant-rate schedule passed every rule"
                     + (f" in {len(plan.attempts)} tries" if plan.attempts else "") + ".")
        if plan.attempts:
            last = plan.attempts[-1]
            lines.append(f"Last tried {last.label}: {schedule_text(last.schedule)}; broke: "
                         + "; ".join(last.violations) + ".")
        lines.append(f"The strategy run pumped {r['n2_scf']:,.0f} SCF over {r['injection_hr']:.1f} h "
                     f"and finished in {r['duration_hr']:.1f} h.")
        return "\n".join(lines)
    b = plan.best
    s = b.summary
    n = len(b.schedule)
    lines.append(f"Pump {schedule_text(b.schedule)}.")
    coast = s["duration_hr"] - sum(h for h, _ in b.schedule)
    lines.append(f"Total N2 {b.n2_scf:,.0f} SCF ({b.n2_scf - r['n2_scf']:+,.0f} SCF against the "
                 f"strategy run); purge takes {s['duration_hr']:.1f} h"
                 + (f", the last {coast:.1f} h on the stored column with the pump off" if coast > 0.2 else "")
                 + f"; max pig-face {s['max_face_psig']:.0f} psig.")
    lines.append(("One rate the whole time." if n == 1 else f"{n} rates, {n - 1} change{'s' if n > 2 else ''}.")
                 + f" Checked: " + "; ".join(x.text for x in b.rules) + ".")
    if len(plan.attempts) > 1:
        lines.append("Rejected first: " + " | ".join(
            f"{a.label} ({schedule_text(a.schedule)}): {'; '.join(a.violations)}"
            for a in plan.attempts[:-1]))
    return "\n".join(lines)


def plan_to_dict(plan: RatePlan) -> dict:
    """JSON-safe view for the app and the assistant."""
    def att(a: Attempt) -> dict:
        return {"label": a.label, "schedule": [[h, r] for h, r in a.schedule],
                "schedule_text": schedule_text(a.schedule), "n2_scf": round(a.n2_scf),
                "passed": a.passed, "violations": a.violations,
                "rules": [{"name": x.name, "ok": x.ok, "text": x.text} for x in a.rules],
                "duration_hr": round(a.summary["duration_hr"], 2),
                "max_face_psig": round(a.summary["max_face_psig"], 1)}
    r = plan.reference
    return {
        "text": plan.text,
        "schedule": [[h, s] for h, s in plan.schedule] if plan.schedule else None,
        "schedule_text": schedule_text(plan.schedule) if plan.schedule else None,
        "best": att(plan.best) if plan.best else None,
        "attempts": [att(a) for a in plan.attempts],
        "reference": {"n2_scf": round(r["n2_scf"]), "injection_hr": round(r["injection_hr"], 2),
                      "duration_hr": round(r["duration_hr"], 2), "max_face_psig": round(r["max_face_psig"], 1),
                      "slack_steps": r["slack_steps"], "overspeed_steps": r["overspeed_steps"]},
    }
