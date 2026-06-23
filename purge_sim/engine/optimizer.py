"""
Spread deployment optimizer.

Given a finite number of booster spreads and a set of booster stations,
finds the optimal assignment of spreads to stations and mobilization timing.

Objective: minimize total N2 injection (cost) while keeping pig moving at
target speed throughout the purge.

Constraints:
  - Each spread occupies one station at a time
  - Spreads move forward only (cannot go back upstream of pig)
  - Mobilization takes mob_time_hr (travel + rig-up) per move
  - A spread can pre-position at a future station before the pig arrives
  - A spread cannot activate until it's fully rigged up at its station

Strategy:
  1. Run the backward pass to find minimum N2 floor and which stations are load-bearing
  2. For each load-bearing station, compute when the pig will arrive
  3. Assign spreads greedily: earliest-arriving load-bearing station gets the first spread
  4. Schedule mobilization so the spread arrives before the pig (with margin)
  5. If N_spreads < N_load_bearing_stations, prioritize by N2 contribution

This is a heuristic optimizer, not a global search. For more optimal solutions,
the user can manually adjust the plan based on the optimizer's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math

import numpy as np

from .backward_pass import BackwardPassResult, BackwardPassConfig, run_backward_pass
from .booster import BoosterConfig, BoosterSpread


@dataclass
class BoosterSitePlan:
    """Which booster stations actually earn a spread, derived from the roadmap.

    A booster is only worth placing where the gas column from the current
    pressure source (SP injection, or the last upstream booster) CANNOT deliver
    the drive the pig needs without exceeding the MOP of some joint between the
    source and the pig — i.e. a gas-MOP bottleneck on a climb. Stations on flat
    or downhill stretches, where the floor is low and gravity helps, are skipped.
    This mirrors the reference model, which staged compression up the long
    SP->summit grade (NW partway, LS at the crest) and ran NO boosters on the
    descent.
    """
    sites_mp: List[float] = field(default_factory=list)
    reasons: Dict[float, str] = field(default_factory=dict)   # mp -> why placed
    uncovered: List[Tuple[float, float, float]] = field(default_factory=list)
    # each: (mp, floor_required, deliverable_mop) where no station could relieve
    notes: List[str] = field(default_factory=list)


def plan_booster_sites(
    roadmap,
    station_mps: List[float],
    start_mp: float,
    end_mp: float,
    max_span_mi: float = 75.0,
    margin_psi: float = 1.0,
) -> BoosterSitePlan:
    """Crest-anchored recompression siting over the pre-run roadmap.

    Recompression stations earn a spread for two physical reasons the bare
    gas-MOP feasibility test misses:

      (1) CREST ISOLATION — a booster boundary at the last station before the
          route's summit separates the high-pressure climb bay from the low-MOP
          descent, so climb gas isn't dragged into descent pipe. -> LS near the
          Tug Mtn summit.
      (2) RECOMPRESSION REACH — no single stage should span too much pipe. A pig
          driven from a far-back booster pins a huge trailing segment at high
          pressure (the descent over-build). Staging a booster near the midpoint
          of any long span keeps each stage's pig-face control local and the
          column lean. -> NW partway up the climb, HW partway down the descent.

    Algorithm: anchor a booster at the last station <= summit (crest isolation),
    then recursively bisect every span longer than `max_span_mi`, placing the
    station nearest each span's midpoint, until no span exceeds the reach. On this
    route (SP=0, summit ~113.6, MT=236) this yields NW(52) + LS(103) + HW(166) —
    matching the reference, which boosted exactly there.

    Falls back to an empty plan (caller then uses the legacy "every passed
    station" behavior) only when there are no candidate stations.
    """
    mp_grid = np.asarray(roadmap.mp, dtype=float)
    elev = np.asarray(roadmap.elev_ft, dtype=float)

    stations = sorted(s for s in station_mps if start_mp < s < end_mp)
    plan = BoosterSitePlan()
    if not stations:
        plan.notes.append("No candidate booster stations between source and exit — "
                          "falling back to configured boosters.")
        return plan

    summit_mp = float(mp_grid[int(np.argmax(elev))]) if len(elev) else end_mp
    placed: set[float] = set()

    def nearest_station(target: float, lo: float, hi: float) -> Optional[float]:
        cand = [s for s in stations if lo < s < hi and s not in placed]
        return min(cand, key=lambda s: abs(s - target)) if cand else None

    # (1) Crest isolation: last station at or before the summit.
    crest_cands = [s for s in stations if s <= summit_mp]
    if crest_cands:
        crest = max(crest_cands)
        placed.add(crest)
        plan.reasons[crest] = (
            f"crest isolation: last station before the summit (MP {summit_mp:.0f}) — "
            f"separates the climb bay from the low-MOP descent")

    # (2) Recompression reach: recursively stage spans longer than max_span_mi.
    def stage(lo: float, hi: float) -> None:
        if hi - lo <= max_span_mi:
            return
        mid = 0.5 * (lo + hi)
        s = nearest_station(mid, lo, hi)
        if s is None:
            return
        placed.add(s)
        plan.reasons[s] = (
            f"recompression staging: nearest station to the midpoint of the "
            f"{hi - lo:.0f}-mi MP {lo:.0f}-{hi:.0f} span (max reach {max_span_mi:.0f} mi)")
        stage(lo, s)
        stage(s, hi)

    anchors = sorted({start_mp, end_mp} | placed)
    for a, b in zip(anchors, anchors[1:]):
        stage(a, b)

    plan.sites_mp = sorted(placed)
    if plan.sites_mp:
        plan.notes.append(
            f"{len(plan.sites_mp)} booster site(s) selected: "
            + ", ".join(f"MP {m:.1f}" for m in plan.sites_mp))
    else:
        plan.notes.append("No booster site met the crest/reach criteria — "
                          "falling back to configured boosters.")
    return plan


@dataclass
class StationAssignment:
    """One spread assigned to one station."""
    spread_id: int
    station_mp: float
    station_name: str
    pig_arrival_hr: Optional[float]      # when pig is expected to reach this station
    mob_start_hr: float                  # when this spread should start mobilizing
    arrive_at_station_hr: float          # when spread arrives and is rigged up
    margin_hr: float                     # how early the spread arrives before pig


@dataclass
class SpreadPlan:
    """Complete optimizer output — one spread deployment plan."""
    n_spreads: int
    mob_time_hr: float
    assignments: List[StationAssignment] = field(default_factory=list)
    unassigned_stations: List[float] = field(default_factory=list)   # mps not covered
    notes: List[str] = field(default_factory=list)

    @property
    def covered_stations_mp(self) -> List[float]:
        return [a.station_mp for a in self.assignments]


@dataclass
class OptimizerConfig:
    """Inputs for the spread deployment optimizer."""
    n_spreads: int
    mob_time_hr: float                   # time per move (travel + rig-up)
    booster_configs: List[BoosterConfig]

    # Pig speed estimate for timing (mph) — use target speed
    pig_speed_mph: float = 3.0

    # Pre-positioning margin: how many hours BEFORE pig arrival must spread be ready
    min_margin_hr: float = 2.0

    # Backward pass result to identify load-bearing stations
    backward_pass: Optional[BackwardPassResult] = None

    # If backward pass is not pre-computed, we need these to run it internally
    backward_pass_cfg: Optional[BackwardPassConfig] = None


def run_optimizer(cfg: OptimizerConfig) -> SpreadPlan:
    """
    Greedy spread deployment optimizer.

    Returns a SpreadPlan with assignments for each spread.
    """
    # Get or compute backward pass
    bp = cfg.backward_pass
    if bp is None and cfg.backward_pass_cfg is not None:
        bp = run_backward_pass(cfg.backward_pass_cfg)

    critical_mps = sorted(bp.critical_boosters) if bp else sorted(
        b.mp for b in cfg.booster_configs
    )

    if not critical_mps:
        return SpreadPlan(
            n_spreads=cfg.n_spreads,
            mob_time_hr=cfg.mob_time_hr,
            notes=["No load-bearing stations found — no booster deployment needed"],
        )

    # Estimate pig arrival time at each station (from purge start at 0 hrs)
    pig_arrival_hr: Dict[float, float] = {}
    for mp in critical_mps:
        pig_arrival_hr[mp] = mp / max(0.001, cfg.pig_speed_mph)

    # Sort stations by arrival time
    stations_by_arrival = sorted(critical_mps, key=lambda mp: pig_arrival_hr[mp])

    # Initialize spreads
    spreads: List[BoosterSpread] = [
        BoosterSpread(spread_id=i, current_mp=0.0, available_at_hr=0.0)
        for i in range(cfg.n_spreads)
    ]

    assignments: List[StationAssignment] = []
    unassigned: List[float] = []

    for station_mp in stations_by_arrival:
        pig_arrives_hr = pig_arrival_hr[station_mp]
        deadline_hr    = pig_arrives_hr - cfg.min_margin_hr

        # Find the spread that can reach this station earliest
        best_spread = None
        best_arrive_hr = math.inf
        best_mob_start_hr = math.inf

        for spread in spreads:
            current = spread.current_mp if spread.current_mp is not None else 0.0
            if current > station_mp:
                continue  # can't go backward
            earliest_mob = spread.available_at_hr
            arrive_hr = earliest_mob + cfg.mob_time_hr
            if arrive_hr < best_arrive_hr:
                best_arrive_hr  = arrive_hr
                best_mob_start_hr = earliest_mob
                best_spread = spread

        if best_spread is None or best_arrive_hr > pig_arrives_hr:
            unassigned.append(station_mp)
            note = (f"Station MP {station_mp:.2f}: no spread can arrive before pig "
                    f"(pig arrives {pig_arrives_hr:.1f}h, earliest spread {best_arrive_hr:.1f}h)")
            continue

        # Check if we need to mobilize early to meet the margin deadline
        mob_start_hr = max(0.0, deadline_hr - cfg.mob_time_hr)
        arrive_hr    = mob_start_hr + cfg.mob_time_hr
        margin_hr    = pig_arrives_hr - arrive_hr

        # Assign
        booster_name = next(
            (b.name for b in cfg.booster_configs if abs(b.mp - station_mp) < 0.01),
            f"MP {station_mp:.2f}"
        )
        assignments.append(StationAssignment(
            spread_id=best_spread.spread_id,
            station_mp=station_mp,
            station_name=booster_name,
            pig_arrival_hr=pig_arrives_hr,
            mob_start_hr=mob_start_hr,
            arrive_at_station_hr=arrive_hr,
            margin_hr=margin_hr,
        ))

        # Update spread state
        best_spread.destination_mp = station_mp
        best_spread.current_mp = station_mp
        best_spread.available_at_hr = arrive_hr

    notes = []
    if unassigned:
        notes.append(
            f"Stations at MP {[f'{mp:.2f}' for mp in unassigned]} could not be covered. "
            f"Consider adding more spreads or pre-positioning earlier."
        )
    if len(assignments) == len(stations_by_arrival):
        notes.append("All load-bearing stations covered.")

    return SpreadPlan(
        n_spreads=cfg.n_spreads,
        mob_time_hr=cfg.mob_time_hr,
        assignments=sorted(assignments, key=lambda a: a.station_mp),
        unassigned_stations=unassigned,
        notes=notes,
    )


def format_spread_plan(plan: SpreadPlan) -> str:
    """Human-readable summary of the spread deployment plan."""
    lines = [
        f"Spread Plan ({plan.n_spreads} spread(s), {plan.mob_time_hr:.1f}h mob time each)",
        "=" * 60,
    ]
    for a in plan.assignments:
        lines.append(
            f"  Spread {a.spread_id+1} → {a.station_name} (MP {a.station_mp:.2f})\n"
            f"    Mobilize at: {a.mob_start_hr:.1f}h  |  "
            f"Arrive: {a.arrive_at_station_hr:.1f}h  |  "
            f"Pig arrives: {a.pig_arrival_hr:.1f}h  |  "
            f"Margin: {a.margin_hr:.1f}h"
        )
    if plan.unassigned_stations:
        lines.append(f"\n  UNCOVERED stations: {[f'MP {mp:.2f}' for mp in plan.unassigned_stations]}")
    for note in plan.notes:
        lines.append(f"\n  Note: {note}")
    return '\n'.join(lines)
