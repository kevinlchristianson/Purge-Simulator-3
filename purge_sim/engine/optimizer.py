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
    margin_psi: float = 1.0,
) -> BoosterSitePlan:
    """Greedy minimal-booster siting over the pre-run roadmap.

    Walk the route upstream->downstream tracking a `source` (the active pressure
    origin). The most the source can drive at point x without a new booster is
    the *minimum joint MOP between the source and x* (a connected gas column is
    ~uniform, so its weakest joint caps it). The drive the pig needs at x is
    roadmap.floor_at(x). Where required floor exceeds deliverable MOP, the source
    can't reach: drop a booster at the LAST candidate station before x (greatest
    reach for the previous source), reset the source to it, and continue. This is
    the classic minimum-stops interval cover and yields the fewest boosters that
    keep the column legal up every climb.

    Returns a BoosterSitePlan. `uncovered` lists points no station can relieve
    (genuine infeasibility — same spots the roadmap flags floor > ceiling).
    """
    mp_grid = np.asarray(roadmap.mp, dtype=float)
    mop_grid = np.asarray(roadmap.mop_psig, dtype=float)
    floor = np.asarray(roadmap.floor_psig, dtype=float)
    n = len(mp_grid)

    stations = sorted(s for s in station_mps if start_mp < s < end_mp)

    def min_mop_between(a: float, b: float) -> float:
        m = (mp_grid >= a) & (mp_grid <= b)
        return float(np.min(mop_grid[m])) if m.any() else math.inf

    plan = BoosterSitePlan()
    placed: set[float] = set()
    source = start_mp
    i = 0
    while i < n:
        x = float(mp_grid[i])
        if x <= source:
            i += 1
            continue
        deliverable = min_mop_between(source, x)
        required = float(floor[i])
        if required > deliverable + margin_psi:
            cand = [s for s in stations if source < s < x and s not in placed]
            if not cand:
                plan.uncovered.append((x, required, deliverable))
                i += 1   # cannot relieve here; record and move on
                continue
            site = max(cand)             # place as late as possible (max reach)
            placed.add(site)
            plan.sites_mp.append(site)
            plan.reasons[site] = (
                f"relieves gas-MOP bottleneck by MP {x:.1f}: floor {required:.0f} psi "
                f"> deliverable {deliverable:.0f} psi from MP {source:.1f}"
            )
            source = site
            # re-test the same x against the new, closer source
        else:
            i += 1

    plan.sites_mp.sort()
    if not plan.sites_mp:
        plan.notes.append("No booster needed — SP injection covers the whole route "
                          "within MOP (no gas-side bottleneck).")
    else:
        plan.notes.append(
            f"{len(plan.sites_mp)} booster site(s) selected: "
            + ", ".join(f"MP {m:.1f}" for m in plan.sites_mp))
    if plan.uncovered:
        worst = max(plan.uncovered, key=lambda u: u[1] - u[2])
        plan.notes.append(
            f"{len(plan.uncovered)} point(s) cannot be relieved by any station "
            f"(worst MP {worst[0]:.1f}: short {worst[1]-worst[2]:.0f} psi) — "
            f"add a station, lower target speed, or re-batch.")
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
