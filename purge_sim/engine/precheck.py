"""
Pre-run job check: what kind of purge this is, before the simulator runs.

This is the arithmetic that used to be done by hand (or by a Claude Code session)
for every new job and written into the scenario notes:

  * Is it speed-capped (the drive can hold target speed everywhere), drive-capped
    (somewhere the line needs more drive than the cap allows, so the pig slows),
    or friction-dominated (most of the drive goes to friction, not head)?
  * Laminar or turbulent at run speed?
  * Where do descents pull harder than friction resists (gravity-assisted runs,
    where the exit backpressure holds the pig and slack line is the risk)?
  * Highest flat pack pressure that keeps every joint at or under MOP with the
    column static (Thunderbird: 600 psig, not 750, because the column's head off
    the crest lands on the joints below it).
  * Minimum N2 for pack-and-coast at target speed: the gas has to deliver the
    required drive at every pig position as it expands into the line behind the pig.
  * For volatile products, whether the column stays above vapor pressure + margin.

Everything here is static terrain + steady friction at one speed, the same model
the roadmap and pig solver use (physics.required_drive_to_clear). It is an
estimate to set up and classify the job, not a substitute for the run.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..data.scenario import ScenarioInputs
from .physics import (liquid_friction_loss_psi, mph_to_fts, pipe_area_ft2, required_drive_to_clear,
                      scf_from_pressure_volume)
from .segment_model import PipeGeometry

MAX_GRID = 400
GRAVITY_WINDOW_MI = 0.5


def _geometry(inp: ScenarioInputs) -> PipeGeometry:
    g = PipeGeometry()
    for s in inp.pipe_segments:
        g.segments.append((s["start_mp"], s["end_mp"], s["od_in"], s["wt_in"]))
    return g


def _id_ft(geom: PipeGeometry, mp: float) -> float:
    od, wt = geom.od_wt_at(mp)
    return (od - 2.0 * wt) / 12.0


def _r(v, nd=0):
    if v is None or not math.isfinite(v):
        return None
    return round(float(v), nd) if nd else int(round(float(v)))


def mop_at(inp: ScenarioInputs, mp) -> np.ndarray:
    """Joint MOP along the route (point-by-point joints if present, else the flat MAOP)."""
    mp = np.atleast_1d(np.asarray(mp, dtype=float))
    if inp.mop_joints:
        j = sorted(inp.mop_joints, key=lambda d: d["mp"])
        return np.interp(mp, [d["mp"] for d in j], [d["mop_psig"] for d in j])
    if inp.maop_psig:
        return np.full(mp.shape, float(inp.maop_psig))
    return np.full(mp.shape, math.inf)


def precheck(inp: ScenarioInputs, min_liquid_psig: Optional[float] = None) -> dict:
    """Classify the job and estimate its key numbers. min_liquid_psig: the lowest pressure
    the liquid may see (vapor pressure + margin) for volatile products, else None."""
    if len(inp.elevation_profile) < 2:
        raise ValueError("the scenario has no elevation profile")
    if not inp.pipe_segments:
        raise ValueError("the scenario has no pipe segments")
    prof = np.asarray(inp.elevation_profile, dtype=float)
    start, end = float(inp.purge_start_mp), float(inp.purge_end_mp)
    if end <= start:
        raise ValueError("purge_end_mp must be past purge_start_mp")
    geom = _geometry(inp)
    sg, visc, rough = inp.fluid_sg, inp.fluid_viscosity_cst, inp.fluid_roughness_ft
    grad = sg * 62.4 / 144.0
    elev_at = lambda x: np.interp(x, prof[:, 0], prof[:, 1])

    # terrain vertices over the purge span (the exit is the pig stop, as in the engine)
    sel = (prof[:, 0] >= start) & (prof[:, 0] <= end)
    tm = np.concatenate([[start], prof[sel, 0], [end]])
    te = elev_at(tm)
    tm, idx = np.unique(tm, return_index=True)
    te = te[idx]
    grid = tm if len(tm) <= MAX_GRID else np.linspace(start, end, MAX_GRID)
    grid = grid[grid < end - 1e-6]
    if grid.size == 0:
        grid = np.array([start])
    e_end = float(elev_at(end))
    exit_psig = float(inp.exit_pressure_run_psig)
    v_tgt = mph_to_fts(inp.target_speed_mph)

    # The pig pushes to the next running pump station ahead (its suction), else the BPCV,
    # else the exit at the pig stop: the same order as pump_stations.effective_exit_condition.
    # The BPCV set point is only known during the run; the exit pressure stands in for it.
    stations = sorted((float(p["mp"]), float(p["suction_psig"])) for p in inp.pump_stations)
    bpcv_mp = float(inp.bpcv["mp"]) if inp.bpcv else None
    has_control = bool(stations) or bpcv_mp is not None

    def exit_for(x: float):
        for mp, suc in stations:
            if mp > x:
                return mp, suc
        if bpcv_mp is not None and bpcv_mp > x:
            return bpcv_mp, exit_psig
        return end, exit_psig

    def required(x: float, v: float) -> float:
        xm, xp = exit_for(x)
        return required_drive_to_clear(x, float(elev_at(x)), xm, xp, float(elev_at(xm)), tm, te,
                                       _id_ft(geom, x), v, sg, visc, rough)

    req = np.array([required(x, v_tgt) for x in grid])
    i_peak = int(np.argmax(req))

    # Drive cap at each pig position: the stated limits and the local gas-side MOP. Past a
    # booster site the spread's discharge pressure stands in for the injection limit.
    mop_span = mop_at(inp, tm)
    mop_min = float(np.min(mop_span))
    booster_mps = sorted(float(b["mp"]) for b in inp.booster_stations)

    def cap_at(x: float) -> float:
        caps = [inp.max_drive_psig] if inp.max_drive_psig else []
        if booster_mps and x >= booster_mps[0]:
            caps.append(inp.spread_discharge_psig)
        elif inp.max_injection_psig:
            caps.append(inp.max_injection_psig)
        m = float(mop_at(inp, x)[0])
        if math.isfinite(m):
            caps.append(m * inp.drive_ceiling_fraction)
        return min(caps) if caps else math.inf

    caps_grid = np.array([cap_at(x) for x in grid])
    cap = float(np.min(caps_grid))
    short = req - caps_grid
    i_peak = int(np.argmax(short)) if np.any(short > 0) else int(np.argmax(req))

    # friction and regime at launch
    D0 = _id_ft(geom, start)
    fpf = liquid_friction_loss_psi(1.0, D0, v_tgt, sg, visc, rough)
    reynolds = v_tgt * D0 / (visc * 1.076391e-5) if visc > 0 else math.inf   # cSt -> ft2/s
    fric_launch = fpf * (exit_for(start)[0] - start) * 5280.0
    friction_share = float(fric_launch / req[0]) if req[0] > 0 else 0.0

    tags: List[str] = []
    findings: List[str] = []
    achievable = None
    if not np.any(short > 0):
        tags.append("speed-capped")
        findings.append(f"The drive cap ({_r(caps_grid[i_peak])} psig) covers the peak requirement at {inp.target_speed_mph:g} mph "
                        f"({_r(req[i_peak])} psig at MP {grid[i_peak]:.2f}), so target speed is the binding limit.")
    else:
        tags.append("drive-capped")
        x = float(grid[i_peak])
        cap_x = float(caps_grid[i_peak])
        lo, hi = 0.0, v_tgt
        if required(x, 1e-4) > cap_x:
            findings.append(f"Static head alone at MP {x:.2f} needs {_r(required(x, 1e-4))} psig, above the "
                            f"{_r(cap_x)} psig drive cap: the pig cannot get past it. Raise the cap (if MOP allows), "
                            f"add a booster, or lower the exit pressure.")
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if required(x, mid) <= cap_x else (lo, mid)
            achievable = lo / mph_to_fts(1.0)
            findings.append(f"At MP {x:.2f} holding {inp.target_speed_mph:g} mph needs {_r(req[i_peak])} psig, above the "
                            f"{_r(cap_x)} psig drive cap; the pig slows to about {achievable:.2f} mph there.")
    if friction_share >= 0.5:
        tags.append("friction-dominated")
        findings.append(f"Friction is {friction_share:.0%} of the launch drive ({fpf * 5280:.1f} psi/mi at "
                        f"{inp.target_speed_mph:g} mph), so speed is the main lever on drive and N2.")
    laminar = reynolds < 2300
    if laminar:
        tags.append("laminar")
        findings.append(f"Laminar at run speed (Re about {reynolds:,.0f}): friction scales linearly with speed.")

    # descents steeper than friction: the column pulls the pig, the exit has to hold it
    fine = np.arange(start, end + 1e-9, 0.05)
    w = max(1, int(round(GRAVITY_WINDOW_MI / 0.05)))
    gravity_spans = []
    if fine.size > w:
        ef = elev_at(fine)
        drop_psi_per_mi = grad * (ef[:-w] - ef[w:]) / GRAVITY_WINDOW_MI
        hot = drop_psi_per_mi > fpf * 5280.0
        i = 0
        while i < hot.size:
            if hot[i]:
                j = i
                while j < hot.size and hot[j]:
                    j += 1
                a, b = float(fine[i]), float(fine[min(j - 1 + w, fine.size - 1)])
                gravity_spans.append({"from_mp": round(a, 2), "to_mp": round(b, 2),
                                      "max_net_psi_per_mi": _r(float(np.max(drop_psi_per_mi[i:j])) - fpf * 5280.0, 1)})
                i = j
            else:
                i += 1
        merged = []
        for s in gravity_spans:
            if merged and s["from_mp"] <= merged[-1]["to_mp"]:
                merged[-1]["to_mp"] = max(merged[-1]["to_mp"], s["to_mp"])
                merged[-1]["max_net_psi_per_mi"] = max(merged[-1]["max_net_psi_per_mi"], s["max_net_psi_per_mi"])
            else:
                merged.append(dict(s))
        gravity_spans = merged
    if gravity_spans:
        tags.append("gravity-assisted")
        total = sum(s["to_mp"] - s["from_mp"] for s in gravity_spans)
        findings.append(f"{len(gravity_spans)} descent(s) ({total:.1f} mi) fall faster than friction resists at "
                        f"{inp.target_speed_mph:g} mph; there the exit backpressure holds the pig back and slack "
                        f"line behind the crest is the risk to watch.")

    # highest flat pack that keeps every joint <= MOP with the column static (no friction credit)
    pack_limit = moving_limit = None
    if math.isfinite(mop_min) and not has_control:
        suffix = np.minimum.accumulate((mop_span + grad * te)[::-1])[::-1]
        liq = np.full(te.shape, math.inf)
        liq[:-1] = suffix[1:] - grad * te[:-1]
        both = np.minimum(liq, mop_span)
        k = int(np.argmin(both))
        static_limit = float(both[k])
        # the same with friction credit while the column moves at target speed
        xm = tm * 5280.0
        suffix_f = np.minimum.accumulate((mop_span + grad * te + fpf * xm)[::-1])[::-1]
        liq_f = np.full(te.shape, math.inf)
        liq_f[:-1] = suffix_f[1:] - grad * te[:-1] - fpf * xm[:-1]
        moving_limit = float(np.min(np.minimum(liq_f, mop_span)))
        pack_limit = min(static_limit, cap)
        if static_limit < min(cap, mop_min) - 1:
            findings.append(f"Highest flat pack pressure that keeps every joint at or under MOP: {_r(static_limit)} psig "
                            f"({static_limit / mop_min:.0%} of MOP) with the column at rest, set by the pig at MP "
                            f"{tm[k]:.2f} (its head lands on the low joints ahead); about {_r(min(moving_limit, mop_min))} "
                            f"psig while it moves at {inp.target_speed_mph:g} mph and friction takes some of that head.")

    # N2: line volume and the minimum pack-and-coast budget at target speed
    seg_vol = 0.0
    for s0, s1, od, wt in geom.segments:
        a, b = max(s0, start), min(s1, end)
        if b > a:
            seg_vol += pipe_area_ft2(od, wt) * (b - a) * 5280.0
    vol_behind = np.array([sum(pipe_area_ft2(od, wt) * max(0.0, min(s1, x) - max(s0, start)) * 5280.0
                               for s0, s1, od, wt in geom.segments) for x in grid])
    need_scf = np.array([scf_from_pressure_volume(p + 14.7, v, inp.n2_temperature_f) if v > 0 else 0.0
                         for p, v in zip(req, vol_behind)])
    end_scf = scf_from_pressure_volume(exit_psig + 14.7, seg_vol, inp.n2_temperature_f)
    i_bind = int(np.argmax(need_scf))
    budget = max(float(need_scf[i_bind]), end_scf)

    # volatile products: liquid pressure at the peaks when the pig nears the end
    vapor = None
    if min_liquid_psig is not None:
        # liquid pressure at each point while the column flows to the exit at target speed
        fpf_x = np.array([liquid_friction_loss_psi(1.0, _id_ft(geom, x), v_tgt, sg, visc, rough) for x in tm])
        flowing = exit_psig + grad * (e_end - te) + fpf_x * (end - tm) * 5280.0
        j = int(np.argmin(flowing))
        vapor = {"min_liquid_psig": _r(min_liquid_psig), "exit_psig": _r(exit_psig),
                 "lowest_liquid_psig": _r(float(flowing[j])), "at_mp": round(float(tm[j]), 2),
                 "ok": exit_psig >= min_liquid_psig and float(flowing[j]) >= min_liquid_psig}
        if not vapor["ok"]:
            findings.append(f"Volatile product: the column must stay at or above {_r(min_liquid_psig)} psig. "
                            f"With {_r(exit_psig)} psig at the exit, the liquid at MP {tm[j]:.2f} drops to about "
                            f"{_r(float(flowing[j]))} psig while flowing at {inp.target_speed_mph:g} mph; raise the "
                            f"exit pressure. The simulator's own slack check only holds 25 psig at peaks.")

    # Pack-and-coast needs room to pack above what the line needs to keep moving.
    pack_ok = bool(pack_limit and pack_limit > float(req[0]))
    if pack_ok:
        findings.append(f"Pack-and-coast is an option: pack up to {_r(pack_limit)} psig against {_r(req[0])} psig "
                        f"needed at launch. Holding {inp.target_speed_mph:g} mph all the way takes at least "
                        f"{budget / 1e6:.2f} MMscf (binds at MP {grid[i_bind]:.1f}); letting the pig slow on the "
                        f"coast takes less, so sweep n2_budget_scf around that.")
    if has_control:
        tags.append("pump/BPCV controlled")

    return {
        "span_mi": round(end - start, 3),
        "exit_mp": round(end, 3),
        "exit_psig": _r(exit_psig),
        "line_volume_bbl": _r(seg_vol / 5.6146),
        "line_volume_ft3": _r(seg_vol),
        "reynolds": _r(reynolds),
        "laminar": laminar,
        "friction_psi_per_mi": _r(fpf * 5280.0, 1),
        "friction_share_at_launch": round(friction_share, 2),
        "exit_note": ("next pump station / BPCV ahead of the pig, then the pig stop" if has_control
                      else "the pig stop (purge_end_mp), as in the simulator"),
        "required_drive_psig": {"at_launch": _r(req[0]), "peak": _r(req[i_peak]), "peak_mp": round(float(grid[i_peak]), 2)},
        "drive_cap_psig": _r(cap),
        "mop_min_psig": _r(mop_min),
        "achievable_speed_at_peak_mph": round(achievable, 2) if achievable is not None else None,
        "static_pack_limit_psig": _r(pack_limit),
        "moving_pack_limit_psig": _r(min(moving_limit, cap)) if pack_limit is not None else None,
        "gravity_spans": gravity_spans[:8],
        "n2_min_coast_budget_scf": _r(budget, -3) if budget else None,
        "n2_budget_binds_at_mp": round(float(grid[i_bind]), 2),
        "n2_fill_at_exit_pressure_scf": _r(end_scf, -3),
        "vapor_check": vapor,
        "job_type": tags,
        "pack_and_coast_possible": pack_ok,
        "findings": findings,
    }


def exit_pressure_at_stop(inp: ScenarioInputs, stop_mp: float, exit_mp: float, delivery_psig: float) -> float:
    """Pressure the line holds at the pig stop when the liquid actually leaves further on.

    The simulator puts the exit at the pig stop (purge_end_mp). When the product keeps
    going past it (Laurel: pig stops at the Allendale BV, MP 4.5; diesel goes on to the
    Billings tank farm, MP 33.3), the stretch beyond still has to be pushed through: the
    pig stop sees the delivery pressure plus that stretch's friction at run rate and
    whatever head it takes to get the column over the terrain in between."""
    prof = np.asarray(inp.elevation_profile, dtype=float)
    if exit_mp > float(prof[-1, 0]) + 1e-6:
        raise ValueError(f"the elevation profile ends at MP {prof[-1, 0]:.3f}, before the exit at MP {exit_mp:.3f}")
    if exit_mp <= stop_mp:
        return float(delivery_psig)
    geom = _geometry(inp)
    elev_at = lambda x: float(np.interp(x, prof[:, 0], prof[:, 1]))
    sel = (prof[:, 0] > stop_mp) & (prof[:, 0] <= exit_mp)
    tm, te = prof[sel, 0], prof[sel, 1]
    od0, wt0 = geom.od_wt_at(inp.purge_start_mp)
    od1, wt1 = geom.od_wt_at(min(exit_mp, stop_mp + 1e-3))
    v = mph_to_fts(inp.target_speed_mph) * pipe_area_ft2(od0, wt0) / pipe_area_ft2(od1, wt1)
    return required_drive_to_clear(stop_mp, elev_at(stop_mp), exit_mp, delivery_psig, elev_at(exit_mp),
                                   tm, te, (od1 - 2 * wt1) / 12.0, v, inp.fluid_sg,
                                   inp.fluid_viscosity_cst, inp.fluid_roughness_ft)
