"""
Export a SimResults run log to a structured text file.

Format is human-readable in the header/summary sections and CSV in the step
data section so it can be parsed or pasted directly for analysis.
"""

from __future__ import annotations

import datetime
from typing import Optional

from .simulator import SimResults, SimConfig


def export_run_log(results: SimResults, path: str, scenario_name: str = "") -> None:
    """
    Write a full run log to `path`.

    Sections:
      1. Header (date, scenario, status)
      2. Configuration summary
      3. Run summary (metrics)
      4. Station shutdown events
      5. Booster activation events
      6. Step-by-step CSV (time, position, speed, pressures, MOP counts)
    """
    cfg = results.config
    lines = []

    # -----------------------------------------------------------------------
    # 1. Header
    # -----------------------------------------------------------------------
    lines += [
        "=" * 64,
        "PURGE SIMULATOR v30 — RUN LOG",
        "=" * 64,
        f"Scenario:    {scenario_name or '(unnamed)'}",
        f"Exported:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Status:      {'COMPLETED' if results.completed else 'ABORTED — ' + results.abort_reason}",
        "",
    ]

    # -----------------------------------------------------------------------
    # 2. Configuration summary
    # -----------------------------------------------------------------------
    bpcv_str = (
        f"MP {cfg.bpcv.mp:.4f}, {cfg.bpcv.elevation_ft:.1f} ft" if cfg.bpcv else "none"
    )
    station_names = ", ".join(s.name or f"MP {s.mp:.2f}" for s in cfg.pump_stations)
    booster_names = ", ".join(b.name or f"MP {b.mp:.2f}" for b in cfg.booster_configs)

    lines += [
        "CONFIGURATION",
        f"  Purge route:       MP {cfg.purge_start_mp:.3f} → {cfg.purge_end_mp:.3f}",
        f"  Target speed:      {cfg.target_speed_mph} mph",
        f"  Speed limits:      {cfg.min_speed_mph} – {cfg.max_speed_mph} mph",
        f"  N2 initial P:      {cfg.n2_initial_pressure_psig:.1f} psig  (auto-calculated)",
        f"  Max injection:     {cfg.max_injection_psig:.1f} psig / {cfg.max_injection_scfm:.0f} SCFM",
        f"  MAOP:              {cfg.maop_psig:.1f} psig",
        f"  Max drive:         {cfg.max_drive_psig:.1f} psig",
        f"  Fluid:             SG={cfg.fluid_sg}, visc={cfg.fluid_viscosity_cst} cSt, rough={cfg.fluid_roughness_ft} ft",
        f"  Exit condition:    {cfg.exit_pressure_behavior} {cfg.exit_pressure_run_psig:.1f} psig",
        f"  BPCV:              {bpcv_str}",
        f"  Pump stations ({len(cfg.pump_stations)}): {station_names}",
        f"  Booster stations ({len(cfg.booster_configs)}): {booster_names}",
        f"  Check valves:      {len(cfg.check_valves)}",
        f"  MOP joints:        {len(cfg.mop_joints)}",
        "",
    ]

    # -----------------------------------------------------------------------
    # 3. Run summary
    # -----------------------------------------------------------------------
    if results.steps:
        last = results.steps[-1]
        duration_hr = last.t_hr
        distance_mi = last.pig_mp - cfg.purge_start_mp
        speeds = [s.pig_speed_mph for s in results.steps]
        faces  = [s.pig_face_psig  for s in results.steps]
        total_viol   = sum(s.mop_violations for s in results.steps)
        total_warn   = sum(s.mop_warnings   for s in results.steps)
        avg_speed = sum(speeds) / len(speeds)
        avg_face  = sum(faces)  / len(faces)
    else:
        duration_hr = distance_mi = avg_speed = avg_face = 0.0
        total_viol = total_warn = 0

    lines += [
        "RUN SUMMARY",
        f"  Duration:          {duration_hr:.2f} hr",
        f"  Distance covered:  {distance_mi:.2f} mi  ({100*distance_mi/max(1e-9,cfg.purge_end_mp-cfg.purge_start_mp):.1f}% of route)",
        f"  Total N2 injected: {results.total_scf_injected:,.0f} SCF",
        f"  Steps recorded:    {len(results.steps):,}",
        f"  Wall clock:        {results.wall_time_s:.1f} s",
        f"  Avg pig speed:     {avg_speed:.2f} mph",
        f"  Min pig speed:     {min(speeds):.2f} mph" if results.steps else "  Min pig speed:     n/a",
        f"  Max pig speed:     {max(speeds):.2f} mph" if results.steps else "  Max pig speed:     n/a",
        f"  Avg pig-face P:    {avg_face:.1f} psig",
        f"  Max pig-face P:    {max(faces):.1f} psig" if results.steps else "  Max pig-face P:    n/a",
        f"  MOP violations:    {total_viol:,}  (joint-steps)",
        f"  MOP warnings:      {total_warn:,}  (joint-steps)",
        "",
    ]

    # -----------------------------------------------------------------------
    # 4. Station shutdown events
    # -----------------------------------------------------------------------
    lines.append("STATION SHUTDOWNS")
    shutdown_logged = []
    for step in results.steps:
        for evt in step.station_shutdown_events:
            shutdown_logged.append(
                f"  t={step.t_hr:7.2f}h  MP {evt['mp']:7.3f}  {evt.get('name',''):<6}  "
                f"{evt['reason']:<28}  pig_dist={evt.get('distance_to_pig_mi', 0):.2f} mi"
                + (
                    f"  avail={evt.get('available_drive_psig',0):.1f} req={evt.get('required_drive_psig',0):.1f} psig"
                    if 'available_drive_psig' in evt else ""
                )
            )
    if shutdown_logged:
        lines += shutdown_logged
    else:
        lines.append("  (none)")
    lines.append("")

    # -----------------------------------------------------------------------
    # 5a. Spread mobilization events
    # -----------------------------------------------------------------------
    if results.spread_events:
        lines.append("SPREAD MOBILIZATIONS")
        for ev in results.spread_events:
            lines.append(
                f"  Spread {ev['spread_id']+1}  MP {ev['from_mp']:.2f} → MP {ev['to_mp']:.2f}  "
                f"t={ev['t_hr']:.2f}h  arrive={ev['arrive_at_hr']:.2f}h  [{ev['reason']}]"
            )
        lines.append("")

    # -----------------------------------------------------------------------
    # 5b. Booster events
    # -----------------------------------------------------------------------
    lines.append("BOOSTER EVENTS  (first activation per station)")
    seen_boosters: set = set()
    for step in results.steps:
        for bs in step.booster_states:
            key = bs.get('mp', bs.get('name', ''))
            if bs.get('running') and key not in seen_boosters:
                seen_boosters.add(key)
                note = "  *** FLOW LIMITED — add compressors ***" if bs.get('flow_limited') else ""
                lines.append(
                    f"  t={step.t_hr:7.2f}h  MP {bs.get('mp', '?'):7.3f}  "
                    f"{bs.get('name',''):<12}  ACTIVATED  "
                    f"suction={bs.get('suction_psig',0):.1f} psig  "
                    f"flow={bs.get('flow_scfm',0):.0f} SCFM{note}"
                )
    if not seen_boosters:
        lines.append("  (no boosters ran)")
    lines.append("")

    # -----------------------------------------------------------------------
    # 6. Step data CSV
    # -----------------------------------------------------------------------
    lines += [
        "STEP DATA (CSV)",
        "time_hr,pig_mp,pig_speed_mph,pig_face_psig,injection_scfm,injection_psig,"
        "exit_psig,exit_mp,exit_description,n_mop_violations,n_mop_warnings,"
        "slack_line,meter_valve,total_scf",
    ]
    for s in results.steps:
        exit_desc_clean = s.exit_description.replace(",", ";")
        lines.append(
            f"{s.t_hr:.4f},{s.pig_mp:.4f},{s.pig_speed_mph:.4f},"
            f"{s.pig_face_psig:.2f},{s.injection_scfm:.1f},{s.injection_psig:.2f},"
            f"{s.exit_psig:.2f},{s.exit_mp:.4f},{exit_desc_clean},"
            f"{s.mop_violations},{s.mop_warnings},"
            f"{int(s.slack_line_risk)},{int(s.meter_valve_active)},"
            f"{s.total_scf:.0f}"
        )

    # -----------------------------------------------------------------------
    # 7. Elevation profile (static reference)
    # -----------------------------------------------------------------------
    if cfg.elevation_profile is not None and len(cfg.elevation_profile) > 0:
        lines += [
            "",
            "ELEVATION PROFILE (CSV)",
            "mp,elev_ft",
        ]
        for row in cfg.elevation_profile:
            lines.append(f"{float(row[0]):.4f},{float(row[1]):.2f}")

    # -----------------------------------------------------------------------
    # 8. MOP profile (static reference)
    # -----------------------------------------------------------------------
    if cfg.mop_joints:
        lines += [
            "",
            "MOP PROFILE (CSV)",
            "mp,mop_psig",
        ]
        for j in cfg.mop_joints:
            lines.append(f"{j.mp:.4f},{j.mop_psig:.1f}")

    # -----------------------------------------------------------------------
    # 9. Per-step pipeline pressure profiles
    #    Gas side: N2 segment pressures at each boundary (mp, psig)
    #    PIG_FACE: pressure right at pig (N2 side)
    #    EXIT: exit condition pressure
    #    INJECT: injection pressure at origin
    #    BOOST_SUCT / BOOST_DISC: active booster suction and discharge
    # -----------------------------------------------------------------------
    lines += [
        "",
        "PIPELINE PRESSURE PROFILES (CSV)",
        "# Use with ELEVATION PROFILE and MOP PROFILE to reconstruct HGL.",
        "# head_ft = pressure_psig / (SG * 0.433) + elev_ft  (liquid side)",
        "# N2 gas side: each segment is uniform pressure (step-function profile).",
        "time_hr,pig_mp,mp,pressure_psig,side",
    ]
    for s in results.steps:
        t   = s.t_hr
        pm  = s.pig_mp
        # Injection point at origin
        lines.append(
            f"{t:.4f},{pm:.4f},{cfg.purge_start_mp:.4f},{s.injection_psig:.2f},INJECT"
        )
        # Gas profile: segment boundaries, sorted by mp (fix the psig-sort bug in builder)
        for mp, psi in sorted(s.gas_pressure_profile, key=lambda x: x[0]):
            lines.append(f"{t:.4f},{pm:.4f},{mp:.4f},{psi:.2f},GAS")
        # Pig face (right at pig)
        lines.append(f"{t:.4f},{pm:.4f},{pm:.4f},{s.pig_face_psig:.2f},PIG_FACE")
        # Exit condition
        lines.append(f"{t:.4f},{pm:.4f},{s.exit_mp:.4f},{s.exit_psig:.2f},EXIT")
        # Active booster suction / discharge
        for bs in s.booster_states:
            if bs.get('running'):
                bmp = bs.get('mp', 0.0)
                lines.append(
                    f"{t:.4f},{pm:.4f},{bmp:.4f},{bs.get('suction_psig', 0.0):.2f},BOOST_SUCT"
                )
                lines.append(
                    f"{t:.4f},{pm:.4f},{bmp:.4f},{bs.get('discharge_psig', 0.0):.2f},BOOST_DISC"
                )

    # -----------------------------------------------------------------------
    # Write file
    # -----------------------------------------------------------------------
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
