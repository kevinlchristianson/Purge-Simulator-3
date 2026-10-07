# Purge Simulator — Project Context for Claude Code

This file is read automatically by Claude Code as persistent project context. Keep it
honest and current — if you (Claude) make a structural change that invalidates something
here, update this file in the same session, don't leave it stale.

---

## What This Project Is

**Purge Simulator v30** — a Python/Tkinter desktop app that models nitrogen-displacement
("purge") operations on liquid pipelines: a pig driven by N2 gas pressure displaces the
liquid product ahead of it.

It started as a rebuild of one specific reference pipeline (a 24" crude line, Portland ME
→ Montreal QC, ~236 mi, "SP→MT") reverse-engineered from a client animation. It has since
been used, unmodified in its core engine, to model **11 real, distinct client pipeline
purges** — different diameters (3"–24"), different fluids (crude, diesel, butane, NGL),
different constraints (speed-capped, drive-capped, friction-dominated, pump-and-BPCV
jobs). See the memory index for the full list of jobs and what each one taught the engine.

Entry points:
- `python app.py` — the standalone app: a local server + browser UI with the in-app Claude
  assistant (`purge_sim/app/`). Packages to a no-Python-needed program with
  `packaging/purge_simulator.spec` (PyInstaller).
- `python main.py` — launches the Tkinter desktop app
- `python run_headless.py <scenario.json>` — runs a scenario with no GUI; this is how
  every real client job has actually been run so far. Auto-generates an xlsx report and
  animation GIF as a side effect.

The v29 predecessor (`Purge_Modeling_Program_v29.py`, a single ~4,800-line file) is fully
superseded and archived outside the repo at `C:\Users\kevin\PurgeSimArchive\` — it is no
longer a reference for current behavior.

---

## Architecture

```
purge_sim/
  engine/
    simulator.py      — simulate(config) -> SimResults. Main timestep loop: pig position,
                         N2 segment inventories, check-valve cascade, booster transfers,
                         pump shutdown eval, BPCV state, per-joint MOP check, injection rate.
    physics.py         — pure, stateless physics functions (friction, Z-factor, haversine, etc.)
    constants.py       — NPS tables, roughness, fluid/N2 property lookups
    segment_model.py   — N2Segment: gas column divided into segments by check valves/boosters,
                         each an isolated SCF inventory; pressure via Peng-Robinson EOS
    check_valve.py     — mainline check-valve state machine (open when upstream P >= downstream P)
    booster.py         — booster compressor logic: draws upstream N2, discharges downstream,
                         holds its mainline check valve closed while running
    bpcv.py            — Back Pressure Control Valve: set point = min across downstream joints
                         of (MOP - static head - friction to that joint)
    pump_stations.py   — pump shutdown logic: shuts down when no longer hydraulically
                         necessary, hard 1-mile safety limit ahead of the pig
    pig_solver.py       — bisects pig velocity so N2 drive pressure = liquid resistance ahead
    mop_check.py       — per-joint pressure vs MOP every timestep, from the HGL
    hgl.py             — Hydraulic Grade Line: full-route pressure profile at one timestep
                         (gas side behind pig, liquid side ahead); feeds the profile chart,
                         the MOP check, and optimization/debugging
    roadmap.py         — pre-run pressure "corridor": ceiling(x)/floor(x) envelope the pig
                         must stay inside at every milepost, computed once before the sim runs
    backward_pass.py   — theoretical minimum-N2 floor, walking backward from the endpoint
    optimizer.py       — forward booster-spread deployment planner (which station, when to
                         mobilize) to hit a target N2 budget without stalling the pig
    animation.py        — generate_animation(): GIF/MP4 export of the animated profile chart
    formatted_report.py — client-facing xlsx: FILL REPORT table + 2 embedded charts
    purge_report.py     — comprehensive 7-sheet technical xlsx (full run-data record)
    log_export.py       — human-readable + CSV run log export
    config_builder.py   — build_sim_config(ScenarioInputs) -> SimConfig; the one place a scenario
                         becomes an engine config (headless runner, tests and the app all use it)
  app/                  — standalone app (python app.py), standard-library HTTP server
    workspace.py       — open scenario, edits (validated; hard rule 4 enforced on edits), runs,
                         result views, exports. The UI and the assistant act only through it.
    assistant.py       — Claude API tool loop over the Workspace (list/load/read/edit/run/sweep/
                         save). Key from ANTHROPIC_API_KEY or ~/.purge_sim/settings.json.
    server.py          — JSON API + page; 127.0.0.1 only, Host check, per-launch token
    importers.py       — new scenario from ILI / KMZ / TXT / Excel via data/ parsers
    settings.py, paths.py — API key/model; bundled vs user dirs (works frozen by PyInstaller)
    static/index.html  — the whole UI, vanilla JS + inline SVG charts, no CDN
  data/
    ili_parser.py      — Rosen ILI Excel parser (auto-detects pump stations, check valves,
                         BPCV, per-joint MOP from the "Additional description" column)
    profile_parser.py  — KMZ / KMZ+GPSVisualizer-TXT / client Excel / plain milepost-elevation
                         text parsers for non-ILI data sources
    scenario.py        — JSON save/load of full simulation input state (human-readable,
                         inline arrays, hand-editable; does NOT store results)
  ui/
    main_window.py      — Tkinter root: left config panel (30%) / right results panel (70%),
                         persists between runs
    config_panel.py     — all simulation inputs, 8 collapsible sections (data source through
                         gas boosters)
    results_panel.py    — tabbed output: Overview, Pig Speed, Pressure + MOP, N2 Inventory,
                         Boosters, Optimizer — six tabs, matplotlib canvases embedded in Tkinter
    charts.py            — matplotlib chart functions, including plot_pipeline_profile() (the
                         dual-axis animated profile chart — used by animation.py and the xlsx
                         reports, NOT yet wired into a live results_panel tab, see Open Items)
    scenario_manager.py — directory-backed named-scenario collection, thin wrapper on data/scenario.py
tools/                  — one-off per-client build/sweep/verify scripts (not part of the app)
app.py                  — standalone app entry point (opens the browser UI)
main.py                 — Tkinter GUI entry point
run_headless.py          — headless entry point (the one actually used for every real job)
packaging/              — PyInstaller spec + Windows build script for the standalone app
```

---

## Repo Data Policy

- `scenarios/<ClientJob>/*.json` — **tracked in git.** These are the real engineering
  inputs, human-readable and diffable, and each carries real rationale in `meta.notes` /
  `source_files` (why a drive pressure or MOP model was chosen, tender context, etc.).
  Client and pipeline names are fine to keep (public info). **Internal file-system paths
  are NOT fine** — scrub any `D:/...`, `C:\Users\...` etc. down to a bare filename before
  committing. This repo is intended to eventually be pushed to GitHub; nothing with a local
  path should ever be in its history.
- `deliverables/<ClientJob>/<variant>/` — **gitignored.** Client-facing output (xlsx, GIF,
  MP4, zip) — fully regeneratable from the matching scenario JSON. Never overwrite an
  existing variant folder; always create a new, distinctly labeled one (Kevin keeps prior
  variants side-by-side for RFQ comparison).
- `dev/<experiment>/` — **gitignored.** Internal engine-tuning experiments on the original
  reference pipeline (booster siting, coast-tuning, N2-minimization runs) — not client work,
  don't confuse these with `deliverables/`.
- `__pycache__/`, `*.pyc`, `*.pkl` (sim caches), root-level generated GIF/PNG exports,
  `.claude/` — all gitignored, pure build/session artifacts.

---

## Hard Engineering Rules (do not violate without flagging it explicitly)

1. **No N2 venting as a relied-upon mechanism.** `_bleed_gas_to_mop` exists only as a
   last-resort gas-side safety net. Any vent > 0 in a result is a red flag to fix by control,
   not to accept.
2. **Zero slack line is the default target, not a hard invariant.** The engine never silently
   tolerates it: any timestep where achievable drive can't clear `min_liquid_psig` sets
   `slack_line_risk` (flagged per-step in the results chart, the CSV log, and the full xlsx's
   "Slack Line Risk" column) — the run keeps going, it doesn't abort or get hidden. Every
   validated job so far holds this with margin (zero slack steps) at the default 25 psig floor.
   Accepting nonzero slack for a specific job is a real client-facing engineering call, made by
   explicitly lowering `min_liquid_psig` with Kevin's sign-off — never by suppressing or
   reinterpreting the flag itself.
3. **The BPCV only drops the HGL, never lifts it.** It is a backpressure valve, not a pump —
   it sets exit backpressure and can clamp excess pressure, but cannot add head.
4. **Booster suction floor is 200 psi hard / 100 psi willing** — never model a booster
   drawing below that without Kevin explicitly signing off on a different floor for that job.

---

## Open Items (as of 2026-10-05)

1. **Pig-solver drive consistency.** The displayed station pressures/HGL use the corrected
   dynamic-suction/peak-clearing hydraulics, but the pig solver's own exit-pressure logic is
   simpler — so the simulated drive can let the green interface sit below the pump-held HGL
   in the chart. Wiring the corrected hydraulics into the solver itself is the main open
   engine item, but it raises N2 (keeping the column truly full over peaks isn't free) —
   this is a real tension with the lean/coast minimization strategy, not just a bug to fix.
2. **Live "Pipeline Profile" UI tab.** The standalone app (`app.py`) has one: a scrub
   slider over `hgl.compute_hgl()` per step (gas/liquid pressure, elevation, MOP, stations).
   The Tkinter `results_panel.py` still has none.
3. **The real self-supporting gap is not the UI.** Every new client job still requires a
   Claude Code session to classify the input data (which stations are real pumps vs. BPCV
   vs. nothing, speed-capped vs. drive-capped vs. friction-dominated, which strategy to use)
   and hand-build the scenario JSON. None of that judgment lives in the software yet. A
   prettier UI or a future web frontend does not fix this on its own — see the roadmap
   discussion in recent sessions (repo cleanup → doc accuracy → capture that judgment in
   the software itself → only then consider a web frontend). The app's assistant is a first
   step: it can import data, classify and set up a scenario, run and compare variants
   in-app, but the judgment still lives in its prompt and the engineer, not in the engine.

---

## Where to Look for More

- `MEMORY.md` (in Claude's memory directory) indexes the full job history, physics
  decisions, and feedback/preferences — read it, it's kept current across sessions.
- `animation_data.md` / `N2_displacement_full_extraction.md` — raw ground-truth data
  extracted from the original reference animation. Historical reference material, not
  current-state claims; left as-is.
