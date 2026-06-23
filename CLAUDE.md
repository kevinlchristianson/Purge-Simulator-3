# Purge Simulator — Project Context for Claude Code

This file was generated from a Claude.ai session that analyzed the reference animation
(`24ML_Nitrogen_Model_20250326_inject_SP_only_Powerpoint_playback_with_notes_1_.mp4`)
alongside the full codebase. Drop this file in the project root — Claude Code reads it
automatically as persistent context.

---

## What This Project Is

**Purge Simulator v30** — a Python/Tkinter desktop app that models a nitrogen displacement
("purge") operation on a 24-inch crude oil pipeline (~240 miles, SP→MT).

Entry point: `main.py` → `purge_sim/ui/main_window.py`

### Architecture
```
purge_sim/
  engine/
    simulator.py      ← main time-step loop, SimConfig, SimStep, SimResults
    physics.py        ← pipe friction, pressure, flow calculations
    pig_solver.py     ← pig velocity at each step
    segment_model.py  ← N2 gas segment tracking ahead of pig
    booster.py        ← nitrogen booster compressor logic
    check_valve.py    ← check valve state machine
    bpcv.py           ← backpressure control valve
    pump_stations.py  ← pump station shutdown logic (PumpStationConfig, PumpStationState)
    backward_pass.py  ← pre-run N2 floor planning
    optimizer.py      ← booster spread deployment planner
    log_export.py     ← run log export
    mop_check.py      ← Maximum Operating Pressure checks
    constants.py
  data/
    scenario.py       ← Scenario save/load (JSON)
    ili_parser.py     ← ILI (in-line inspection) data parser
    profile_parser.py ← elevation/pipe profile parser
  ui/
    main_window.py    ← Tkinter root window, toolbar, menu, run threading
    config_panel.py   ← left panel: all simulation inputs
    results_panel.py  ← right panel: tabbed results (Overview, Pig Speed, Pressure+MOP, N2 Inventory, Boosters, Optimizer)
    charts.py         ← matplotlib chart functions (plot_pig_speed, plot_pressure_profile, plot_n2_inventory, plot_mop_check, plot_booster_usage)
    scenario_manager.py
```

---

## What the Reference Animation Shows

The animation is a **Portland Montreal Pipe Line** conceptual model presentation
(PowerPoint playback). It steps through ~440 seconds showing the pig advancing from
milepost 0 (SP) to ~240 (MT), with the pipeline state updating each frame.

### The Core Chart (the thing we need to replicate)

**Title:** "24-inch Main Line Profile - Nitrogen Displacement of Crude Oil Line-fill"

**Axes:**
- X: Milepost (0–240)
- Left Y: Elevation / Total Head (ft)  — range ~0 to 6,000 ft
- Right Y: Pressure (psig)             — range ~ -2,000 to +1,000 psig

**Data series on the chart:**

| Series | Color | Animated (changes each step)? |
|--------|-------|-------------------------------|
| Ground elevation profile | Dark navy blue | ❌ Static background |
| MOP envelope (hydro/SMYS) | Orange | ❌ Static |
| Max Operating Head (stepped per segment) | Red | ❌ Static |
| **Hydraulic Grade Line (HGL)** | Light blue | ✅ YES — key animation |
| **Pump station operating pressures** (labeled dots) | Blue circles + italic numbers | ✅ YES |
| **N2 interface** (vertical line, pig→injection) | Green vertical | ✅ YES — sweeps right |
| **Pig position** marker | Orange square | ✅ YES — moves right |
| **N2 injection rate** (SCFM÷10) | Green triangle + number at SP | ✅ YES |
| **Flow rate** (SCFM÷10, right side) | Pink/magenta dot | ✅ YES |
| Batch interface diamonds (oil side, ahead of pig) | Brown ◆ | ✅ YES — move right |
| Low alarm markers | Red ✕ | Appear near pig at certain steps |

**Station labels along X axis:** SP, RY, NW, SH, LS, SU, HW, SC, MT

**Example values seen mid-animation (t≈300s, pig at ~MP 75):**
- Injection rate: 800 SCFM (shown as 800 at green triangle)
- Flow rate: 395.7 (×10 = 3957 SCFM, pink dot right side)
- Station pressures: SP=288, RY=288, NW=524, SH=503, LS=370, SU=302, HW=353, SC=560 psig
- HGL drops from ~5000 ft head at NW down to ~4200 ft at LS, reflecting elevation changes

**Annotation callout boxes** appear at key operational phases:
- "Nitrogen Displacement Start-up"
- "Begin nitrogen injection at SP"
- "Operate LS pump"
- "Throttle for steady LS disch. press."
- "Hold steady SC backpressure"
- "Flow rate will be steady"

---

## Gap Analysis: What's Missing in the Code

### 1. The Main Animated Profile Chart — DOES NOT EXIST
`charts.py` has no function for the dual-axis pipeline profile chart. The existing charts are:
- `plot_pig_speed` — speed vs time + speed vs milepost (post-run static)
- `plot_pressure_profile` — pig-face pressure vs milepost (post-run static)
- `plot_n2_inventory`, `plot_mop_check`, `plot_booster_usage`

**None of these show the spatial pipeline state at a single timestep.**

### 2. SimStep Missing: Pump Station Operating Pressures
`SimStep` (in `simulator.py`) currently tracks:
```python
booster_states: List[dict]           # booster compressor states
station_shutdown_events: List[dict]  # shutdown events only
```
It does NOT store **current operating pressure at each pump station per timestep**.
The animation shows live psig labels at every active station updating each frame.

**Need to add to SimStep:**
```python
station_pressures: List[dict] = field(default_factory=list)
# Each dict: {mp, name, discharge_psig, suction_psig, status}
```

### 3. No Hydraulic Grade Line (HGL) Computation for Visualization
`backward_pass.py` and `pig_solver.py` reason about pressures internally, but no code
produces a **spatially-resolved hydraulic grade line** (pressure vs milepost across the
full route at a given timestep) for rendering.

The HGL connects: injection point → each active pump station discharge → pig face → exit.
Between stations it's a sloped line reflecting friction + elevation.

**Need a helper function:**
```python
def compute_hgl(step: SimStep, config: SimConfig) -> List[tuple]:
    """Returns [(mp, pressure_psig), ...] for the full route at this timestep."""
```

### 4. Elevation Profile Not Used in Visualization
`SimConfig.elevation_profile` is a numpy array `[[mp, elev_ft], ...]` used in physics
calculations but **never rendered in any chart**. It's the dark blue background of the
animation.

### 5. Dual Y-Axis + Unit Conversion Not Implemented
The animation plots elevation in **feet of total head** alongside pressure in **psig**
on the same chart with two Y-axes. Converting psig → ft of head:
```python
head_ft = psig * 144 / (fluid_sg * 62.4)
```
`SimConfig.fluid_sg` exists (default 0.85). This conversion is needed to overlay
pressure lines on the elevation-axis chart.

### 6. No Profile Playback UI
`results_panel.py` shows static post-run charts only. The animation is a **step-through
playback** — like scrubbing a timeline. There's no slider or step control in the UI.

### 7. Batch Interface Markers
The brown ◆ diamonds represent oil batch interfaces ahead of the pig. These are tracked
conceptually (via check valves / segment boundaries) but not stored as explicit markers
per step or rendered.

---

## Build Plan (Priority Order)

### Priority 1 — Add `station_pressures` to `SimStep`
**File:** `purge_sim/engine/simulator.py`

Add field to `SimStep`:
```python
station_pressures: List[dict] = field(default_factory=list)
# {mp, name, discharge_psig, status}
```
In the main simulation loop, after `evaluate_shutdowns()`, populate this field with each
station's current estimated operating pressure. The discharge pressure at an active station
can be inferred from the HGL computation (or the backward pass logic already in the code).

### Priority 2 — Add `plot_pipeline_profile()` to `charts.py`
**File:** `purge_sim/ui/charts.py`

New function signature:
```python
def plot_pipeline_profile(fig, step: SimStep, config: SimConfig, elev_profile: np.ndarray):
    """
    Dual-axis pipeline profile chart for a single timestep.
    Left Y: Elevation / Total Head (ft)
    Right Y: Pressure (psig)
    X: Milepost

    Renders:
    - Ground elevation (dark blue, left axis)
    - MOP envelope (orange, left axis, converted from psig)
    - Operating Head gradient/HGL (light blue, left axis)
    - Pump station pressure dots + psig labels (blue circles)
    - N2 interface vertical line (green)
    - Pig position marker (orange square)
    - Injection rate label (green, at injection MP)
    - Flow rate label (pink, right edge)
    - Batch interface markers if available (brown diamonds)
    """
```

### Priority 3 — Add "Profile" Tab with Playback Slider
**File:** `purge_sim/ui/results_panel.py`

Add a new tab "Pipeline Profile" with:
- A matplotlib figure showing `plot_pipeline_profile()` for the current step
- A `ttk.Scale` slider (0 → len(steps)-1) to scrub through timesteps
- Previous/Next step buttons
- Display of current time (t_hr) and pig milepost
- Auto-play button (steps through at ~10-30 fps using `root.after()`)

### Priority 4 — HGL Helper
**File:** `purge_sim/engine/simulator.py` or new `purge_sim/engine/hgl.py`

```python
def compute_hgl(step: SimStep, config: SimConfig) -> List[tuple]:
    """
    Returns [(mp, pressure_psig), ...] spatial HGL across full route.
    Uses gas_pressure_profile for N2 side (pig→injection).
    Uses station_pressures + friction model for oil side (pig→exit).
    """
```

---

## Key Physical Context for Claude Code

- **Pipeline:** 24-inch OD, ~240 miles, SP (Portland ME) → MT (Montreal QC)
- **Fluid:** Crude oil, SG ~0.85, viscosity ~2.7 cSt
- **Pig:** Displacement pig separating N2 (behind) from crude oil (ahead)
- **N2 injection:** At SP only (this model scenario: "inject SP only")
- **Pump stations:** SP, RY, NW, SH, LS, SU, HW, SC, MT — stations shut down as pig arrives
- **Booster compressors:** N2 booster spreads recompress gas behind the pig at intermediate points
- **MOP:** Maximum Operating Pressure varies by pipe segment (set by hydrostatic test and SMYS)
- **HGL:** Hydraulic Grade Line = elevation + pressure head — slopes downward from each
  pump station to the next due to friction losses and elevation changes
- **Total Head** = elevation_ft + (pressure_psig × 144) / (SG × 62.4)

---

## Notes on v29 vs v30

`Purge_Modeling_Program_v29.py` (~221KB) is the monolithic predecessor. The current
refactored structure in `purge_sim/` is v30. When in doubt about business logic or
original behavior, the v29 file is the reference. It likely contains an earlier version
of the profile chart that can be used as a starting point.

---

*Generated by Claude.ai — June 2026*
*Source: analysis of MP4 animation + full codebase review*
