"""
Scenario save/load — JSON serialization of simulation inputs.

Scenarios capture the full input state so a run is 100% reproducible.
They do NOT store simulation results (results are large — recompute on load).

The format is human-readable JSON. Arrays are stored inline (not base64)
for easy inspection and hand-editing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Scenario data model (flat dict of primitives + lists)
# ---------------------------------------------------------------------------

@dataclass
class ScenarioMeta:
    name: str
    created_at: str = ""
    modified_at: str = ""
    notes: str = ""
    source_files: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec='seconds')
        if not self.modified_at:
            self.modified_at = self.created_at


@dataclass
class ScenarioInputs:
    """
    All user-configurable simulation inputs in a flat, JSON-serializable structure.
    Lists of infra items are stored as lists of dicts.
    Elevation profile stored as list of [mp, elev_ft] pairs.
    """
    # Pipeline geometry
    purge_start_mp: float = 0.0
    purge_end_mp: float = 236.0
    pipe_segments: List[dict] = field(default_factory=list)
    # e.g. [{'start_mp': 0, 'end_mp': 103.28, 'od_in': 24.0, 'wt_in': 0.313}, ...]

    # Elevation profile
    elevation_profile: List[list] = field(default_factory=list)
    # e.g. [[0.0, 52.0], [0.5, 48.0], ...]

    # Fluid
    fluid_name: str = "Diesel"
    fluid_sg: float = 0.85
    fluid_viscosity_cst: float = 2.7
    fluid_roughness_ft: float = 0.00015

    # N2 gas
    n2_temperature_f: float = 45.0
    max_injection_psig: Optional[float] = None
    max_injection_scfm: float = 20_000.0

    # Pack-and-coast strategy: inject this many SCF holding drive near MOP, then cut SP and
    # coast on PV. None = lean floor-defending controller.
    n2_budget_scf: Optional[float] = None
    drive_mop_fraction: float = 0.9

    # Exit / endpoint
    exit_pressure_run_psig: float = 50.0
    exit_pressure_end_psig: float = 10.0
    exit_pressure_behavior: str = 'taper_last_n_miles'
    throttle_down_miles: float = 5.0

    # Pig speed
    min_speed_mph: float = 0.5
    max_speed_mph: float = 6.0
    target_speed_mph: float = 2.0

    # MAOP / drive envelope
    maop_psig: Optional[float] = None
    max_drive_psig: Optional[float] = None

    # Infrastructure (manually entered or auto-detected from ILI)
    check_valves: List[dict] = field(default_factory=list)
    # e.g. [{'mp': 6.86, 'name': 'CV-6.86', 'is_pump_station': False}, ...]

    pump_stations: List[dict] = field(default_factory=list)
    # e.g. [{'mp': 26.45, 'name': 'Station A', 'suction_psig': 30.0}, ...]

    booster_stations: List[dict] = field(default_factory=list)
    # e.g. [{'mp': 26.45, 'name': 'RY'}, {'mp': 51.62, 'name': 'NW'}, ...]
    # Location only — pressure/flow specs come from spread settings below.

    deployed_booster_mps: Optional[List[float]] = None
    # If set, deploy EXACTLY these booster MPs (snapped to nearest booster_stations entry)
    # instead of the auto-optimizer. None = let the optimizer choose. Lets a scenario pin a
    # specific booster set (e.g. NW+LS only, HW off).

    n_spreads: int = 2
    mob_time_hr: float = 18.0            # hours per move (travel + rig-up)
    spread_discharge_psig: float = 1200.0
    spread_suction_min_psig: float = 150.0
    spread_max_flow_scfm: float = 12_000.0

    bpcv: Optional[dict] = None
    # e.g. {'mp': 205.64, 'elevation_ft': 137.0, 'name': 'St Cesaire BPCV'}

    # MOP joints from ILI data — saved here so headless runs and reloaded scenarios
    # have full per-joint pressure limits without needing the original ILI file.
    # Each dict: {mp, mop_psig, elevation_ft, od_in, wt_in}
    mop_joints: List[dict] = field(default_factory=list)

    # Data source reference
    data_source: str = ""   # 'ILI', 'KMZ', 'TXT', 'Excel', 'manual'
    data_file: str = ""

    # Simulation control
    dt_hr: float = 1.0 / 60.0
    adaptive_dt: bool = True
    mop_warning_fraction: float = 0.95


@dataclass
class Scenario:
    meta: ScenarioMeta
    inputs: ScenarioInputs

    def to_dict(self) -> dict:
        return {
            'meta': asdict(self.meta),
            'inputs': asdict(self.inputs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Scenario':
        meta = ScenarioMeta(**d.get('meta', {}))
        # Strip keys no longer in ScenarioInputs so old saved scenarios load cleanly
        import dataclasses
        valid = {f.name for f in dataclasses.fields(ScenarioInputs)}
        inputs_raw = {k: v for k, v in d.get('inputs', {}).items() if k in valid}
        inputs = ScenarioInputs(**inputs_raw)
        return cls(meta=meta, inputs=inputs)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_scenario(scenario: Scenario, file_path: str) -> None:
    """Save scenario to a JSON file."""
    scenario.meta.modified_at = datetime.now().isoformat(timespec='seconds')
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(scenario.to_dict(), f, indent=2, default=_json_default)


def load_scenario(file_path: str) -> Scenario:
    """Load a scenario from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return Scenario.from_dict(d)


def list_scenarios(directory: str) -> List[dict]:
    """Return a list of scenario summaries from a directory (name, path, modified_at)."""
    out = []
    if not os.path.isdir(directory):
        return out
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(directory, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            meta = d.get('meta', {})
            out.append({
                'name':        meta.get('name', fname),
                'path':        path,
                'modified_at': meta.get('modified_at', ''),
                'notes':       meta.get('notes', ''),
            })
        except Exception:
            out.append({'name': fname, 'path': path, 'modified_at': '', 'notes': ''})
    return out


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for numpy types."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
