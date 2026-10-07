"""
Tests for the job intake (purge_sim/app/intake.py), its lookups (pipe_catalog, fluids)
and the pre-run check (engine/precheck.py). No network, no simulation runs.
"""
import dataclasses
import glob
import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from purge_sim.app import intake                                   # noqa: E402
from purge_sim.app.assistant import execute_tool                   # noqa: E402
from purge_sim.app.workspace import InputError, Workspace          # noqa: E402
from purge_sim.data import fluids, pipe_catalog as pipes          # noqa: E402
from purge_sim.data.scenario import Scenario, ScenarioInputs, ScenarioMeta, load_scenario  # noqa: E402
from purge_sim.engine.precheck import exit_pressure_at_stop, precheck  # noqa: E402

SCEN = os.path.join(REPO_ROOT, "scenarios")

# Kevin's example: "I'm purging this diesel line from MP 0 to MP 35 to tankage at the
# endpoint. It's 6 inch. Drive pressure capped at 500 psi, MOP 900 psi at all points."
EXAMPLE = {"fluid": "diesel", "purge_end_mp": 35, "exit_type": "tankage", "nps": "6",
           "max_drive_psig": 500, "mop_basis": "flat", "mop_psig": 900}


def _fresh_kmz_scenario(length_mi=36.0):
    mp = np.linspace(0, length_mi, 400)
    el = 3000 + 200 * np.sin(mp / 4) + 5 * mp
    inp = ScenarioInputs(purge_start_mp=0.0, purge_end_mp=length_mi,
                         elevation_profile=[[float(a), float(b)] for a, b in zip(mp, el)],
                         route_latlon=[[45.0 + i * 1e-3, -108.0] for i in range(20)],
                         pipe_segments=[{"start_mp": 0.0, "end_mp": length_mi, "od_in": 24.0, "wt_in": 0.313}],
                         data_source="KMZ+USGS")
    return Scenario(meta=ScenarioMeta(name="kmz job", intake={"fresh_import": True}), inputs=inp)


def _ws(sc):
    ws = Workspace()
    ws.scenario = sc
    return ws


# ---------------------------------------------------------------- lookups

def test_pipe_catalog():
    assert pipes.od_in("6") == 6.625 and pipes.od_in('6"') == 6.625 and pipes.od_in("NPS 8") == 8.625
    assert pipes.standard_wt_in("6") == 0.280 and pipes.standard_wt_in("16") == 0.375
    assert pipes.standard_wt_in("8", "XS") == 0.500
    # Poplar-Richey: 10.75 x 0.219 needs at least X52 for 1440 psig; X42 is ~1230
    assert pipes.min_grade_for(10.75, 0.219, 1440) == "X52"
    assert round(pipes.barlow_psig(10.75, 0.219, "X42")) == 1232
    chk = pipes.barlow_check(10.75, 0.219, 1440, "X42")
    assert chk["ok"] is False and chk["min_grade"] == "X52"
    assert pipes.FLANGE_CLASS_PSIG["600"] == 1440


def test_fluid_library():
    assert fluids.get("ngl_y1").min_liquid_psig == 385      # PH-30: Y1 held at 385 psig
    assert fluids.get("diesel").min_liquid_psig is None
    assert abs(fluids.api_to_sg(40) - 0.825) < 1e-3
    assert fluids.match("Light Sweet Crude (assumed API 40)") == "light_crude"
    assert fluids.match("Y1 NGL (de-ethanized / EP mix)") == "ngl_y1"
    assert fluids.match("Powder River Basin Light Sweet Crude (assumed API 40)") == "powder_river"
    assert fluids.match("WCS dilbit") == "wcs"
    assert fluids.match("Crude Oil (API 39)") is None
    assert all(fluids.get(k).min_liquid_psig is None
               for k in ("bakken", "permian", "wcs", "synthetic_crude", "msw", "powder_river"))


# ---------------------------------------------------------------- intake

def test_fresh_import_asks_for_pipe_product_and_mop():
    v = intake.view(_fresh_kmz_scenario())
    assert set(v["missing"]) == {"nps", "fluid", "mop_psig"}


def test_example_request_builds_the_scenario():
    ws = _ws(_fresh_kmz_scenario())
    r = ws.apply_intake(EXAMPLE)
    inp = ws.scenario.inputs
    assert r["missing"] == []
    assert inp.purge_end_mp == 35 and inp.fluid_name == "Diesel" and inp.fluid_sg == 0.838
    assert inp.pipe_segments == [{"start_mp": 0.0, "end_mp": 36.0, "od_in": 6.625, "wt_in": 0.280}]
    assert inp.maop_psig == 900 and len(inp.mop_joints) == 400
    assert {j["mop_psig"] for j in inp.mop_joints} == {900.0}
    assert inp.max_drive_psig == 500 and inp.max_injection_psig == 500
    assert inp.exit_pressure_run_psig == 50 and inp.exit_pressure_behavior == "constant_run"
    assert inp.n2_budget_scf is None
    # what wasn't said is reported, not hidden
    joined = " ".join(r["assumptions"])
    for word in ("Wall thickness", "Target pig speed", "Delivery pressure", "Specific gravity"):
        assert word in joined
    assert "drive-capped" in r["precheck"]["job_type"]
    notes = ws.scenario.meta.notes
    assert notes.count(intake.NOTES_BEGIN) == 1 and "ASSUMED - CONFIRM" in notes
    # a correction replaces the notes block and drops that assumption
    r = ws.apply_intake({"wt_in": 0.188})
    assert ws.scenario.inputs.pipe_segments[0]["wt_in"] == 0.188
    assert "Wall thickness" not in " ".join(r["assumptions"])
    assert ws.scenario.meta.notes.count(intake.NOTES_BEGIN) == 1
    assert ws.scenario.meta.intake["answers"]["wt_in"] == 0.188
    # an answer cleared back to blank uses its default again
    ws.apply_intake({"wt_in": None})
    assert ws.scenario.inputs.pipe_segments[0]["wt_in"] == 0.280


def test_pack_and_coast_is_sized_from_the_precheck():
    ws = _ws(_fresh_kmz_scenario())
    ws.apply_intake(EXAMPLE)
    r = ws.apply_intake({"strategy": "pack_and_coast"})
    inp = ws.scenario.inputs
    assert inp.n2_budget_scf == r["precheck"]["n2_min_coast_budget_scf"]
    assert inp.drive_mop_fraction == pytest.approx(r["precheck"]["static_pack_limit_psig"] / 900, abs=1e-3)
    assert any(a.startswith("N2 budget") for a in r["assumptions"])
    ws.apply_intake({"n2_budget_scf": 1_000_000, "pack_pressure_psig": 450})
    assert ws.scenario.inputs.n2_budget_scf == 1_000_000
    assert ws.scenario.inputs.drive_mop_fraction == 0.5


def test_reverse_route_moves_every_milepost():
    ws = _ws(_fresh_kmz_scenario())
    ws.apply_intake(EXAMPLE)
    before = ws.scenario.inputs
    ws.apply_intake({"reverse_route": True})
    after = ws.scenario.inputs
    assert after.elevation_profile[0] == [0.0, before.elevation_profile[-1][1]]
    assert after.route_latlon == before.route_latlon[::-1]
    # the pig still covers the same physical stretch (old MP 0-35 = new MP 1-36)
    assert (after.purge_start_mp, after.purge_end_mp) == (1.0, 36.0)
    assert after.mop_joints[0]["mp"] == 0.0 and after.mop_joints[0]["elevation_ft"] == before.elevation_profile[-1][1]


def test_liquid_exit_past_the_pig_stop():
    ws = _ws(_fresh_kmz_scenario())
    ws.apply_intake({**EXAMPLE, "purge_end_mp": 20, "fluid_exit_mp": 36})
    inp = ws.scenario.inputs
    assert inp.purge_end_mp == 20
    assert inp.exit_pressure_run_psig > 50 + 16 * 0.5    # 16 mi of 6" diesel friction at least
    with pytest.raises(InputError, match="before the pig stop"):
        ws.apply_intake({"fluid_exit_mp": 10})


def test_laurel_exit_beyond_pig_stop_is_far_above_delivery():
    """Laurel's pig stops at the Allendale BV (MP 4.489) but the diesel goes on to the
    Billings tank farm (MP 33.329) over the rimrocks; the pig stop sees far more than
    the 50 psig delivery pressure."""
    sc = load_scenario(os.path.join(SCEN, "CHS_LaurelAllendale", "laurel_allendale_3mph.json"))
    p = exit_pressure_at_stop(sc.inputs, 4.489, 33.329, 50.0)
    assert 300 < p < 500


def test_volatile_product_defaults_above_vapor_pressure():
    ws = _ws(_fresh_kmz_scenario())
    r = ws.apply_intake({**EXAMPLE, "fluid": "ngl_y1", "mop_psig": 1440, "max_drive_psig": 700})
    assert ws.scenario.inputs.exit_pressure_run_psig == 385
    assert r["precheck"]["vapor_check"]["min_liquid_psig"] == 385


def test_bad_answers_are_rejected():
    ws = _ws(_fresh_kmz_scenario())
    with pytest.raises(InputError, match="unknown question"):
        ws.apply_intake({"colour": "blue"})
    with pytest.raises(InputError, match="standard pipe size"):
        ws.apply_intake({"nps": "7"})
    with pytest.raises(InputError, match="must be one of"):
        ws.apply_intake({"fluid": "molasses"})
    with pytest.raises(InputError, match="speeds must satisfy"):
        ws.apply_intake({**EXAMPLE, "target_speed_mph": 3, "max_speed_mph": 2})
    assert ws.scenario.meta.intake == {"fresh_import": True}    # nothing half-applied


def test_saved_scenarios_round_trip_unchanged():
    """Applying no answers to any bundled job leaves its inputs exactly as they are."""
    n = 0
    for path in sorted(glob.glob(os.path.join(SCEN, "*", "*.json"))):
        with open(path) as f:
            raw = json.load(f)
        if not (isinstance(raw, dict) and "inputs" in raw):
            continue
        sc = load_scenario(path)
        assert intake.view(sc)["missing"] == [], path
        new, _, _ = intake.apply(sc, {})
        assert dataclasses.asdict(new) == dataclasses.asdict(sc.inputs), path
        n += 1
    assert n >= 20


# ---------------------------------------------------------------- pre-run check

def test_precheck_matches_past_job_reasoning():
    # Thunderbird: 600 psig pack chosen because the column's head overpressured joints
    # at 675; the at-rest and moving limits bracket that choice.
    tb = precheck(load_scenario(os.path.join(SCEN, "Bridger_THUNDERBIRD", "thunderbird_packcoast.json")).inputs)
    assert tb["static_pack_limit_psig"] < 600 < tb["moving_pack_limit_psig"] < 675
    assert "drive-capped" in tb["job_type"] and "friction-dominated" in tb["job_type"]
    # Tipville East 10: heavy crude, laminar at run speed
    tv = precheck(load_scenario(os.path.join(SCEN, "CHS_TipvilleSantaRita_East10", "tipville_east10_3mph.json")).inputs)
    assert tv["laminar"] and "speed-capped" in tv["job_type"]
    # Poplar-Richey: minimum coast budget near the 1.3 MMscf the job used
    pr = precheck(load_scenario(os.path.join(SCEN, "Bridger_PoplarRichey", "poplar_richey_packcoast.json")).inputs)
    assert 1.0e6 < pr["n2_min_coast_budget_scf"] < 1.5e6 and pr["pack_and_coast_possible"]
    # SP->MT: pump stations and a BPCV set the exit, so no flat pack limit
    sm = precheck(load_scenario(os.path.join(SCEN, "PMPL_SPtoMT", "sp_to_mt_shls_packcoast_48m.json")).inputs)
    assert "pump/BPCV controlled" in sm["job_type"] and sm["static_pack_limit_psig"] is None


# ---------------------------------------------------------------- assistant tools

def test_assistant_setup_tools():
    ws = _ws(_fresh_kmz_scenario())
    v = execute_tool(ws, "get_job_setup", {})
    assert set(v["missing"]) == {"nps", "fluid", "mop_psig"}
    assert next(q for q in v["questions"] if q["id"] == "fluid")["options"][0] == "diesel"
    r = execute_tool(ws, "set_job_setup", {"answers": EXAMPLE, "reason": "user described the job"})
    assert r["missing"] == [] and ws.scenario.inputs.maop_psig == 900
    assert "job_type" in execute_tool(ws, "precheck_job", {})
    p = execute_tool(ws, "pipe_lookup", {"nps": "10", "wt_in": 0.219, "mop_psig": 1440})
    assert p["min_grade_for_mop"] == "X52" and p["od_in"] == 10.75
    with pytest.raises(InputError):
        execute_tool(ws, "pipe_lookup", {"nps": "7"})
