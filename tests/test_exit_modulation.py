"""
The exit (nearest thing downstream of the pig) holds a minimum inlet pressure and, when
modulating, raises it to hold the pig at max speed, up to what it can hold; past that, or
when fixed, the pig overspeeds and the step is flagged. Tests the pig solver, the exit
window, a whole run under each behavior, and the app's validation of the new inputs.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

from purge_sim.app.client_report import build_data, render_html              # noqa: E402
from purge_sim.app.workspace import InputError, Workspace, inputs_summary, results_summary   # noqa: E402
from purge_sim.engine.log_export import export_run_log                        # noqa: E402
from purge_sim.engine.purge_report import purge_report_rows                   # noqa: E402
from purge_sim.engine.physics import liquid_friction_loss_psi, mph_to_fts     # noqa: E402
from purge_sim.engine.pig_solver import PigSolverConfig, solve_pig_speed      # noqa: E402
from purge_sim.engine.pump_stations import (                                  # noqa: E402
    PumpStationConfig, PumpStationState, StationStatus, effective_exit_condition,
    effective_exit_window)

SHORT = "bundled:CHS_TipvilleSantaRita_East10/tipville_east10_3mph.json"
WB08 = "bundled:WB08_Spindle_MP10_butane/wb08_butane_1000psi.json"          # legacy step 450 -> 175 psig
SPMT = "bundled:PMPL_SPtoMT/sp_to_mt_shls_packcoast_48m.json"              # BPCV is the exit, packed column

# A flat 10" diesel line, pig at MP 0, exit 10 miles ahead at 50 psig.
CFG = PigSolverConfig(od_in=10.75, wt_in=0.365, roughness_ft=0.00015, sg=0.84, viscosity_cst=3.0,
                      min_speed_mph=1.0, max_speed_mph=3.0, target_speed_mph=2.0, temperature_f=45.0)
EXIT_MP, EXIT_PSIG = 10.0, 50.0


def _flat(mp):
    return 1000.0


def _resistance(v_mph, exit_psig=EXIT_PSIG, miles=EXIT_MP):
    D_ft = (CFG.od_in - 2 * CFG.wt_in) / 12.0
    return exit_psig + liquid_friction_loss_psi(miles * 5280.0, D_ft, mph_to_fts(v_mph), CFG.sg,
                                                 CFG.viscosity_cst, CFG.roughness_ft)


def _solve(drive, exit_max=None, **kw):
    return solve_pig_speed(CFG, 0.0, 1000.0, EXIT_PSIG, EXIT_MP, "Tankage inlet", drive, _flat,
                           exit_max_psig=exit_max, **kw)


# ---------------------------------------------------------------- pig solver

def test_surplus_drive_is_held_by_the_exit_when_it_has_no_limit():
    drive = _resistance(3.0) + 80.0
    r = _solve(drive)                                   # no limit: the legacy behavior
    assert r.pig_speed_mph == pytest.approx(3.0)
    assert r.endpoint_added_psi == pytest.approx(80.0, abs=0.01)
    assert r.exit_psig == pytest.approx(EXIT_PSIG + 80.0, abs=0.01)
    assert r.exit_min_psig == EXIT_PSIG and r.meter_valve_active and not r.overspeed
    # the pressure balance closes: drive = raised exit + friction at max speed
    assert r.exit_psig + r.liquid_friction_psi == pytest.approx(drive, abs=0.01)


def test_exit_raises_only_what_max_speed_needs():
    r = _solve(_resistance(3.0) + 80.0, exit_max=EXIT_PSIG + 500.0)
    assert r.pig_speed_mph == pytest.approx(3.0)
    assert r.endpoint_added_psi == pytest.approx(80.0, abs=0.01)
    r = _solve(_resistance(2.5))                        # between target and max: nothing added
    assert 2.0 < r.pig_speed_mph < 3.0 and r.endpoint_added_psi == 0.0 and not r.overspeed
    assert r.exit_psig == EXIT_PSIG


def test_exit_at_its_limit_lets_the_pig_overspeed_and_flags_it():
    drive = _resistance(3.0) + 80.0
    r = _solve(drive, exit_max=EXIT_PSIG + 30.0)
    assert r.overspeed and r.pig_speed_mph > 3.0
    assert r.overspeed_mph == pytest.approx(r.pig_speed_mph - 3.0)
    assert r.endpoint_added_psi == pytest.approx(30.0) and r.exit_psig == pytest.approx(EXIT_PSIG + 30.0)
    # the pig runs where the drive balances the column with the exit at its maximum
    assert _resistance(r.pig_speed_mph, EXIT_PSIG + 30.0) == pytest.approx(drive, rel=1e-4)
    # the N2 controller's aim is unchanged by what the exit does
    assert r.drive_for_target_psig == pytest.approx(_solve(drive).drive_for_target_psig)


def test_fixed_exit_adds_nothing():
    drive = _resistance(3.0) + 80.0
    r = _solve(drive, exit_max=EXIT_PSIG)
    assert r.overspeed and r.endpoint_added_psi == 0.0 and r.exit_psig == EXIT_PSIG
    assert _resistance(r.pig_speed_mph) == pytest.approx(drive, rel=1e-4)
    # a maximum below the minimum is the same as fixed
    assert _solve(drive, exit_max=EXIT_PSIG - 20.0).endpoint_added_psi == 0.0


def test_slow_pig_is_untouched():
    r = _solve(_resistance(1.5))
    assert 1.0 < r.pig_speed_mph < 2.0 and r.endpoint_added_psi == 0.0 and r.exit_psig == EXIT_PSIG
    r = _solve(_resistance(0.5))
    assert r.slack_line_risk and r.exit_min_psig == EXIT_PSIG and not r.overspeed


def test_added_pressure_is_measured_against_the_exit_not_a_peak():
    # A hill 4 miles ahead, 300 ft above the pig, binds the drive at max speed (the column
    # must stay full over it); what the exit adds is still the surplus over the exit's own
    # friction + head, so the raised exit pressure is the real pressure at the exit.
    tm, te = np.array([4.0]), np.array([1300.0])
    drive = _resistance(3.0) + 300.0
    r = _solve(drive, exit_max=None, terrain_mp=tm, terrain_elev=te)
    assert r.pig_speed_mph == pytest.approx(3.0)
    assert r.endpoint_added_psi == pytest.approx(300.0, abs=0.01)


# ---------------------------------------------------------------- exit window

def test_exit_window_follows_the_nearest_device():
    st = [PumpStationState(config=PumpStationConfig(mp=20.0, name="A", suction_psig=40.0, max_suction_psig=300.0)),
          PumpStationState(config=PumpStationConfig(mp=60.0, name="B", suction_psig=30.0))]
    lo, hi, mp, desc = effective_exit_window(st, 5.0, 120.0, 80.0, 50.0, 100.0,
                                             bpcv_max_psig=400.0, tankage_max_psig=250.0)
    assert (lo, hi, mp) == (40.0, 300.0, 20.0) and "Station A" in desc
    st[0].status = StationStatus.SHUT_DOWN
    assert effective_exit_window(st, 5.0, 120.0, 80.0, 50.0, 100.0, 400.0, 250.0)[:3] == (30.0, None, 60.0)
    st[1].status = StationStatus.SHUT_DOWN
    assert effective_exit_window(st, 5.0, 120.0, 80.0, 50.0, 100.0, 400.0, 250.0) == (120.0, 400.0, 80.0, "BPCV set point")
    assert effective_exit_window(st, 85.0, 120.0, 80.0, 50.0, 100.0, 400.0, 250.0) == (50.0, 250.0, 100.0, "Tankage inlet")
    assert effective_exit_condition(st, 85.0, 120.0, 80.0, 50.0, 100.0) == (50.0, 100.0, "Tankage inlet")


# ---------------------------------------------------------------- whole run

def test_modulating_run_holds_max_speed_and_reports_what_the_exit_added():
    ws = Workspace()
    ws.load(SHORT)
    assert ws.scenario.inputs.exit_behavior == "modulating"      # old files load as the norm
    res = ws.run()
    s = results_summary(res)
    assert res.completed and s["overspeed_steps"] == 0
    assert s["pig_speed_mph"]["max"] <= res.config.max_speed_mph + 1e-6
    assert s["endpoint_throttling_steps"] > 0 and s["endpoint_added_psi_max"] > 0
    assert s["exit_psig_max"] == pytest.approx(res.config.exit_pressure_run_psig + s["endpoint_added_psi_max"], abs=0.1)
    steps = res.steps
    assert all(st.exit_min_psig == res.config.exit_pressure_run_psig for st in steps)
    assert all(st.exit_psig == pytest.approx(st.exit_min_psig + st.endpoint_added_psi) for st in steps)
    assert all(st.exit_max_psig == 1000.0 for st in steps)        # the MOP at the pig stop
    assert not any("over max speed" in f for f in s["flags"])


def test_fixed_run_overspeeds_and_flags_it():
    ws = Workspace()
    ws.load(SHORT)
    ws.update_inputs({"exit_behavior": "fixed"})
    res = ws.run()
    s = results_summary(res)
    assert res.completed and s["overspeed_steps"] > 0 and s["overspeed_mph_max"] > 0
    assert s["pig_speed_mph"]["max"] > res.config.max_speed_mph
    assert s["endpoint_throttling_steps"] == 0 and s["exit_psig_max"] == res.config.exit_pressure_run_psig
    assert any("over max speed" in f and "fixed" in f for f in s["flags"])
    assert all(st.exit_max_psig == st.exit_min_psig for st in res.steps)


def test_exit_maximum_can_be_typed_and_is_validated():
    ws = Workspace()
    ws.load(SHORT)
    with pytest.raises(InputError, match="exit_behavior"):
        ws.update_inputs({"exit_behavior": "floating"})
    with pytest.raises(InputError, match="can't be below"):
        ws.update_inputs({"exit_max_pressure_psig": 20})
    with pytest.raises(InputError, match="max_suction_psig"):
        ws.update_inputs({"pump_stations": [{"mp": 5.0, "name": "P", "suction_psig": 40, "max_suction_psig": 30}]})
    ws.update_inputs({"exit_max_pressure_psig": 80})          # 30 psi of room, not enough on the descents
    res = ws.run()
    s = results_summary(res)
    assert all(st.exit_max_psig == 80.0 for st in res.steps)
    assert s["endpoint_added_psi_max"] == pytest.approx(30.0, abs=0.1)
    assert s["overspeed_steps"] > 0 and any("80 psig" in f for f in s["flags"])


# ---------------------------------------------------------------- reports, views, pre-run check

def test_reports_carry_the_exit_pressure_and_overspeed(tmp_path):
    ws = Workspace()
    ws.load(SHORT)
    ws.update_inputs({"exit_behavior": "fixed"})
    res = ws.run()
    s = results_summary(res)
    assert "meter_valve_steps" not in s and s["max_speed_mph"] == 3.0 and s["exit_min_psig"] == 50.0
    rows = purge_report_rows(res)
    assert any(r["overspeed"] for r in rows)
    assert all({"exit_psi", "exit_min_psi", "endpoint_added_psi", "overspeed"} <= set(r) for r in rows)
    export_run_log(res, str(tmp_path / "log.txt"), "t")
    txt = (tmp_path / "log.txt").read_text()
    assert "exit_psig,exit_min_psig,endpoint_added_psi," in txt and ",slack_line,overspeed,overspeed_mph," in txt
    assert "meter_valve" not in txt and "fixed at 50.0 psig" in txt and "Over max speed:" in txt
    html = render_html(build_data(res, "t", inputs=ws.scenario.inputs))
    assert "Endpoint Pressure (psi)" in html and '"flag":"over"' in html and '"has_endpoint":true' in html
    openpyxl = pytest.importorskip("openpyxl")
    from purge_sim.engine.formatted_report import export_formatted_report
    from purge_sim.engine.purge_report import export_purge_report
    export_purge_report(res, str(tmp_path / "full.xlsx"), "t")
    export_formatted_report(res, str(tmp_path / "client.xlsx"), "t")
    wb = openpyxl.load_workbook(str(tmp_path / "full.xlsx"))
    hdr = [c.value for c in wb["Condensed Results"][3]]
    assert "Endpoint Added (psi)" in hdr and "Over Max Speed" in hdr
    assert "Meter Valve Active" not in [c.value for c in wb["Raw Data"][3]]
    inputs = {r[0].value: r[1].value for r in wb["Simulation Inputs"].iter_rows(min_row=3) if r[0].value}
    assert inputs["Exit Behavior"].startswith("fixed") and "Throttle Down (miles from end)" not in inputs
    client = openpyxl.load_workbook(str(tmp_path / "client.xlsx"))["Purge Report"]
    assert "Endpoint Pressure (psi)" in [c.value for c in client[17]]


def test_client_table_keeps_its_layout_when_the_exit_never_modulates():
    ws = Workspace()
    ws.load(SHORT)
    ws.update_inputs({"max_speed_mph": 50.0})      # nothing ever pushes the pig that fast
    res = ws.run()
    s = results_summary(res)
    assert s["endpoint_throttling_steps"] == 0 and s["overspeed_steps"] == 0
    data = build_data(res, "t", inputs=ws.scenario.inputs)
    assert not data["purge"]["has_endpoint"]
    assert [c["h"] for c in data["purge"]["cols"]][:8] == ["Miles", "Elevation (ft)", "Elapsed Time (hr)", "Drive Pressure (psi)",
                                                           "Friction Loss (psi)", "Injection Rate (SCFM)", "Cumulative N2 (SCF)",
                                                           "Pig Speed (mph)"]
    assert data["basis"][6][0] == "Endpoint pressure" and "modulating up to the MOP at the exit" in data["basis"][6][1]


def test_legacy_exit_schedule_is_named_on_load_and_nowhere_else():
    ws = Workspace()
    ws.load(WB08)
    note = inputs_summary(ws.scenario)["legacy_exit_schedule"]
    assert note.startswith("legacy schedule step_last_n_miles") and "450 to 175 psig" in note
    ws.load(SHORT)
    assert inputs_summary(ws.scenario)["legacy_exit_schedule"] is None
    ws.load(SPMT)       # carries taper_last_n_miles 50 -> 50: a schedule that changes nothing
    assert inputs_summary(ws.scenario)["legacy_exit_schedule"] is None


def test_precheck_says_what_the_exit_must_hold():
    ws = Workspace()
    ws.load(SHORT)
    h = ws.precheck()["endpoint_hold"]
    assert h["ok"] and not h["fixed"] and h["exit"] == "tank inlet" and h["exit_max_psig"] == 1000
    ws.update_inputs({"exit_behavior": "fixed"})
    h = ws.precheck()["endpoint_hold"]
    assert h["fixed"] and h["room_psi"] == 0
    # SP to MT pack-and-coast: the packed column pushes harder than the BPCV can hold on the descent
    ws.load(SPMT)
    pc = ws.precheck()
    h = pc["endpoint_hold"]
    assert not h["ok"] and h["exit"] == "BPCV" and h["cause"] == "packed N2 column"
    assert h["needed_psi"] > h["room_psi"] and h["overspeed_mph"] > ws.scenario.inputs.max_speed_mph
    assert "overspeed risk" in pc["job_type"] and any("BPCV would have to add" in f for f in pc["findings"])
