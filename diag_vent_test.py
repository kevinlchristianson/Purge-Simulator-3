"""Measure venting / violations across scenarios after the floor-biased booster cap."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_headless import build_sim_config
from purge_sim.data.scenario import load_scenario
from purge_sim.engine import simulator as S

BASE = "C:/Users/kevin/PurgeSimScenarios/"
SCEN = [
    "SP to MT_Unlimited Boosters_Unlimited Booster Flow.json",
    "SP to MT_No Boosters.json",
    "SP to MT_Only LS Pump_2 Spreads_Unlimited Booster Flow.json",
]
for name in SCEN:
    cfg = build_sim_config(load_scenario(BASE + name).inputs)
    r = S.simulate(cfg)
    viol = sum(s.mop_violations for s in r.steps)
    last = r.steps[-1] if r.steps else None
    faces = [s.pig_face_psig for s in r.steps]
    sites = r.booster_plan.sites_mp if r.booster_plan else None
    print(f"{name[:46]:46s} vent={r.total_scf_vented:>12,.0f}  inj={r.total_scf_injected:>12,.0f}  "
          f"viol={viol:>3d}  maxface={max(faces):6.1f}  finalMP={last.pig_mp:6.1f}  "
          f"{'OK' if r.completed else 'ABORT'}")
    if sites is not None:
        print(f"    booster sites: {[round(m,1) for m in sites]}")
