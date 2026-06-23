"""Build and inspect the pre-run pressure roadmap for each scenario."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_headless import build_sim_config
from purge_sim.data.scenario import load_scenario
from purge_sim.engine.mop_check import thin_mop_joints
from purge_sim.engine.roadmap import build_roadmap
from scipy.interpolate import interp1d

SCEN = [
    "SP to MT_Unlimited Boosters_Unlimited Booster Flow.json",
    "SP to MT_No Boosters.json",
    "SP to MT_Only LS Pump_2 Spreads_Unlimited Booster Flow.json",
]
BASE = "C:/Users/kevin/PurgeSimScenarios/"

for name in SCEN:
    cfg = build_sim_config(load_scenario(BASE + name).inputs)
    joints = thin_mop_joints(cfg.mop_joints) if cfg.mop_joints else []
    mjmp  = np.array([j.mp for j in joints]); order = np.argsort(mjmp)
    mjmp  = mjmp[order]
    mjmop = np.array([j.mop_psig for j in joints])[order]
    mjel  = np.array([j.elevation_ft for j in joints])[order]
    ea_arr = np.asarray(cfg.elevation_profile, float)
    ip = interp1d(ea_arr[:,0], ea_arr[:,1], kind='linear', bounds_error=False,
                  fill_value=(ea_arr[0,1], ea_arr[-1,1]))
    rm = build_roadmap(mjmp, mjmop, mjel, cfg, lambda mp: float(ip(mp)))
    print("=" * 78)
    print(name)
    print(rm.report())
    # show corridor at the venting hot-spots
    print("  corridor samples (mp: floor -> ceiling [gas/liq]):")
    for x in (40, 70, 80, 105, 120, 160, 174, 200):
        c = float(np.interp(x, rm.mp, rm.ceiling_psig))
        f = float(np.interp(x, rm.mp, rm.floor_psig))
        g = float(np.interp(x, rm.mp, rm.ceiling_gas_psig))
        l = float(np.interp(x, rm.mp, rm.ceiling_liq_psig))
        flag = "  <-- INFEASIBLE" if f > c else ""
        print(f"    MP {x:3d}: {f:6.0f} -> {c:6.0f}   [gas {g:6.0f} / liq {l:6.0f}]{flag}")
