"""Diagnostic: where does the MOP-relief venting come from?

Monkeypatches simulator._bleed_gas_to_mop to record, per vent event, the
segment's mp span, its pressure, the binding ceiling, and SCF removed — then
aggregates by 20-mile bucket and by cause.
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import purge_sim.engine.simulator as S
from purge_sim.engine.segment_model import SegmentList
from run_headless import build_sim_config
from purge_sim.data.scenario import load_scenario

events = []  # (mp_lo, mp_hi, pressure, ceiling, vented, is_last)

_orig = S._bleed_gas_to_mop

def _patched(segs, mj_mp, mj_mop, mop_cap, n2_temperature_f):
    if mj_mp is None:
        return 0.0
    vented = 0.0
    n = len(segs)
    for i in range(n):
        seg = segs[i]
        lo = int(np.searchsorted(mj_mp, seg.upstream_mp,   side='left'))
        hi = int(np.searchsorted(mj_mp, seg.downstream_mp, side='right'))
        ceiling = float(np.min(mj_mop[lo:hi])) if hi > lo else math.inf
        is_last = (i == n - 1)
        if is_last and math.isfinite(mop_cap) and mop_cap > 0:
            ceiling = min(ceiling, mop_cap)
        if math.isfinite(ceiling) and seg.pressure_psig > ceiling:
            target_scf = seg.scf_at_pressure(max(0.0, ceiling))
            excess = seg.scf - target_scf
            if excess > 0:
                P = seg.pressure_psig
                removed = seg.remove_scf(excess)
                vented += removed
                events.append((seg.upstream_mp, seg.downstream_mp, P, ceiling, removed, is_last))
    return vented

S._bleed_gas_to_mop = _patched

path = sys.argv[1] if len(sys.argv) > 1 else \
    "C:/Users/kevin/PurgeSimScenarios/SP to MT_Unlimited Boosters_Unlimited Booster Flow.json"
scen = load_scenario(path)
cfg = build_sim_config(scen.inputs)
res = S.simulate(cfg)

total = sum(e[4] for e in events)
last_v = sum(e[4] for e in events if e[5])
intr_v = total - last_v
print(f"\nTotal vented: {total:,.0f} SCF  over {len(events)} vent events")
print(f"  pig-adjacent (last) segment: {last_v:,.0f} SCF ({100*last_v/max(1,total):.0f}%)")
print(f"  interior segments:           {intr_v:,.0f} SCF ({100*intr_v/max(1,total):.0f}%)")

print("\nVenting by segment upstream-mp bucket (20 mi):")
buckets = {}
for lo, hi, P, ceil, v, isl in events:
    b = int(lo // 20) * 20
    buckets.setdefault(b, [0.0, 0, 0.0])
    buckets[b][0] += v
    buckets[b][1] += 1
    buckets[b][2] += (P - ceil) * v  # weighted overpressure
for b in sorted(buckets):
    v, cnt, wop = buckets[b]
    print(f"  MP {b:3d}-{b+20:3d}: {v:12,.0f} SCF  ({cnt:5d} events)  avg over-MOP {wop/max(1,v):5.1f} psi")

print("\nTop 12 single vent events by SCF:")
for e in sorted(events, key=lambda x: -x[4])[:12]:
    lo, hi, P, ceil, v, isl = e
    tag = "LAST" if isl else "intr"
    print(f"  seg [{lo:6.1f},{hi:6.1f}] P={P:6.1f} ceil={ceil:6.1f} over={P-ceil:6.1f}  vent={v:11,.0f}  {tag}")
