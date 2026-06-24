"""Render the reference-style animated pipeline profile to a GIF.

Runs a scenario, samples timesteps by pig position, draws plot_pipeline_profile for
each, and assembles an animated GIF (pillow). Also drops a single mid-run PNG for a
quick eyeball. ffmpeg isn't required (GIF via pillow).

Usage:  python make_profile_gif.py
"""
import io
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_headless import build_sim_config
from purge_sim.data.scenario import load_scenario
from purge_sim.engine import simulator as S
from purge_sim.ui.charts import plot_pipeline_profile

BASE = "C:/Users/kevin/PurgeSimScenarios/"
SCENARIO = BASE + "SP to MT_Unlimited Boosters_Unlimited Booster Flow.json"
OUT_GIF = "C:/Users/kevin/Purge-Simulator-3/profile_animation.gif"
OUT_PNG = "C:/Users/kevin/Purge-Simulator-3/profile_frame_test.png"

# Operational narrative keyed to pig milepost (from the reference annotations).
PHASES = [
    (0,   14,  "Begin nitrogen injection at SP"),
    (14,  22,  "Shut off N2 injection — expansion drives the pig (coast)"),
    (22,  50,  "Resume modest SP injection"),
    (50,  70,  "Compressor at NW to avoid higher SP injection pressures/volumes"),
    (70,  90,  "Valve-controlled transfer of upstream N2 — no compressor at SH"),
    (90,  114, "Compressor at LS as interface ascends Tug Mtn"),
    (114, 160, "Over Tug Mtn; boost intermittently through the ups/downs"),
    (160, 174, "Compressor at HW as interface climbs Sutton Mtn"),
    (174, 205, "Descend Sutton Mtn — cease SC backpressure before the pig arrives"),
    (205, 234, "Open MT control valve; pig approaching the trap"),
]


def annotation_for(mp: float) -> str:
    for lo, hi, txt in PHASES:
        if lo <= mp < hi:
            return txt
    return "Pig landing at MT"


def render(step, cfg, roadmap):
    fig = plt.figure(figsize=(12, 7), dpi=96)
    plot_pipeline_profile(fig, step, cfg, roadmap=roadmap,
                          annotation=annotation_for(step.pig_mp))
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CACHE = "C:/Users/kevin/Purge-Simulator-3/.profile_sim_cache.pkl"


def _load_or_run():
    import pickle
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            print("loaded cached sim")
            return pickle.load(f)
    cfg = build_sim_config(load_scenario(SCENARIO).inputs)
    r = S.simulate(cfg)
    with open(CACHE, "wb") as f:
        pickle.dump((cfg, r), f)
    return cfg, r


def main(mp_step=3.0):
    cfg, r = _load_or_run()
    print(f"sim: {len(r.steps)} steps, total N2 {r.total_scf_n2/1e6:.1f}M")

    # one mid-run PNG to eyeball
    mid = min(r.steps, key=lambda s: abs(s.pig_mp - 60))
    fig = plt.figure(figsize=(12, 7), dpi=96)
    plot_pipeline_profile(fig, mid, cfg, roadmap=r.roadmap,
                          annotation=annotation_for(mid.pig_mp))
    fig.savefig(OUT_PNG)
    plt.close(fig)
    print("saved", OUT_PNG)

    # sample steps by pig MP and build the GIF
    targets, mp = [], cfg.purge_start_mp
    while mp <= cfg.purge_end_mp:
        targets.append(mp)
        mp += mp_step
    seen = set()
    frames = []
    for tgt in targets:
        st = min(r.steps, key=lambda s: abs(s.pig_mp - tgt))
        if id(st) in seen:
            continue
        seen.add(id(st))
        frames.append(render(st, cfg, r.roadmap))
    print(f"rendered {len(frames)} frames")
    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:],
                   duration=450, loop=0, optimize=True)
    print("saved", OUT_GIF)


if __name__ == "__main__":
    main()
