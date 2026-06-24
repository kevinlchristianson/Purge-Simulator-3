"""Render the profile with RY & NW pump stations disabled (LS only)."""
import io, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_headless import build_sim_config
from purge_sim.data.scenario import load_scenario
from purge_sim.engine import simulator as S
from purge_sim.ui.charts import plot_pipeline_profile
from make_profile_gif import annotation_for, SCENARIO

cfg = build_sim_config(load_scenario(SCENARIO).inputs)
for ps in cfg.pump_stations:
    if ps.name in ("RY", "NW"):
        ps.enabled = False
print("pumps:", [(p.name, p.enabled) for p in cfg.pump_stations])
r = S.simulate(cfg)
print("sim:", len(r.steps), "steps, N2", round(r.total_scf_n2 / 1e6, 1), "M")

step = min(r.steps, key=lambda s: abs(s.pig_mp - 30))
fig = plt.figure(figsize=(12, 7), dpi=96)
plot_pipeline_profile(fig, step, cfg, roadmap=r.roadmap,
                      annotation="RY & NW shut down — LS is the only running pump")
fig.savefig("C:/Users/kevin/Purge-Simulator-3/profile_noRYNW.png")
plt.close(fig)

frames, mp = [], cfg.purge_start_mp
while mp <= cfg.purge_end_mp:
    st = min(r.steps, key=lambda s: abs(s.pig_mp - mp))
    fig = plt.figure(figsize=(12, 7), dpi=96)
    plot_pipeline_profile(fig, st, cfg, roadmap=r.roadmap, annotation=annotation_for(st.pig_mp))
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    frames.append(Image.open(buf).convert("RGB")); mp += 3
frames[0].save("C:/Users/kevin/Purge-Simulator-3/profile_noRYNW.gif", save_all=True,
               append_images=frames[1:], duration=450, loop=0, optimize=True)
print("saved profile_noRYNW.png and .gif")
