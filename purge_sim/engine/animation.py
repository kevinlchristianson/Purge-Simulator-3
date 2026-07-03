"""
animation.py — Animated pipeline profile GIF / MP4 generator.

Entry point: generate_animation(results, cfg, out_gif, ...)

Called by:
  run_headless.py   — automatic deliverable after every headless run
  make_profile_gif.py — standalone CLI wrapper
"""
from __future__ import annotations

import io
import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

try:
    from PIL import Image as _PILImage
    _PIL = True
except ImportError:
    _PIL = False


# ---------------------------------------------------------------------------
# Phase annotation auto-detection
# ---------------------------------------------------------------------------

def _auto_phases(results, cfg) -> List[Tuple[float, float, str]]:
    """Derive operational phase labels from injection, booster, and station events."""
    start, end = cfg.purge_start_mp, cfg.purge_end_mp
    events: List[Tuple[float, str]] = []

    # Injection on / off transitions
    prev_inj = 0.0
    for step in results.steps:
        cur = step.injection_scfm
        if prev_inj < 1.0 <= cur:
            events.append((step.pig_mp, "N2 injection active"))
        elif prev_inj >= 1.0 > cur:
            events.append((step.pig_mp, "Gas expansion drives pig (injection off)"))
        prev_inj = cur

    # First activation per booster
    seen_b: set = set()
    for step in results.steps:
        for bs in (step.booster_states or []):
            k = bs.get("name") or bs.get("mp")
            if bs.get("running") and k not in seen_b:
                seen_b.add(k)
                name = bs.get("name") or f"MP {bs.get('mp', '')}"
                events.append((step.pig_mp, f"Booster active: {name}"))

    # Pump station shutdowns
    seen_s: set = set()
    for step in results.steps:
        for ev in (step.station_shutdown_events or []):
            k = ev.get("name") or ev.get("mp")
            if k not in seen_s:
                seen_s.add(k)
                events.append((step.pig_mp,
                               f"Station {ev.get('name', '')} shutdown"))

    if not events:
        return [(start, end, "N2 displacing fluid")]

    events.sort(key=lambda e: e[0])

    phases: List[Tuple[float, float, str]] = []
    if events[0][0] > start + 0.5:
        phases.append((start, events[0][0], "Purge startup"))
    for i, (mp, lbl) in enumerate(events):
        nxt = events[i + 1][0] if i + 1 < len(events) else end
        phases.append((mp, nxt, lbl))

    return phases


def _label_for_mp(mp: float, phases: List[Tuple[float, float, str]]) -> str:
    for lo, hi, txt in phases:
        if lo <= mp < hi:
            return txt
    return ""


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_frame(step, cfg, roadmap, phases, figsize, dpi):
    """Render one frame and return a PIL RGB Image."""
    from ..ui.charts import plot_pipeline_profile
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plot_pipeline_profile(fig, step, cfg, roadmap=roadmap,
                          annotation=_label_for_mp(step.pig_mp, phases))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return _PILImage.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_animation(
    results,
    cfg,
    out_gif: str,
    out_mp4: Optional[str] = None,
    phases: Optional[List[Tuple[float, float, str]]] = None,
    mp_step: float = 3.0,
    figsize: Tuple[float, float] = (12, 7),
    dpi: int = 96,
    frame_ms: int = 450,
    progress_cb=None,
) -> None:
    """
    Render the animated pipeline profile to GIF (and optionally MP4).

    Args:
        results:     SimResults from simulate()
        cfg:         SimConfig used for the run
        out_gif:     Output .gif path (always produced)
        out_mp4:     Optional .mp4 path; skipped silently if no encoder is available
        phases:      [(start_mp, end_mp, annotation_text), ...]; auto-detected if None
        mp_step:     Miles between animation frames (smaller = more frames, slower)
        figsize:     matplotlib figure size in inches (width, height)
        dpi:         Render resolution
        frame_ms:    Display time per GIF frame in milliseconds
        progress_cb: Optional callable(frames_done: int, total_frames: int)
    """
    if not _MPL:
        raise ImportError("matplotlib required:  pip install matplotlib")
    if not _PIL:
        raise ImportError("Pillow required:  pip install Pillow")
    if not results.steps:
        raise ValueError("No simulation steps to animate")

    if phases is None:
        phases = _auto_phases(results, cfg)

    # Sample steps at evenly-spaced pig-milepost targets
    tgts = list(np.arange(cfg.purge_start_mp, cfg.purge_end_mp + mp_step, mp_step))
    if not tgts or abs(tgts[-1] - cfg.purge_end_mp) > 0.01:
        tgts.append(cfg.purge_end_mp)

    seen: set = set()
    sample: list = []
    for t in tgts:
        st = min(results.steps, key=lambda s, _t=t: abs(s.pig_mp - _t))
        if id(st) not in seen:
            seen.add(id(st))
            sample.append(st)

    total = len(sample)
    frames = []
    for i, step in enumerate(sample):
        frames.append(_render_frame(step, cfg, results.roadmap, phases, figsize, dpi))
        if progress_cb:
            progress_cb(i + 1, total)

    # GIF (always)
    gif_dir = os.path.dirname(os.path.abspath(out_gif))
    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=True)

    # MP4 (optional — try imageio, then matplotlib/ffmpeg, skip on failure)
    if out_mp4:
        _write_mp4(frames, out_mp4, fps=max(1, round(1000 / frame_ms)))


def _write_mp4(frames: list, path: str, fps: int = 2) -> None:
    mp4_dir = os.path.dirname(os.path.abspath(path))
    if mp4_dir:
        os.makedirs(mp4_dir, exist_ok=True)

    # Method 1: imageio (preferred — just needs imageio installed)
    try:
        import imageio
        imageio.mimwrite(path, [np.array(f) for f in frames], fps=fps)
        return
    except Exception as e:
        print(f"  MP4 (imageio): {e} — trying ffmpeg writer…")

    # Method 2: matplotlib FuncAnimation + ffmpeg writer
    try:
        import matplotlib.animation as manim
        arr0 = np.array(frames[0])
        h, w = arr0.shape[:2]
        fig2, ax2 = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi
                                  if (_MPL and (dpi := 96)) else 96)
        ax2.axis("off")
        fig2.subplots_adjust(0, 0, 1, 1)
        im = ax2.imshow(arr0)

        def _upd(i, _frames=frames):
            im.set_array(np.array(_frames[i]))
            return [im]

        ani = manim.FuncAnimation(fig2, _upd, frames=len(frames),
                                   interval=500, blit=True)
        ani.save(path, writer="ffmpeg", fps=fps, dpi=96)
        plt.close(fig2)
    except Exception as e2:
        print(f"  MP4 skipped (no encoder available): {e2}")
