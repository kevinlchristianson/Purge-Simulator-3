"""
animation.py — Animated pipeline profile GIF / MP4 generator.

Entry point: generate_animation(results, cfg, out_gif, ...)

Called by:
  run_headless.py   — automatic deliverable after every headless run
  make_profile_gif.py — standalone CLI wrapper

For partial purges (where the elevation profile spans more than the purge section),
two animation pairs are produced:
  <base>_profile.gif      — zoomed to the purge section (start→end)
  <base>_profile_full.gif — full elevation-profile extent for pipeline context
"""
from __future__ import annotations

import io
import os
import re
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
# Label derivation from scenario name
# ---------------------------------------------------------------------------

def _parse_endpoint_labels(scenario_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to extract start/end station names from the scenario name string.

    Patterns tried (first match wins):
      "… - Sinclair to Bartlett …"  → ("Sinclair", "Bartlett")
      "SP to MT …"                  → ("SP", "MT")
    """
    # Pattern 1: "- <start> to <end>" (possibly with parenthesised suffix)
    m = re.search(r'-\s*(.+?)\s+to\s+(.+?)(?:\s*[\(\[]|$)', scenario_name, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Pattern 2: bare "<word> to <word>"
    m = re.search(r'\b(\w[\w\s]*?)\s+to\s+([\w][\w\s]*?)(?:\s|$)', scenario_name, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


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

def _render_frame(step, cfg, roadmap, phases, figsize, dpi,
                  title=None, start_label=None, end_label=None, xlim=None):
    """Render one frame and return a PIL RGB Image."""
    from ..ui.charts import plot_pipeline_profile
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plot_pipeline_profile(fig, step, cfg, roadmap=roadmap,
                          annotation=_label_for_mp(step.pig_mp, phases),
                          title=title, start_label=start_label, end_label=end_label,
                          xlim=xlim)
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
    n_frames: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 7),
    dpi: int = 96,
    frame_ms: int = 450,
    progress_cb=None,
    # Label / title overrides
    title: Optional[str] = None,
    start_label: Optional[str] = None,
    end_label: Optional[str] = None,
    scenario_name: Optional[str] = None,
) -> None:
    """
    Render the animated pipeline profile to GIF (and optionally MP4).

    For partial purges (elevation data extends beyond the purge section), a second
    pair of files is also written: <stem>_full.gif / <stem>_full.mp4 showing the
    full elevation extent for operational context.

    Args:
        results:       SimResults from simulate()
        cfg:           SimConfig used for the run
        out_gif:       Output .gif path — purge-section view (always produced)
        out_mp4:       Optional .mp4 path; skipped silently if no encoder available
        phases:        [(start_mp, end_mp, annotation), ...]; auto-detected if None
        mp_step:       Route miles per frame; sets the frame count when n_frames is None
        n_frames:      Number of frames, evenly spaced in elapsed time
        figsize:       matplotlib figure size in inches (width, height)
        dpi:           Render resolution
        frame_ms:      Display time per GIF frame in milliseconds
        progress_cb:   Optional callable(frames_done: int, total_frames: int)
        title:         Chart title override
        start_label:   X-axis label for purge start
        end_label:     X-axis label for purge end
        scenario_name: Used to auto-derive start/end labels if not provided
    """
    if not _MPL:
        raise ImportError("matplotlib required:  pip install matplotlib")
    if not _PIL:
        raise ImportError("Pillow required:  pip install Pillow")
    if not results.steps:
        raise ValueError("No simulation steps to animate")

    # --- Label derivation ---------------------------------------------------
    # 1. Explicit params win.
    # 2. Pump station names at start/end.
    # 3. Scenario name parsing.
    # 4. "MP X.X" fallback.
    if start_label is None or end_label is None:
        ps_sorted = sorted(getattr(cfg, 'pump_stations', []), key=lambda p: p.mp)
        if start_label is None:
            matches = [p for p in ps_sorted if abs(p.mp - cfg.purge_start_mp) < 2.0]
            start_label = matches[0].name if matches else None
        if end_label is None:
            bpcv = getattr(cfg, 'bpcv', None)
            if bpcv and abs(bpcv.mp - cfg.purge_end_mp) < 2.0:
                end_label = bpcv.name
            else:
                matches = [p for p in ps_sorted if abs(p.mp - cfg.purge_end_mp) < 2.0]
                end_label = matches[-1].name if matches else None

    if (start_label is None or end_label is None) and scenario_name:
        sl, el = _parse_endpoint_labels(scenario_name)
        if start_label is None:
            start_label = sl
        if end_label is None:
            end_label = el

    if phases is None:
        phases = _auto_phases(results, cfg)

    # --- Sample steps evenly in elapsed time ---------------------------------
    # One frame per mp_step miles of route sets the frame count; the frames themselves
    # are evenly spaced in time, so playback runs at a constant hours-per-second instead
    # of rushing through slow stretches (or, sampled by step, stalling on the launch).
    if n_frames is None:
        n_frames = int(np.ceil((cfg.purge_end_mp - cfg.purge_start_mp) / mp_step)) + 1
    sample = [results.steps[i] for i in results.time_frame_indices(max(2, n_frames))]

    # --- Detect partial purge (elevation data wider than purge section) -----
    ep_mp = np.asarray(cfg.elevation_profile, dtype=float)[:, 0]
    full_xlim = None
    is_partial = (float(ep_mp[0]) < cfg.purge_start_mp - 0.5 or
                  float(ep_mp[-1]) > cfg.purge_end_mp + 0.5)
    if is_partial:
        full_xlim = (float(ep_mp[0]) - 3, float(ep_mp[-1]) + 3)

    # --- Render frames -------------------------------------------------------
    total = len(sample)
    frames_purge: list = []
    frames_full:  list = []

    for i, step in enumerate(sample):
        frames_purge.append(_render_frame(step, cfg, results.roadmap, phases, figsize, dpi,
                                          title=title, start_label=start_label,
                                          end_label=end_label, xlim=None))
        if is_partial:
            frames_full.append(_render_frame(step, cfg, results.roadmap, phases, figsize, dpi,
                                             title=title, start_label=start_label,
                                             end_label=end_label, xlim=full_xlim))
        if progress_cb:
            progress_cb(i + 1, total)

    fps = max(1, round(1000 / frame_ms))

    # --- Write purge-section GIF / MP4 --------------------------------------
    _write_gif(frames_purge, out_gif, frame_ms)
    if out_mp4:
        _write_mp4(frames_purge, out_mp4, fps)

    # --- Write full-line GIF / MP4 (partial purges only) --------------------
    if is_partial and frames_full:
        stem, ext = os.path.splitext(out_gif)
        full_gif = stem + "_full" + ext
        _write_gif(frames_full, full_gif, frame_ms)
        if out_mp4:
            full_mp4 = os.path.splitext(out_mp4)[0] + "_full" + os.path.splitext(out_mp4)[1]
            _write_mp4(frames_full, full_mp4, fps)
        print(f"  Full-line view -> {full_gif}")


def _write_gif(frames: list, path: str, frame_ms: int) -> None:
    gif_dir = os.path.dirname(os.path.abspath(path))
    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=True)


def _write_mp4(frames: list, path: str, fps: int = 2) -> None:
    mp4_dir = os.path.dirname(os.path.abspath(path))
    if mp4_dir:
        os.makedirs(mp4_dir, exist_ok=True)

    # Method 1: imageio (preferred)
    try:
        import imageio
        imageio.mimwrite(path, [np.array(f) for f in frames], fps=fps)
        return
    except Exception as e:
        print(f"  MP4 (imageio): {e} — trying ffmpeg writer…")

    # Method 2: matplotlib FuncAnimation + ffmpeg
    try:
        import matplotlib.animation as manim
        arr0 = np.array(frames[0])
        h, w = arr0.shape[:2]
        _dpi = 96
        fig2, ax2 = plt.subplots(figsize=(w / _dpi, h / _dpi), dpi=_dpi)
        ax2.axis("off")
        fig2.subplots_adjust(0, 0, 1, 1)
        im = ax2.imshow(arr0)

        def _upd(i, _frames=frames):
            im.set_array(np.array(_frames[i]))
            return [im]

        ani = manim.FuncAnimation(fig2, _upd, frames=len(frames),
                                   interval=500, blit=True)
        ani.save(path, writer="ffmpeg", fps=fps, dpi=_dpi)
        plt.close(fig2)
    except Exception as e2:
        print(f"  MP4 skipped (no encoder available): {e2}")
