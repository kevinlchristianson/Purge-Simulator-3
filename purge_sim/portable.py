"""
Portable mode: keep scenarios, outputs and settings next to the app instead of
in the Windows user profile, so the whole thing can live on an external drive
and move between computers.

It switches on when a folder named PurgeSimData sits next to the app (next to
app.py / main.py in a source checkout, next to PurgeSimulator.exe in a build).
Everything is found relative to the app itself, so the drive letter can differ
from one computer to the next.

    PurgeSimData/
        Scenarios/        saved scenarios (was C:\\Users\\<you>\\PurgeSimScenarios)
        Outputs/          run reports, GIFs, zips (was ...\\PurgeSimOutputs)
        settings/         settings.json (API key, branding), logo, elevation cache

It works by setting the same PURGE_SIM_* environment variables the app already
reads, before anything else loads, so every module (app, Tkinter app, reports,
elevation cache) picks it up. A variable that is already set is left alone.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

DATA_FOLDER = "PurgeSimData"


def app_dir() -> str:
    """Folder holding PurgeSimulator.exe (frozen) or app.py (source checkout)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir(base: Optional[str] = None) -> str:
    return os.path.join(base or app_dir(), DATA_FOLDER)


def enable(base: Optional[str] = None, create: bool = False) -> Optional[str]:
    """Point the app's data at <app>/PurgeSimData if that folder exists (or create=True).
    Returns the data folder when portable mode is on, else None."""
    root = data_dir(base)
    if not os.path.isdir(root):
        if not create:
            return None
        os.makedirs(root, exist_ok=True)
    for sub in ("Scenarios", "Outputs", "settings"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    env = {
        "PURGE_SIM_SCENARIOS_DIR": os.path.join(root, "Scenarios"),
        "PURGE_SIM_OUTPUTS_DIR": os.path.join(root, "Outputs"),
        "PURGE_SIM_SETTINGS": os.path.join(root, "settings", "settings.json"),
        "PURGE_SIM_ELEV_CACHE": os.path.join(root, "settings", "elevation_cache.sqlite"),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)
    os.environ["PURGE_SIM_PORTABLE"] = root
    return root


def active() -> Optional[str]:
    return os.environ.get("PURGE_SIM_PORTABLE") or None
