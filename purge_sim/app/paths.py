"""
Where the standalone app finds its bundled files and keeps the user's own.

Bundled (read-only): the `scenarios/` library and the web UI. When the app is
frozen by PyInstaller these live under sys._MEIPASS; from a source checkout they
live in the repo.

User data (writable): scenarios the user saves, run outputs, and settings. These
never go inside the repo or the frozen bundle.
"""

from __future__ import annotations

import os
import sys


def resource_root() -> str:
    """Root that holds `scenarios/` and `purge_sim/` (repo root, or the frozen bundle)."""
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return frozen
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bundled_scenarios_dir() -> str:
    return os.path.join(resource_root(), "scenarios")


def static_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def guide_path() -> str:
    """The user guide shown on the app's Guide tab (docs/USER_GUIDE.md)."""
    return os.path.join(resource_root(), "docs", "USER_GUIDE.md")


def _home(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def user_scenarios_dir() -> str:
    # Same folder the Tkinter app's Save dialog defaults to, so both apps share scenarios.
    return os.environ.get("PURGE_SIM_SCENARIOS_DIR") or _home("PurgeSimScenarios")


def outputs_dir() -> str:
    return os.environ.get("PURGE_SIM_OUTPUTS_DIR") or _home("PurgeSimOutputs")


def settings_path() -> str:
    return os.environ.get("PURGE_SIM_SETTINGS") or _home(".purge_sim", "settings.json")
