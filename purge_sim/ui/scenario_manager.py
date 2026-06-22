"""
Scenario manager — directory-backed collection of named scenarios.
Thin wrapper around data/scenario.py for use by the UI.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..data.scenario import Scenario, load_scenario, save_scenario, list_scenarios


class ScenarioManager:
    """Manages a directory of scenario JSON files."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def list(self) -> List[dict]:
        """Return list of {name, path, modified_at, notes} dicts."""
        return list_scenarios(self.directory)

    def load(self, path: str) -> Scenario:
        return load_scenario(path)

    def save(self, scenario: Scenario, name: str) -> str:
        """Save scenario to directory/name.json. Returns the file path."""
        path = os.path.join(self.directory, f"{name}.json")
        save_scenario(scenario, path)
        return path

    def delete(self, path: str) -> None:
        if os.path.isfile(path):
            os.remove(path)

    def exists(self, name: str) -> bool:
        return os.path.isfile(os.path.join(self.directory, f"{name}.json"))
