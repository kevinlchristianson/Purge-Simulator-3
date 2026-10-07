"""
Product library for the job intake: typical specific gravity, viscosity and vapor
pressure for the fluids past purges have displaced.

Values are the ones past jobs used when the client gave nothing better (diesel
0.838 / 3.5 cSt, light sweet crude 0.82 / 3.5 cSt, heavy crude 0.93 / 325 cSt,
commercial butane 0.575 / 0.28 cSt, Y1 NGL 0.51 / 0.18 cSt), so they are starting
points, not data. The intake labels each one ASSUMED until the user overrides it.

Volatile products (butane, propane, NGL) carry a vapor pressure: the liquid column
has to stay above it everywhere ahead of the pig or it flashes. Past jobs held
100 psi above vapor pressure at the exit and at every peak (PH-30: Y1 at ~300 psia
-> 385 psig minimum).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

ATM_PSI = 14.7
VAPOR_MARGIN_PSI = 100.0


@dataclass(frozen=True)
class Fluid:
    key: str
    name: str
    sg: float
    viscosity_cst: float
    vapor_psia: Optional[float] = None   # at ~100 F; None = not volatile at purge pressures
    note: str = ""

    @property
    def min_liquid_psig(self) -> Optional[float]:
        """Lowest pressure the column may see anywhere: vapor pressure + margin."""
        if self.vapor_psia is None:
            return None
        return round(self.vapor_psia - ATM_PSI + VAPOR_MARGIN_PSI, 0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["min_liquid_psig"] = self.min_liquid_psig
        return d


FLUIDS: Dict[str, Fluid] = {f.key: f for f in [
    Fluid("diesel", "Diesel", 0.838, 3.5, None, "ULSD at ~60 F"),
    Fluid("gasoline", "Gasoline", 0.74, 0.6, 12.0, "RVP ~9-12 psi; any tankage backpressure covers it"),
    Fluid("jet", "Jet fuel / kerosene", 0.80, 1.6),
    Fluid("light_crude", "Light sweet crude (API ~40)", 0.82, 3.5),
    Fluid("medium_crude", "Medium crude (API ~30)", 0.876, 15.0),
    Fluid("heavy_crude", "Heavy crude (API ~20)", 0.93, 325.0, None, "laminar at purge speeds"),
    Fluid("butane", "Butane (commercial spec)", 0.575, 0.28, 70.0),
    Fluid("propane", "Propane", 0.507, 0.20, 190.0),
    Fluid("ngl_y1", "Y1 NGL (de-ethanized / EP mix)", 0.51, 0.18, 300.0),
    Fluid("water", "Water", 1.0, 1.0),
]}


def get(key: str) -> Fluid:
    try:
        return FLUIDS[key]
    except KeyError:
        raise KeyError(f"unknown fluid {key!r}; one of {sorted(FLUIDS)}")


def options() -> List[dict]:
    return [{"value": f.key, "label": f.name} for f in FLUIDS.values()]


def api_to_sg(api: float) -> float:
    return 141.5 / (api + 131.5)


def match(name: str) -> Optional[str]:
    """Best library key for a free-text fluid name (from an old scenario), or None."""
    n = (name or "").lower()
    for key, words in [("ngl_y1", ["ngl", "y1", "y-grade"]), ("butane", ["butane"]),
                       ("propane", ["propane"]), ("diesel", ["diesel", "ulsd"]),
                       ("gasoline", ["gasoline", "mogas"]), ("jet", ["jet", "kerosene"]),
                       ("water", ["water"])]:
        if any(w in n for w in words):
            return key
    if "crude" in n:
        return "heavy_crude" if "heavy" in n else "light_crude" if ("light" in n or "sweet" in n) else None
    return None
