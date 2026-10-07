"""
Map landmarks from a tally or pressure sheet: block valves, launchers/receivers, aerial
markers, above-ground markers and crossings. They are for the app's Map tab only; the
engine never reads them (pump stations, check valves and the BPCV are parsed separately
because they change the hydraulics).

A PxP sheet carries a feature code (AM, AGR, MKP, RR, ...) and a valve type column; a
Rosen tally only has free text, so both go through classify().
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

# kind -> label shown on the map
KINDS = {
    "block_valve": "Block valve",
    "launcher_receiver": "Launcher / receiver",
    "aerial_marker": "Aerial marker",
    "ground_marker": "Above-ground marker",
    "crossing": "Crossing",
}

# PxP 'Descriptions' feature codes
_CODE_KIND = {
    "AM": "aerial_marker",
    "AGR": "ground_marker", "AGM": "ground_marker", "MKP": "ground_marker",
    "REC": "launcher_receiver", "LAU": "launcher_receiver", "TRAP": "launcher_receiver",
    "RR": "crossing", "LAK": "crossing", "RIV": "crossing", "RD": "crossing", "HWY": "crossing",
}
# codes that are hydraulics (parsed elsewhere) or not worth a map pin
_SKIP_CODES = {"PMP", "CHECK", "PIPE", "TWL", "STQ", ""}
_CHECK_VALVE_TYPES = {"CHECK", "SWING CHECK", "CV"}

_RE_CHECK = re.compile(r"\bcheck\s*valve\b|\bcv\b|back\s*pressure|\bbpcv\b", re.I)
_RE_TRAP = re.compile(r"\blauncher\b|\breceiver\b|\bpig\s*trap\b|\bscraper\s*trap\b", re.I)
_RE_VALVE = re.compile(r"\bblock\s*valve\b|\bmainline\s*valve\b|\bmlv\b|\bbv\b|\bvalve\b", re.I)
_RE_AERIAL = re.compile(r"\baerial\s*marker\b|^\s*am\s*\d", re.I)
_RE_GROUND = re.compile(r"\bagm\b|\bagr\b|above\s*ground\s*marker|marker\s*plate|\bmarker\b", re.I)
_RE_CROSSING = re.compile(r"\brail\s*road\b|\brailway\b|\brr\b|\broad\b|\bhighway\b|\bhwy\b|\briver\b|"
                          r"\bcreek\b|\bhdd\b|\bcrossing\b|\blake\b|\bpond\b|\binterstate\b", re.I)


def classify(code: str = "", text: str = "", valve: str = "", coded: bool = False) -> Optional[str]:
    """The landmark kind for one row, or None. code: a PxP feature code; text: the row's
    name/description; valve: a PxP valve type (BALL, GATE, CHECK...). coded=True for a
    sheet with feature codes and a valve column (PxP): valves and markers come only from
    those, since its free text also names valves in notes ("D/S END KRUSE BV")."""
    code = (code or "").strip().upper()
    text = (text or "").strip()
    valve = (valve or "").strip().upper()
    if code in ("PMP", "CHECK") or valve in _CHECK_VALVE_TYPES or _RE_CHECK.search(text):
        return None
    if code in _CODE_KIND:
        return _CODE_KIND[code]   # a marker standing at a valve is still a marker, not the valve
    if _RE_TRAP.search(text) or code == "REC":
        return "launcher_receiver"
    if valve:
        return "block_valve"
    if code and code not in _SKIP_CODES:
        return None   # a coded feature we don't map
    if coded:
        return "crossing" if _RE_CROSSING.search(text) else None
    if _RE_AERIAL.search(text):
        return "aerial_marker"
    if _RE_GROUND.search(text):
        return "ground_marker"
    if _RE_VALVE.search(text):
        return "block_valve"
    if _RE_CROSSING.search(text):
        return "crossing"
    return None


def _clean_name(text: str, kind: str, code: str) -> str:
    t = re.sub(r"^\s*(AGR|AGM)\s*\|\s*", "", text or "").strip()
    t = re.sub(r"\s*\|\s*", " ", t)
    if not t:
        t = KINDS[kind]
    if kind == "crossing" and code.upper() == "RR" and "RR" in t.upper():
        t = "Railroad crossing"
    return t


def collect(rows: Iterable[Tuple[float, str, str, str]], coded: bool = False,
            dedupe_mi: float = 0.02) -> List[dict]:
    """rows: (mp, code, text, valve). Returns [{mp, kind, name}] in milepost order, one per
    feature (a sheet often lists the same valve on two or three nearby rows)."""
    out: List[dict] = []
    for mp, code, text, valve in rows:
        try:
            mp = float(mp)
        except (TypeError, ValueError):
            continue
        if mp != mp:
            continue
        kind = classify(code, text, valve, coded)
        if kind is None:
            continue
        name = _clean_name(text, kind, code or "")
        if any(o["kind"] == kind and abs(o["mp"] - mp) <= dedupe_mi for o in out[-6:]):
            continue
        out.append({"mp": round(mp, 4), "kind": kind, "name": name})
    out.sort(key=lambda o: o["mp"])
    return out
