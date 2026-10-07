"""
Line pipe lookups: nominal size -> OD, standard wall thicknesses, API 5L grades,
flange-class pressure ratings and the Barlow pressure check.

These replace the numbers that used to be looked up by hand for every new job
("6 inch" -> 6.625" OD, "what WT is STD for 8 inch", "does 0.219 X42 hold 1440").
Every value a job relies on still has to be confirmed against the client's data;
the intake marks anything taken from here as an assumption.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..engine.constants import NPS_OD_IN, resolve_nps

# ASME B36.10M standard (STD) and extra-strong (XS) wall thickness, inches.
# NPS 14 and up: STD = 0.375, XS = 0.500.
_STD_XS_WT_IN: Dict[str, tuple] = {
    "2": (0.154, 0.218), "2 1/2": (0.203, 0.276), "3": (0.216, 0.300), "3 1/2": (0.226, 0.318),
    "4": (0.237, 0.337), "5": (0.258, 0.375), "6": (0.280, 0.432), "8": (0.322, 0.500),
    "10": (0.365, 0.500), "12": (0.375, 0.500),
}

# API 5L specified minimum yield strength, psi.
GRADE_SMYS_PSI: Dict[str, float] = {
    "B": 35_000, "X42": 42_000, "X46": 46_000, "X52": 52_000, "X56": 56_000,
    "X60": 60_000, "X65": 65_000, "X70": 70_000, "X80": 80_000,
}

# Flange class -> pressure rating, psig. These are the ratings past jobs have used for
# a flat "ANSI class" MOP envelope (e.g. Poplar-Richey: ANSI 600 = 1440 psig), the
# older B16.5 carbon-steel figures still common in pipeline practice. Current B16.5
# Group 1.1 at 100 F is slightly higher (285 / 740 / 1480 / 2220 / 3705); the lower
# figures are the conservative choice. Confirm against the client's station ratings.
FLANGE_CLASS_PSIG: Dict[str, float] = {
    "150": 275.0, "300": 720.0, "600": 1440.0, "900": 2160.0, "1500": 3600.0,
}

# 49 CFR 195.106 design factor for liquid lines (0.72); Barlow uses it with E = T = 1.
DEFAULT_DESIGN_FACTOR = 0.72


def nps_key(nps) -> str:
    """Normalize '6', 6, '6"', '6 in', 'NPS 6' to a key of NPS_OD_IN. Raises KeyError."""
    s = str(nps).strip().lower().replace('"', "").replace("inch", "").replace("in", "")
    s = s.replace("nps", "").strip()
    return resolve_nps(s)


def od_in(nps) -> float:
    return NPS_OD_IN[nps_key(nps)]


def standard_wt_in(nps, weight: str = "STD") -> Optional[float]:
    """STD or XS wall for an NPS, or None where the table has no entry (below NPS 2)."""
    k = nps_key(nps)
    i = 0 if weight.upper() == "STD" else 1
    if k in _STD_XS_WT_IN:
        return _STD_XS_WT_IN[k][i]
    try:
        if float(k) >= 14:
            return (0.375, 0.500)[i]
    except ValueError:
        pass
    return None


def nps_for_od(od: float) -> Optional[str]:
    for k, v in NPS_OD_IN.items():
        if abs(v - od) < 0.01:
            return k
    return None


def barlow_psig(od_in_: float, wt_in: float, grade: str,
                design_factor: float = DEFAULT_DESIGN_FACTOR) -> float:
    """Barlow design pressure P = 2 S t F / D (E = T = 1)."""
    smys = GRADE_SMYS_PSI[grade.upper()]
    return 2.0 * smys * wt_in * design_factor / od_in_


def min_grade_for(od_in_: float, wt_in: float, mop_psig: float,
                  design_factor: float = DEFAULT_DESIGN_FACTOR) -> Optional[str]:
    """Lowest API 5L grade whose Barlow pressure covers mop_psig, or None if none does."""
    for g in GRADE_SMYS_PSI:
        if barlow_psig(od_in_, wt_in, g, design_factor) >= mop_psig - 1e-6:
            return g
    return None


def barlow_check(od_in_: float, wt_in: float, mop_psig: float, grade: Optional[str] = None,
                 design_factor: float = DEFAULT_DESIGN_FACTOR) -> dict:
    """Does the pipe support the stated MOP? With a grade: its Barlow pressure and the
    verdict. Without one: the minimum grade that would justify the MOP."""
    out = {"od_in": od_in_, "wt_in": wt_in, "mop_psig": mop_psig, "design_factor": design_factor,
           "min_grade": min_grade_for(od_in_, wt_in, mop_psig, design_factor)}
    if grade and grade.upper() in GRADE_SMYS_PSI:
        p = barlow_psig(od_in_, wt_in, grade, design_factor)
        out.update(grade=grade.upper(), barlow_psig=round(p, 0), ok=p >= mop_psig - 1e-6)
    return out
