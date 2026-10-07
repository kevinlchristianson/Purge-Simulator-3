"""
Tell which kind of pipeline data file this is, without parsing all of it.

  'ili'     — Rosen ILI tally (data/ili_parser.py): milepost header plus pipe/MOP/
              description columns
  'pxp'     — point-by-point MOP and operating pressure sheet (data/pxp_parser.py)
  'profile' — anything else with route data: KMZ/KML, TXT/CSV, or an Excel sheet with
              just milepost/elevation or lat/lon columns (data/profile_parser.py)

Only the first rows of each sheet are read (data/xlsx_reader.py), so this is quick
even on a 20,000-row workbook.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

from .pxp_parser import find_pxp_header
from .xlsx_reader import read_sheets

FORMAT_LABELS = {
    "ili": "Rosen ILI tally",
    "pxp": "point-by-point (PxP) pressure sheet",
    "profile": "elevation profile",
}

_PEEK_ROWS = 30


def _peek_excel(path: str) -> Dict[str, List[list]]:
    return read_sheets(path, max_rows=_PEEK_ROWS)


def _is_rosen(rows: List[list]) -> bool:
    """A milepost header row that also has pipe or MOP columns. A plain milepost/elevation
    sheet is a profile, not a tally."""
    for row in rows:
        text = " ".join(str(v).lower() for v in row if v is not None)
        if not re.search(r"mile ?post|calculated mile", text):
            continue
        return any(k in text for k in ("wall thickness", "maximum operating pressure", "additional description",
                                       "pipe diameter", "joint"))
    return False


def detect_format(path: str) -> str:
    """'ili', 'pxp' or 'profile' (see module docstring)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".xlsx", ".xls", ".xlsm"):
        return "profile"
    sheets = _peek_excel(path)
    if any(find_pxp_header([tuple(r) for r in rows]) is not None for rows in sheets.values()):
        return "pxp"
    # parse_ili reads the first sheet, so that's the one that has to be a tally
    first = next(iter(sheets.values()), [])
    if _is_rosen(first):
        return "ili"
    return "profile"
