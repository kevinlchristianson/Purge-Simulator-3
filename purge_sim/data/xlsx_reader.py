"""
Fast cell-value reader for .xlsx/.xlsm workbooks.

Operator data workbooks (PxP pressure sheets especially) carry chart sheets that make
openpyxl.load_workbook slow, since it parses every chart even in read-only mode: about
20 s on the GL-09 PxP before reading a single cell. This reads the worksheet XML
directly with the standard library and stops at the rows and columns asked for, which
takes a few seconds for the same file. .xls (old binary Excel) falls back to pandas.

Values come back as strings (numbers unconverted, cached formula results included) or
None, the way they're stored in the file.
"""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col_index(ref: str) -> int:
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def peek_xlsx(path: str, max_rows: Optional[int] = 30, sheet: Optional[str] = None,
              max_col: Optional[int] = None) -> Dict[str, List[list]]:
    """{sheet name: first max_rows rows as lists of cell values} for an .xlsx/.xlsm.
    max_rows=None reads every row; sheet limits it to one sheet; max_col drops cells past
    that many columns. Values are strings (numbers unconverted) or None."""
    out: Dict[str, List[list]] = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: List[str] = []
        if "xl/sharedStrings.xml" in names:
            with z.open("xl/sharedStrings.xml") as f:
                for _, el in ET.iterparse(f):
                    if el.tag == _NS + "si":
                        shared.append("".join(t.text or "" for t in el.iter(_NS + "t")))
                        el.clear()
        rels = {}
        with z.open("xl/_rels/workbook.xml.rels") as f:
            for el in ET.parse(f).getroot():
                target = el.get("Target", "")
                target = target.lstrip("/") if target.startswith("/") else "xl/" + target
                rels[el.get("Id")] = target
        with z.open("xl/workbook.xml") as f:
            sheets = [(s.get("name"), rels.get(s.get(_REL_NS + "id")))
                      for s in ET.parse(f).getroot().iter(_NS + "sheet")]
        for name, target in sheets:
            if not target or target not in names or "worksheets/" not in target:
                continue                       # chart sheets and the like
            if sheet is not None and name != sheet:
                continue
            rows: List[list] = []
            with z.open(target) as f:
                for _, el in ET.iterparse(f):
                    if el.tag != _NS + "row":
                        continue
                    row: list = []
                    for c in el.iter(_NS + "c"):
                        i = _col_index(c.get("r", "")) if c.get("r") else len(row)
                        if max_col is not None and i >= max_col:
                            continue
                        t = c.get("t")
                        v = c.find(_NS + "v")
                        if t == "inlineStr":
                            val = "".join(x.text or "" for x in c.iter(_NS + "t"))
                        elif v is None:
                            val = None
                        elif t == "s":
                            val = shared[int(v.text)]
                        else:
                            val = v.text
                        row.extend([None] * (i + 1 - len(row)))
                        row[i] = val
                    rows.append(row)
                    el.clear()
                    if max_rows is not None and len(rows) >= max_rows:
                        break
            out[name] = rows
    return out


def read_sheets(path: str, max_rows: Optional[int] = None, sheet: Optional[str] = None,
                max_col: Optional[int] = None) -> Dict[str, List[list]]:
    """peek_xlsx for .xlsx/.xlsm, pandas for anything else Excel can open."""
    if path.lower().endswith((".xlsx", ".xlsm")):
        try:
            return peek_xlsx(path, max_rows, sheet, max_col)
        except (zipfile.BadZipFile, KeyError, ET.ParseError):
            pass
    import pandas as pd
    frames = pd.read_excel(path, sheet_name=sheet if sheet is not None else None, header=None, nrows=max_rows)
    if sheet is not None:
        frames = {sheet: frames}
    out = {}
    for name, df in frames.items():
        if max_col is not None:
            df = df.iloc[:, :max_col]
        out[name] = df.astype(object).where(df.notna(), None).values.tolist()
    return out
