"""
Company branding for the app, the client web page and the xlsx reports.

The brand is chosen in the app's Settings and kept in the user settings file
(~/.purge_sim/settings.json, key "brand"), so each company that uses the
simulator can carry its own name, colors and logo. A custom logo is copied next
to the settings file. Presets:

  enermech  EnerMech logo, blue #00489A to green #128D36 (the default)
  plain     no logo or company name, the original navy/blue report colors
  custom    the company name, colors and logo entered in Settings

Only chrome is branded (header bars, buttons, report title and header fills).
Chart data colors (N2, liquid, MOP...) stay fixed so a chart reads the same
whichever brand is on.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Optional

PRESETS = {
    "enermech": {"name": "EnerMech", "primary": "#00489A", "secondary": "#128D36",
                 "mid": "#084766", "highlight": "#23A63D", "logo_on": "dark"},
    "plain": {"name": "", "primary": "#1F3864", "secondary": "#2E75B6",
              "mid": "#344E6E", "highlight": "#2E75B6", "logo_on": "dark"},
}
DEFAULT_PRESET = "enermech"
_BUILTIN_LOGOS = {"enermech": "enermech-logo-white.png"}
LOGO_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
MAX_LOGO_BYTES = 1024 * 1024
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def settings_path() -> str:
    # Same file as purge_sim.app.paths.settings_path(); kept here so the engine's
    # report writers can read the brand without importing the app.
    return os.environ.get("PURGE_SIM_SETTINGS") or os.path.join(
        os.path.expanduser("~"), ".purge_sim", "settings.json")


def _stored() -> dict:
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        b = data.get("brand") if isinstance(data, dict) else None
        return b if isinstance(b, dict) else {}
    except (OSError, ValueError):
        return {}


def _mix(a: str, b: str, t: float) -> str:
    """Blend hex color a toward b by t (0..1)."""
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(pa, pb))


def builtin_logo_path(preset: str) -> Optional[str]:
    name = _BUILTIN_LOGOS.get(preset)
    if not name:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "static", "brand", name)


def custom_logo_path() -> Optional[str]:
    base = os.path.join(os.path.dirname(settings_path()), "brand_logo")
    for ext in LOGO_TYPES:
        if os.path.isfile(base + ext):
            return base + ext
    return None


def current() -> dict:
    """The active brand: preset, name, colors (#RRGGBB) and logo path (or None)."""
    s = _stored()
    preset = s.get("preset") if s.get("preset") in ("enermech", "plain", "custom") else DEFAULT_PRESET
    if preset == "custom":
        b = dict(PRESETS["plain"])
        b["name"] = str(s.get("name") or "")[:80]
        for k in ("primary", "secondary", "highlight"):
            if isinstance(s.get(k), str) and _HEX.match(s[k]):
                b[k] = s[k].upper()
        b["mid"] = _mix(b["primary"], b["secondary"], 0.5)
        b["logo_on"] = "light" if s.get("logo_on") == "light" else "dark"
        logo = custom_logo_path()
    else:
        b = dict(PRESETS[preset])
        logo = builtin_logo_path(preset)
    b["preset"] = preset
    b["logo_path"] = logo if logo and os.path.isfile(logo) else None
    b["dark"] = _mix(b["primary"], "#000000", 0.25)      # xlsx title bar
    b["tint"] = _mix(b["primary"], "#FFFFFF", 0.92)      # xlsx zebra rows
    return b


def logo_data_uri(b: Optional[dict] = None) -> str:
    """The logo as a data: URI ("" when there is none), for pages that must stay one file."""
    b = b or current()
    p = b.get("logo_path")
    if not p:
        return ""
    with open(p, "rb") as f:
        raw = f.read()
    mime = LOGO_TYPES.get(os.path.splitext(p)[1].lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def public(b: Optional[dict] = None) -> dict:
    """What the UI and the client page need: colors, name and the logo as a data URI."""
    b = b or current()
    out = {k: b[k] for k in ("preset", "name", "primary", "secondary", "mid", "highlight", "logo_on")}
    out["logo"] = logo_data_uri(b)
    out["has_custom_logo"] = bool(custom_logo_path())
    return out


_xlsx_cache: list = [None, None]   # [settings file mtime, brand], so a report doesn't re-read it per cell


def xlsx_color(c: str) -> str:
    """Resolve a report color: "brand:<key>" becomes the active brand's RRGGBB, anything else passes."""
    if not (isinstance(c, str) and c.startswith("brand:")):
        return c
    try:
        mtime = os.path.getmtime(settings_path())
    except OSError:
        mtime = 0.0
    if _xlsx_cache[0] != mtime or _xlsx_cache[1] is None:
        _xlsx_cache[:] = [mtime, current()]
    return _xlsx_cache[1][c[6:]].lstrip("#").upper()


def validate(d: dict) -> dict:
    """Clean a brand setting from the Settings dialog; raises ValueError on a bad value."""
    preset = d.get("preset")
    if preset not in ("enermech", "plain", "custom"):
        raise ValueError("branding must be enermech, plain or custom")
    out = {"preset": preset}
    if preset == "custom":
        out["name"] = str(d.get("name") or "").strip()[:80]
        for k in ("primary", "secondary", "highlight"):
            v = d.get(k)
            if v is not None:
                if not (isinstance(v, str) and _HEX.match(v)):
                    raise ValueError(f"{k} color must look like #1A2B3C")
                out[k] = v.upper()
        out["logo_on"] = "light" if d.get("logo_on") == "light" else "dark"
    return out


def save_logo(filename: str, raw: bytes) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in LOGO_TYPES:
        raise ValueError("logo must be a .png, .jpg or .svg file")
    if not raw or len(raw) > MAX_LOGO_BYTES:
        raise ValueError("logo must be under 1 MB")
    remove_logo()
    path = os.path.join(os.path.dirname(settings_path()), "brand_logo" + (".jpg" if ext == ".jpeg" else ext))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(raw)
    return path


def remove_logo() -> None:
    p = custom_logo_path()
    while p:
        os.remove(p)
        p = custom_logo_path()
