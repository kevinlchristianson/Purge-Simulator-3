"""
App settings: the Claude API key and model for the assistant, whether to ask
before sending a route's coordinates to USGS for an elevation lookup, and the
company branding (purge_sim/branding.py).

The key is never hardcoded or bundled. It comes from the ANTHROPIC_API_KEY
environment variable, or from the settings file the user fills in through the
app's Settings dialog (~/.purge_sim/settings.json, readable only by the user
where the OS supports it). The environment variable wins when both are set.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .. import branding
from .paths import settings_path

DEFAULT_MODEL = "claude-opus-5-5"


def _read() -> dict:
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data: dict) -> None:
    path = settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def api_key() -> Optional[str]:
    return os.environ.get("ANTHROPIC_API_KEY") or _read().get("anthropic_api_key") or None


def api_key_source() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "environment"
    if _read().get("anthropic_api_key"):
        return "settings"
    return "none"


def model() -> str:
    return os.environ.get("PURGE_SIM_MODEL") or _read().get("model") or DEFAULT_MODEL


def elevation_ask_first() -> bool:
    """Off by default: a KMZ without elevation is looked up in USGS 3DEP as it's imported."""
    return bool(_read().get("elevation_ask_first", False))


def public_view() -> dict:
    """Settings as shown to the UI: never includes the key itself."""
    key = api_key()
    return {
        "api_key_set": bool(key),
        "api_key_source": api_key_source(),
        "api_key_hint": ("…" + key[-4:]) if key else "",
        "model": model(),
        "elevation_ask_first": elevation_ask_first(),
        "brand": branding.public(),
    }


def update(api_key_value: Optional[str] = None, model_value: Optional[str] = None,
           clear_key: bool = False, elevation_ask_first_value: Optional[bool] = None,
           brand_value: Optional[dict] = None) -> dict:
    data = _read()
    if brand_value is not None:
        data["brand"] = branding.validate(brand_value)    # ValueError -> 400 before anything is written
    if elevation_ask_first_value is not None:
        data["elevation_ask_first"] = bool(elevation_ask_first_value)
    if clear_key:
        data.pop("anthropic_api_key", None)
    elif api_key_value:
        data["anthropic_api_key"] = api_key_value.strip()
    if model_value is not None:
        model_value = model_value.strip()
        if model_value:
            data["model"] = model_value
        else:
            data.pop("model", None)
    _write(data)
    return public_view()
