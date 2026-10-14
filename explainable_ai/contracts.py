from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


def read_json(source: str | Path | dict[str, Any] | list[Any]) -> Any:
    if isinstance(source, (dict, list)):
        return source
    with Path(source).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def clip_text(value: Any, limit: int = 600) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if not value:
        return None
    return value if len(value) <= limit else value[: limit - 1] + "…"
