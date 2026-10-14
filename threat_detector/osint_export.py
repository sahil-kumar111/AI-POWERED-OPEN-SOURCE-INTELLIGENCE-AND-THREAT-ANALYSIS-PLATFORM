"""Stable JSON export of existing Location Threat Monitor analysis results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_osint_results(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save the exact already-analyzed rows as UTF-8 JSON; no re-analysis is performed."""
    if not isinstance(results, list):
        raise TypeError("OSINT results must be a list")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path
