"""Loads the Phase 1 JSON Schemas straight from docs/data_contracts.md so there is a
single source of truth (no competing copy of the schemas)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

DOC = Path(__file__).resolve().parents[1] / "docs" / "data_contracts.md"


def load_phase1_schemas() -> Dict[str, dict]:
    text = DOC.read_text()
    out = {}
    for block in re.findall(r"```json\n(.*?)```", text, re.S):
        obj = json.loads(block)
        if "$schema" in obj and "$id" in obj:
            out[obj["$id"]] = obj
    return out
