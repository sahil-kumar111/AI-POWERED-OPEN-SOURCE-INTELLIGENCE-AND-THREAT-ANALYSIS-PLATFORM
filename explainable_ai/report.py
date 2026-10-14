from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .contracts import read_json, utc_now
from .video_explainer import explain_video_report
from .nlp_explainer import explain_nlp_output
from .fusion_explainer import explain_fusion_score


def build_xai_report(*, video_report: str | Path | dict[str, Any] | None = None,
                     combined_intelligence: str | Path | dict[str, Any] | None = None,
                     threat_score: str | Path | dict[str, Any] | None = None,
                     osint_results: str | Path | list[Any] | None = None,
                     visual_explanation: dict[str, Any] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "aegis.xai.v1",
        "generated_at_utc": utc_now(),
        "explanation_policy": "Explain saved upstream evidence without altering predictions or the fusion score.",
        "explanations": {},
    }
    if video_report is not None: report["explanations"]["video"] = explain_video_report(video_report)
    if combined_intelligence is not None: report["explanations"]["nlp"] = explain_nlp_output(combined_intelligence)
    if threat_score is not None: report["explanations"]["fusion"] = explain_fusion_score(threat_score)
    if osint_results is not None:
        from .osint_explainer import explain_osint_result
        raw = read_json(osint_results)
        rows = raw if isinstance(raw, list) else raw.get("results", raw.get("events", [])) if isinstance(raw, dict) else []
        report["explanations"]["osint"] = [explain_osint_result(row) for row in rows if isinstance(row, dict)]
    if visual_explanation is not None:
        report["explanations"]["visual"] = visual_explanation
    if not report["explanations"]:
        raise ValueError("Provide at least one input report to explain")
    return report


def write_xai_report(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    report = build_xai_report(**kwargs)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
