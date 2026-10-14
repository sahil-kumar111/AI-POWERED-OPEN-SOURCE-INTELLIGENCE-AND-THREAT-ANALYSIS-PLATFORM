from __future__ import annotations

from typing import Any
from .contracts import clip_text, number


def explain_osint_result(result: dict[str, Any]) -> dict[str, Any]:
    """Build an evidence-trace explanation from a Location Threat Monitor row.

    This explains fields already returned by analysis_engine.py. It does not
    fabricate a model's hidden reasoning or infer absent geographic thresholds.
    """
    if not isinstance(result, dict):
        raise ValueError("OSINT result must be a JSON object")
    title = clip_text(result.get("title"), 400) or "Untitled source item"
    category = result.get("category", "Unknown")
    severity = result.get("severity", "Unknown")
    status = result.get("geographic_classification", "Unknown")
    distance = number(result.get("distance_km"))
    evidence = []
    if category and category != "Unknown": evidence.append({"field": "category", "value": category, "meaning": "Threat category supplied by the upstream analysis pipeline."})
    if severity and severity != "Unknown": evidence.append({"field": "severity", "value": severity, "meaning": "Severity label supplied by the upstream analysis pipeline; its exact rule is not exposed in this result row."})
    if status and status != "Unknown": evidence.append({"field": "geographic_classification", "value": status, "meaning": _geo_meaning(status)})
    if distance is not None: evidence.append({"field": "distance_km", "value": round(distance, 3), "meaning": "Reported distance from the monitoring location, preserved from the source result."})
    if result.get("incident_location"): evidence.append({"field": "incident_location", "value": result.get("incident_location"), "meaning": "Location text reported for the article/incident; not independently verified here."})
    if result.get("source"): evidence.append({"field": "source", "value": result.get("source"), "meaning": "Publisher/source label attached to the result."})
    if result.get("publishedAt"): evidence.append({"field": "publishedAt", "value": result.get("publishedAt"), "meaning": "Publication timestamp, which is not necessarily the incident timestamp."})
    desc = clip_text(result.get("description"), 600)
    if desc: evidence.append({"field": "description", "value": desc, "meaning": "Article summary text; treat as reported text rather than independently verified fact."})
    return {
        "explanation_type": "osint_result_evidence_trace",
        "title": title, "summary": f"The upstream pipeline labelled this result as category '{category}', severity '{severity}', and geographic status '{status}'. The explanation below exposes which saved fields support that summary; it does not invent hidden model logic.",
        "evidence": evidence,
        "limitations": ["This is a trace of output fields, not SHAP or a causal explanation of the upstream classifier.", "Publication time can differ from incident time.", "Article text, incident location, and severity require source review."],
    }


def _geo_meaning(status: str) -> str:
    lowered = str(status).lower()
    if lowered == "local": return "The upstream pipeline classified the incident as Local."
    if lowered == "nearby": return "The upstream pipeline classified the incident as Nearby."
    if lowered == "outside": return "The upstream pipeline classified the incident as Outside the selected proximity class."
    return "Geographic class assigned by the upstream pipeline; its threshold is not available in this result row."
