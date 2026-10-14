from __future__ import annotations

from typing import Any
import re
from .contracts import number, read_json, safe_dict, safe_list


def explain_fusion_score(score: dict[str, Any] | str) -> dict[str, Any]:
    """Explain the existing deterministic weighted-rule score without changing it."""
    data = read_json(score)
    if not isinstance(data, dict):
        raise ValueError("Threat score input must be a JSON object")
    contributions = []
    for row in safe_list(data.get("contributions")):
        if not isinstance(row, dict):
            continue
        weight, confidence, sub_score = number(row.get("weight")), number(row.get("confidence")), number(row.get("sub_score"))
        recomputed = weight * confidence if weight is not None and confidence is not None else None
        contributions.append({
            "event_id": row.get("event_id"), "event_type": row.get("event_type"),
            "weight": weight, "confidence": confidence, "reported_sub_score": sub_score,
            "weight_times_confidence_check": round(recomputed, 6) if recomputed is not None else None,
            "arithmetic_matches": abs(recomputed - sub_score) <= 1e-5 if recomputed is not None and sub_score is not None else None,
            "reason": row.get("reason"), "source": safe_dict(row.get("source")),
            "raw_ref": row.get("raw_ref"), "timestamp": row.get("timestamp"),
            "timestamp_basis": row.get("timestamp_basis"),
        })
    excludes = []
    for row in safe_list(data.get("excluded_events")):
        if isinstance(row, dict):
            excludes.append({
                "event_id": row.get("event_id"), "reason": row.get("reason"),
                "source": safe_dict(row.get("source")), "raw_ref": row.get("raw_ref"),
            })

    level = data.get("threat_level")
    score_value = number(data.get("threat_score"))
    metadata = safe_dict(data.get("run_metadata"))
    # Do not let a contradictory/missing upstream metadata flag make demo fixtures
    # look like live incidents. Known synthetic markers override a false flag.
    all_event_ids = [str(x) for x in safe_list(data.get("event_ids"))]
    all_event_ids.extend(str(row.get("event_id")) for row in contributions if row.get("event_id") is not None)
    all_event_ids.extend(str(row.get("event_id")) for row in excludes if row.get("event_id") is not None)
    reasons_to_check = [str(x) for x in safe_list(data.get("reasons"))]
    reasons_to_check.extend(str(row.get("reason")) for row in contributions if row.get("reason") is not None)
    zone_text = str(data.get("zone") or "")
    synthetic_id_detected = any(re.match(r"(?i)^(demo|synthetic)([-_:]|$)", event_id.strip()) for event_id in all_event_ids)
    synthetic_reason_detected = any(
        marker in reason.lower()
        for reason in reasons_to_check
        for marker in ("synthetic fixture", "synthetic demo", "synthetic_demo", "demo fixture", "demo data")
    )
    validation_zone_detected = zone_text.strip().lower() in {"validation_only", "validation-only", "test_fixture", "test-fixture"}
    synthetic_fixture_detected = synthetic_id_detected or synthetic_reason_detected or validation_zone_detected
    score_semantics = data.get("score_semantics") or "explainable weighted rule score; not a calibrated probability"
    validation_only = bool(metadata.get("validation_only")) or synthetic_fixture_detected
    level_rules = {
        "LOW": "score < 0.30",
        "MEDIUM": "0.30 <= score < 0.60",
        "HIGH": "0.60 <= score < 0.80",
        "CRITICAL": "0.80 <= score <= 1.00",
    }
    expected_level = None
    if score_value is not None and 0 <= score_value <= 1:
        expected_level = "LOW" if score_value < 0.30 else "MEDIUM" if score_value < 0.60 else "HIGH" if score_value < 0.80 else "CRITICAL"
    return {
        "explanation_type": "deterministic_fusion_trace",
        "zone": data.get("zone"), "window": safe_dict(data.get("window")),
        "threat_score": score_value, "threat_level": level,
        "threat_level_rule": level_rules.get(str(level).upper(), "Unknown level boundary"),
        "threshold_consistency_check": expected_level == str(level).upper() if expected_level is not None and level is not None else None,
        "score_semantics": score_semantics,
        "evidence_count": data.get("evidence_count"), "event_ids": safe_list(data.get("event_ids")),
        "contributions": contributions, "exclusions": excludes,
        "reported_reasons": safe_list(data.get("reasons")),
        "contribution_sum": round(sum(x["reported_sub_score"] for x in contributions if x["reported_sub_score"] is not None), 6),
        "contribution_sum_matches_score": abs(sum(x["reported_sub_score"] for x in contributions if x["reported_sub_score"] is not None) - score_value) <= 1e-5 if score_value is not None else None,
        "validation_only": validation_only,
        "validation_basis": (
            "upstream_run_metadata_and_synthetic_marker" if metadata.get("validation_only") and synthetic_fixture_detected
            else "upstream_run_metadata" if metadata.get("validation_only")
            else "synthetic_marker_detected" if synthetic_fixture_detected
            else None
        ),
        "warning": (
            "Synthetic/demo fixture evidence detected from event identifiers, reason text, or validation zone. Treat this score as software-validation output only, not as a real-world threat assessment."
            if synthetic_fixture_detected and not metadata.get("validation_only")
            else metadata.get("warning") if validation_only and metadata.get("warning")
            else "Software-validation result only. Do not use this score for real-world threat decisions."
            if validation_only
            else "This is an automated rule-based assessment, not ground truth; human verification is required."
        ),
        "limitations": [
            "The existing fusion score was read and explained; this module did not change or retrain its rules.",
            "Weights are manually specified heuristic weights, not learned probabilities.",
            "Excluded events did not contribute to the reported score; see each exclusion reason.",
            *(["Marked validation-only because demo/synthetic fixture markers were found even though upstream metadata did not flag the score."] if synthetic_fixture_detected and not metadata.get("validation_only") else []),
        ],
    }
