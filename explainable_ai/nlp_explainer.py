from __future__ import annotations

from typing import Any
from .contracts import clip_text, read_json, safe_dict, safe_list


def explain_nlp_output(combined: dict[str, Any] | str) -> dict[str, Any]:
    """Explain the saved NLP/reconciliation fields without pretending SHAP ran.

    Current AEGIS NLP output uses transparent rules, entity extraction, and
    source-level reconciliation rather than a trained text classifier. Therefore
    explanations trace matched evidence, statuses, entities, and source provenance.
    """
    data = read_json(combined)
    if not isinstance(data, dict):
        raise ValueError("Combined intelligence input must be a JSON object")
    nlp = safe_dict(data.get("nlp_analysis"))
    assessment = safe_dict(data.get("overall_assessment"))
    gemini = safe_dict(data.get("gemini_video_understanding"))
    reconciliation = safe_dict(nlp.get("threat_reconciliation"))
    source_evidence = safe_dict(gemini.get("text_derived_threat_evidence"))

    categories = []
    keys = sorted(set(reconciliation) | set(source_evidence))
    for category in keys:
        rec = safe_dict(reconciliation.get(category))
        text_rows = safe_list(source_evidence.get(category))
        evidence_rows = []
        for row in text_rows[:10]:
            if not isinstance(row, dict):
                continue
            evidence_rows.append({
                "matched_term": row.get("matched_term"),
                "evidence_excerpt": clip_text(row.get("evidence"), 500),
                "negated_or_denied": row.get("negated_or_denied"),
                "hypothetical_context": row.get("hypothetical_context"),
                "uncertain_language": row.get("uncertain_language"),
                "historical_context_possible": row.get("historical_context_possible"),
                "future_or_warning_context_possible": row.get("future_or_warning_context_possible"),
            })
        detector = safe_dict(rec.get("detector_evidence"))
        status = rec.get("status") or ("text_evidence_only" if text_rows else "not_present_in_reconciliation")
        categories.append({
            "category": category,
            "reconciliation_status": status,
            "detector_support": rec.get("detector_support"),
            "text_support": rec.get("gemini_text_support"),
            "clear_text_support": rec.get("gemini_clear_text_support"),
            "detector_summary": {
                "detected": detector.get("detected"),
                "detection_count": detector.get("detection_count"),
                "frames_with_detection": detector.get("frames_with_detection"),
                "detection_ratio": detector.get("detection_ratio"),
                "average_confidence": detector.get("average_confidence"),
                "max_confidence": detector.get("max_confidence"),
                "source": detector.get("source"),
            } if detector else None,
            "matched_text_evidence": evidence_rows,
            "source_explanation": rec.get("explanation"),
            "interpretation_note": rec.get("interpretation_note"),
        })

    entities = []
    for ent in safe_list(nlp.get("entities"))[:100]:
        if isinstance(ent, dict):
            entities.append({
                "text": ent.get("text"), "type": ent.get("type"),
                "source": ent.get("source"), "verified": ent.get("verified", False),
                "evidence": clip_text(ent.get("evidence"), 300),
            })

    return {
        "explanation_type": "nlp_evidence_trace",
        "source_video": safe_dict(data.get("video")).get("filename"),
        "nlp_method": nlp.get("entity_extraction_method"),
        "threat_reconciliation_status": assessment.get("status"),
        "decision_basis": assessment.get("decision_basis"),
        "corroborated_categories": safe_list(assessment.get("corroborated_categories")),
        "detector_only_categories": safe_list(assessment.get("detector_only_categories")),
        "description_only_categories": safe_list(assessment.get("description_only_categories")),
        "uncertain_text_categories": safe_list(assessment.get("uncertain_text_categories")),
        "categories": categories,
        "entities": entities,
        "keywords": safe_list(nlp.get("keywords"))[:50],
        "sentiment": safe_dict(nlp.get("sentiment")),
        "limitations": safe_list(nlp.get("limitations")) + [
            "This module traces the existing NLP pipeline output. It does not perform SHAP attribution because no trained text-classifier prediction interface is present in this schema.",
            "Generated video descriptions and extracted entities are unverified until reviewed by a human.",
        ],
    }
