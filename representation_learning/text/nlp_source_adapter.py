"""Adapter for the existing NLP Intelligence `process_record` output.

The caller must provide the original incident description and retrieval timestamp.
Generated `analysis_text` is used only to verify that the supplied description belongs
to the NLP record; it is never embedded.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

from .contracts import MAX_TEXT_CHARS, SCHEMA_VERSION, TextEvidenceItem, text_sha256
from .errors import InvalidInputError


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def nlp_record_to_text_evidence(
    record: Mapping[str, Any],
    *,
    original_description: str,
    retrieved_at: str,
    language: str = "und",
) -> TextEvidenceItem:
    """Map one NLP record without embedding generated templates or detector boilerplate."""
    if not isinstance(record, Mapping):
        raise InvalidInputError("NLP record must be a mapping")
    if not isinstance(original_description, str) or not original_description.strip():
        raise InvalidInputError("original_description must be supplied and non-blank")
    if len(original_description) > MAX_TEXT_CHARS:
        raise InvalidInputError(
            f"original_description exceeds {MAX_TEXT_CHARS} characters; it is rejected, never truncated"
        )
    if not isinstance(retrieved_at, str) or not retrieved_at.strip():
        raise InvalidInputError("retrieved_at must be supplied as an ISO-8601 timestamp with UTC offset")
    if not isinstance(language, str) or not language.strip():
        raise InvalidInputError("language must be non-blank; use 'und' when unknown")

    analysis_text = record.get("analysis_text")
    if not isinstance(analysis_text, str) or original_description not in analysis_text:
        raise InvalidInputError(
            "original_description could not be verified in NLP analysis_text; refusing to embed"
        )

    source_meta = record.get("source")
    if not isinstance(source_meta, Mapping):
        source_meta = {}
    record_id = record.get("record_id")
    source_type = source_meta.get("type")
    source_report = source_meta.get("source_report")
    input_video = source_meta.get("input_video")
    source_id = record_id if isinstance(record_id, str) and record_id.strip() else (
        "nlp-record:" + _stable_hash({
            "source_type": source_type,
            "source_report": source_report,
            "input_video": input_video,
            "description_sha256": text_sha256(original_description),
        })
    )
    text_hash = text_sha256(original_description)
    evidence_id = "te_" + hashlib.sha256(
        (source_id + "|" + text_hash).encode("utf-8")
    ).hexdigest()[:16]

    source = {
        "source_name": source_type if isinstance(source_type, str) and source_type.strip()
                       else "nlp_intelligence",
        "source_url": None,
        "source_item_id": source_id,
        "retrieved_at": retrieved_at,
    }

    nlp_context = {}
    for key in ("entities", "threats", "events", "model_metadata"):
        if key in record:
            nlp_context[key] = record[key]
    if source_report is not None:
        nlp_context["source_report"] = source_report

    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "source_kind": "event_description",
        "source": source,
        "original_text": original_description,
        "original_text_sha256": text_hash,
        "language": language,
        "truncated": False,
        "sensitivity": {
            "may_contain_personal_data": True,
            "retain_text": False,
        },
    }
    if nlp_context:
        payload["nlp_context"] = nlp_context
    return TextEvidenceItem.from_dict(payload)
