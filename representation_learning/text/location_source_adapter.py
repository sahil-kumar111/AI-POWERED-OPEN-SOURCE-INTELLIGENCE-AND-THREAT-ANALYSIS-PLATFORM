"""Adapter for the existing `threat_detector.analysis_engine.analyze_threats` results.

This module is deliberately a mapper only: it performs no network calls, geocoding,
language detection, deduplication, truncation, or embedding.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

from .contracts import SCHEMA_VERSION, MAX_TEXT_CHARS, TextEvidenceItem, text_sha256
from .errors import InvalidInputError


def _required_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{field} must be supplied as an ISO-8601 timestamp with UTC offset")
    return value


def _stable_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def location_result_to_text_evidence(
    result: Mapping[str, Any],
    *,
    retrieved_at: str,
    language: str = "und",
    source_name: Optional[str] = None,
) -> TextEvidenceItem:
    """Convert one location-threat result to a validated TextEvidenceItem.

    `retrieved_at` must be provided by the caller; this adapter never invents a clock time.
    The original text contains only the article title and description. Threat labels and
    geographic metadata remain metadata and are not appended to the embedding text.
    Separate result rows are preserved, including rows sharing a URL but representing
    different categories or incident locations.
    """
    if not isinstance(result, Mapping):
        raise InvalidInputError("location result must be a mapping")

    retrieved_at = _required_timestamp(retrieved_at, "retrieved_at")
    if not isinstance(language, str) or not language.strip():
        raise InvalidInputError("language must be a non-blank string; use 'und' when unknown")

    title = result.get("title")
    description = result.get("description")
    if title is not None and not isinstance(title, str):
        raise InvalidInputError("title must be a string or null")
    if description is not None and not isinstance(description, str):
        raise InvalidInputError("description must be a string or null")

    parts = [part for part in (title, description) if isinstance(part, str) and part.strip()]
    original_text = "\n".join(parts)
    if not original_text.strip():
        raise InvalidInputError("location result has no article title or description")
    if len(original_text) > MAX_TEXT_CHARS:
        raise InvalidInputError(
            f"article text has {len(original_text)} characters (maximum {MAX_TEXT_CHARS}); "
            "it is rejected, never truncated"
        )

    publisher = source_name if source_name is not None else result.get("source")
    if not isinstance(publisher, str) or not publisher.strip():
        raise InvalidInputError("source_name is required from the article's publisher metadata")

    url = result.get("url")
    if url is not None and not isinstance(url, str):
        raise InvalidInputError("url must be a string or null")
    if isinstance(url, str):
        url = url.strip() or None

    published_at = result.get("publishedAt")
    if published_at is not None and (not isinstance(published_at, str) or not published_at.strip()):
        raise InvalidInputError("publishedAt must be an ISO-8601 timestamp string or null")

    category = result.get("category")
    severity = result.get("severity")
    incident_location = result.get("incident_location")
    lat = result.get("incident_latitude")
    lon = result.get("incident_longitude")

    # Identity includes incident context so different incidents/categories on the same
    # article URL are not accidentally merged. The text hash is part of the documented ID.
    source_identity = {
        "source_url": url,
        "source_name": publisher,
        "published_at": published_at,
        "title": title,
        "category": category,
        "incident_location": incident_location,
        "incident_latitude": lat,
        "incident_longitude": lon,
    }
    incident_key = _stable_json_hash(source_identity)
    source_id = (url + "#incident=" + incident_key) if url else ("newsitem:" + incident_key)
    text_hash = text_sha256(original_text)
    evidence_suffix = hashlib.sha256(
        (source_id + "|" + text_hash).encode("utf-8")
    ).hexdigest()[:16]
    evidence_id = "te_" + evidence_suffix

    source = {
        "source_name": publisher,
        "source_url": url,
        "source_item_id": source_id,
        "retrieved_at": retrieved_at,
    }
    if published_at is not None:
        source["published_at"] = published_at

    nlp_context = {}
    if isinstance(category, str) and category.strip():
        nlp_context["threat_category"] = category
    if isinstance(severity, str) and severity.strip():
        nlp_context["severity_label"] = severity
    if isinstance(result.get("geographic_classification"), str):
        nlp_context["geographic_classification"] = result["geographic_classification"]
    if isinstance(result.get("distance_km"), (int, float)) and not isinstance(result.get("distance_km"), bool):
        nlp_context["distance_km"] = result["distance_km"]

    location = {}
    if isinstance(incident_location, str):
        location["place_label"] = incident_location
    if isinstance(lat, (int, float)) and not isinstance(lat, bool):
        location["lat"] = lat
    if isinstance(lon, (int, float)) and not isinstance(lon, bool):
        location["lon"] = lon

    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "source_kind": "article",
        "source": source,
        "original_text": original_text,
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
    if location:
        payload["location"] = location

    return TextEvidenceItem.from_dict(payload)


def adapt_location_results(
    results: Any,
    *,
    retrieved_at: str,
    language: str = "und",
    source_name: Optional[str] = None,
) -> list[TextEvidenceItem]:
    """Map result rows independently; intentionally does not deduplicate."""
    if isinstance(results, (str, bytes)) or not isinstance(results, (list, tuple)):
        raise InvalidInputError("results must be a list or tuple of location result mappings")
    return [
        location_result_to_text_evidence(
            result,
            retrieved_at=retrieved_at,
            language=language,
            source_name=source_name,
        )
        for result in results
    ]
