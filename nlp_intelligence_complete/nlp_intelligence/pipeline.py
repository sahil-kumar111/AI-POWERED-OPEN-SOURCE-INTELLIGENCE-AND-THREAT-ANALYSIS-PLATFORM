from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TEXT_LIMIT = 100_000

THREAT_TAXONOMY = {
    "weapon": {"weapon", "gun", "pistol", "rifle", "shotgun", "knife", "firearm", "armed"},
    "fire": {"fire", "burning", "flames", "arson"},
    "smoke": {"smoke", "smoke plume"},
    "explosion": {"explosion", "blast", "detonation", "bomb"},
    "crowd_disturbance": {"riot", "stampede", "mob", "crowd disturbance", "public disorder"},
    "security_breach": {"intrusion", "break-in", "breach", "trespassing", "unauthorized entry"},
    "cybersecurity": {"malware", "ransomware", "phishing", "cyberattack", "data breach"},
    "natural_hazard": {"flood", "earthquake", "cyclone", "storm", "landslide"},
}

NEGATION_TERMS = {"no", "not", "never", "without", "none", "neither", "deny", "denies", "denied"}
HISTORICAL_TERMS = {"previously", "earlier", "last year", "last month", "historical", "in 2020", "in 2021", "in 2022", "in 2023", "in 2024", "in 2025"}
FUTURE_TERMS = {"may", "might", "could", "possible", "potential", "expected", "warning", "forecast", "will", "planned"}
UNCERTAINTY_TERMS = {"alleged", "unconfirmed", "unverified", "reportedly", "suspected", "claims", "claim", "rumor", "rumour"}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "for", "from", "with", "by", "as", "this", "that", "it",
    "video", "frame", "frames", "detected", "detection", "confidence", "average",
    "category", "true", "false", "yes", "no", "person", "people", "report", "reported",
    "model", "analysis", "analyzed", "across", "video detector"
}

def _read_json(report: dict[str, Any] | str | Path) -> tuple[dict[str, Any], str | None]:
    source_path = str(report) if isinstance(report, (str, Path)) else None
    if isinstance(report, dict):
        data = report
    else:
        with open(report, "r", encoding="utf-8") as file:
            data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Input video report must be a JSON object.")
    return data, source_path

def _safe_string(value: Any, limit: int = TEXT_LIMIT) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return str(value)[:limit]

def _validate_report(report: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(report, dict):
        return ["Report must be a JSON object."]
    for key in ("detection_counts", "detection_ratios", "average_confidence"):
        if key in report and not isinstance(report[key], dict):
            errors.append(f"{key} must be a JSON object when supplied.")
    if "threat_events" in report and not isinstance(report["threat_events"], list):
        errors.append("threat_events must be a JSON array when supplied.")
    return errors

def build_analysis_text(report: dict[str, Any], incident_description: str = "") -> str:
    """Create transparent text context from the report; label it as detector output."""
    pieces = []
    description = _safe_string(incident_description).strip()
    if description:
        pieces.append("User-provided incident description: " + description)

    video_path = _safe_string(report.get("input_video"))
    if video_path:
        pieces.append("Analyzed video file: " + Path(video_path).name + ".")

    counts = report.get("detection_counts", {})
    ratios = report.get("detection_ratios", {})
    scores = report.get("average_confidence", {})
    if isinstance(counts, dict):
        for category in ("weapon", "fire", "smoke", "person"):
            count = counts.get(category, 0)
            ratio = ratios.get(category) if isinstance(ratios, dict) else None
            score = scores.get(category) if isinstance(scores, dict) else None
            pieces.append(
                f"Video model output: {category} detection count={count}; "
                f"frame ratio={ratio}; mean confidence={score}."
            )
    return " ".join(pieces)[:TEXT_LIMIT]

def _load_spacy():
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError(
            "spaCy is missing. Install dependencies with: pip install -r requirements.txt "
            "then run: python -m spacy download en_core_web_sm"
        ) from error
    try:
        return spacy.load("en_core_web_sm")
    except OSError as error:
        raise RuntimeError(
            "spaCy English model is missing. Run: python -m spacy download en_core_web_sm"
        ) from error

def extract_entities(text: str, nlp=None) -> list[dict[str, Any]]:
    """Extract PERSON, ORG, GPE, LOC, FAC, DATE, TIME, EVENT entities from real text."""
    text = _safe_string(text).strip()
    if not text:
        return []
    nlp = nlp or _load_spacy()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "type": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "confidence": None,
            "evidence": text[max(0, ent.start_char - 60):min(len(text), ent.end_char + 60)],
            "method": "spaCy_en_core_web_sm",
        })
    return entities

def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]

def _classify_text_threats(text: str) -> list[dict[str, Any]]:
    """Transparent lexical candidates; scores are not probabilities."""
    results = []
    for sentence in _sentences(text):
        lowered = sentence.lower()
        for category, vocabulary in THREAT_TAXONOMY.items():
            matches = [term for term in sorted(vocabulary, key=len, reverse=True)
                       if re.search(r"\b" + re.escape(term) + r"\b", lowered)]
            if not matches:
                continue

            words = set(re.findall(r"\b[\w'-]+\b", lowered))
            negated = any(term in words for term in NEGATION_TERMS)
            historical = any(term in lowered for term in HISTORICAL_TERMS)
            future = any(term in words or term in lowered for term in FUTURE_TERMS)
            uncertain = any(term in words for term in UNCERTAINTY_TERMS)

            if negated:
                status = "negated_or_denied"
            elif historical:
                status = "historical_reference"
            elif future:
                status = "potential_future_warning"
            elif uncertain:
                status = "unverified_or_alleged"
            else:
                status = "candidate_reported_mention"

            results.append({
                "category": category,
                "status": status,
                "confidence": None,
                "evidence": sentence,
                "matched_terms": matches,
                "method": "keyword_rules",
                "requires_context_review": True,
            })
    return results

def extract_threats(report: dict[str, Any], text: str) -> list[dict[str, Any]]:
    """Preserve video detector evidence, then append text-derived candidates."""
    results = []
    counts = report.get("detection_counts", {})
    ratios = report.get("detection_ratios", {})
    confidences = report.get("average_confidence", {})
    counts = counts if isinstance(counts, dict) else {}
    ratios = ratios if isinstance(ratios, dict) else {}
    confidences = confidences if isinstance(confidences, dict) else {}

    for category in ("weapon", "fire", "smoke"):
        count = counts.get(category, 0)
        if isinstance(count, (int, float)) and count > 0:
            results.append({
                "category": category,
                "status": "detected_by_video_model",
                "detected": True,
                "confidence": confidences.get(category),
                "detection_count": count,
                "detection_ratio": ratios.get(category),
                "evidence": (
                    f"Video model reported {count} {category} detections; "
                    f"frame ratio={ratios.get(category)}."
                ),
                "method": "video_report",
                "requires_context_review": True,
            })

    # Text-derived candidates are only meaningful when the caller provides a
    # genuine description/transcript, not the template generated from detections.
    candidates = _classify_text_threats(text)
    existing = {(item["category"], item["status"]) for item in results}
    for item in candidates:
        key = (item["category"], item["status"])
        if key not in existing:
            results.append(item)
            existing.add(key)
    return results

def extract_events(report: dict[str, Any], record_id: str) -> list[dict[str, Any]]:
    """Group adjacent frames with identical threat sets into event intervals."""
    raw_events = report.get("threat_events", [])
    if not isinstance(raw_events, list):
        return []
    rows = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        frame = item.get("frame")
        categories = item.get("threats", [])
        if isinstance(frame, int) and isinstance(categories, list):
            rows.append((frame, sorted({str(x) for x in categories})))
    rows.sort(key=lambda row: row[0])

    fps_value = report.get("fps")
    fps = float(fps_value) if isinstance(fps_value, (int, float)) and fps_value > 0 else None
    groups = []
    for frame, categories in rows:
        if groups and groups[-1]["threat_categories"] == categories and frame <= groups[-1]["end_frame"] + 1:
            groups[-1]["end_frame"] = frame
        else:
            groups.append({
                "start_frame": frame,
                "end_frame": frame,
                "threat_categories": categories,
            })

    output = []
    for group in groups:
        start_frame, end_frame = group["start_frame"], group["end_frame"]
        output.append({
            "event_id": f"{record_id}-event-{len(output) + 1}",
            "event_type": "video_detected_threat",
            "threat_categories": group["threat_categories"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_sec": round(start_frame / fps, 2) if fps else None,
            "end_time_sec": round(end_frame / fps, 2) if fps else None,
            "event_time": None,
            "location_mentions": [],
            "evidence": "Video report threat categories: " + ", ".join(group["threat_categories"]),
            "record_id": record_id,
            "confidence": None,
            "verification_status": "not_independently_verified",
        })
    return output

def analyze_sentiment(text: str) -> dict[str, Any]:
    """Run VADER only on caller-supplied natural-language text."""
    if not _safe_string(text).strip():
        return {
            "label": "not_applicable",
            "score": None,
            "compound": None,
            "model": None,
            "reason": "No natural-language description was supplied.",
        }
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return {
            "label": "unavailable",
            "score": None,
            "compound": None,
            "model": "VADER",
            "reason": "vaderSentiment is not installed. Run: pip install vaderSentiment",
        }

    scores = SentimentIntensityAnalyzer().polarity_scores(text)
    compound = scores["compound"]
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    return {
        "label": label,
        "score": round(abs(compound), 4),
        "compound": compound,
        "component_scores": scores,
        "model": "VADER",
        "reason": "Sentiment is a text-tone signal, not threat severity.",
    }

def extract_keywords(text: str, limit: int = 15) -> list[str]:
    """Combine threat vocabulary matches and frequency-ranked content words."""
    text = _safe_string(text)
    lowered = text.lower()
    candidates = []
    for vocabulary in THREAT_TAXONOMY.values():
        for term in vocabulary:
            if re.search(r"\b" + re.escape(term) + r"\b", lowered):
                candidates.append(term)
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", lowered)
    counts = Counter(token for token in tokens if token not in STOPWORDS)
    candidates.extend(token for token, _ in counts.most_common(limit * 2))
    result = []
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
        if len(result) >= limit:
            break
    return result

def process_record(
    report: dict[str, Any] | str | Path,
    *,
    incident_description: str = "",
    record_id: str | None = None,
    use_ner: bool = True,
) -> dict[str, Any]:
    data, source_path = _read_json(report)
    errors = _validate_report(data)
    if errors:
        raise ValueError("; ".join(errors))

    default_id = Path(_safe_string(data.get("input_video") or "video_report")).stem or "video-report"
    record_id = str(record_id or data.get("record_id") or default_id)
    description = _safe_string(incident_description).strip()
    analysis_text = build_analysis_text(data, description)

    # NER must run on real text, not on templated numeric detector summaries.
    entities = extract_entities(description) if use_ner and description else []
    threats = extract_threats(data, description)
    events = extract_events(data, record_id)
    sentiment = analyze_sentiment(description)
    keywords = extract_keywords((description + " " + " ".join(
        item["category"] for item in threats
    )).strip())

    return {
        "record_id": record_id,
        "source": {
            "type": "video_intelligence_report",
            "source_report": source_path,
            "input_video": data.get("input_video"),
            "output_video": data.get("output_video"),
        },
        "input_type": "video_detection_report",
        "analysis_text": analysis_text,
        "entities": entities,
        "threats": threats,
        "events": events,
        "sentiment": sentiment,
        "keywords": keywords,
        "language": "en" if description else "und",
        "model_metadata": {
            "entity_extraction": "spaCy en_core_web_sm" if use_ner and description else "not_run_no_natural_language_text",
            "threat_extraction": "video_report_fields_plus_contextual_keyword_rules",
            "event_detection": "contiguous_frame_grouping",
            "sentiment": "VADER" if description else "not_run_no_natural_language_text",
            "keyword_extraction": "threat_lexicon_plus_frequency",
            "limitations": [
                "Video detector predictions are evidence, not independently verified facts.",
                "Person detection does not identify a person by name.",
                "No geographic location is inferred from detection JSON.",
                "NER and sentiment require genuine natural-language context.",
                "Rule-based text threat candidates require contextual review.",
                "VADER is optimized for English social-media-like text and may perform poorly on formal reports."
            ],
        },
    }

def process_batch(
    reports: Iterable[dict[str, Any] | str | Path],
    *,
    descriptions: list[str] | None = None,
    use_ner: bool = True,
) -> list[dict[str, Any]]:
    report_list = list(reports)
    if descriptions is not None and len(descriptions) != len(report_list):
        raise ValueError("descriptions must have the same length as reports.")
    results = []
    for index, report in enumerate(report_list):
        description = descriptions[index] if descriptions is not None else ""
        try:
            results.append(process_record(report, incident_description=description, use_ner=use_ner))
        except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as error:
            results.append({
                "record_id": f"invalid-record-{index}",
                "status": "failed",
                "error": str(error),
            })
    return results
