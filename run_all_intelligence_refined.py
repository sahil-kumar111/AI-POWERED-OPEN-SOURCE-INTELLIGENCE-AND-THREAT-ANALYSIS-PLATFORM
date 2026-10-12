#!/usr/bin/env python3
"""Run video detection + Gemini visual description + evidence-aware NLP fusion.

Usage:
    export GEMINI_API_KEY="..."
    python3 run_all_intelligence.py
    python3 run_all_intelligence.py input/example.mp4

The existing threat_video_pipeline.py is invoked unchanged. This runner creates
its own final combined report and does not overwrite the original detector JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from google import genai
except ImportError:
    genai = None

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEO_PIPELINE = PROJECT_ROOT / "threat_video_pipeline.py"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

THREAT_TERMS = {
    "weapon": [
        "weapon", "gun", "firearm", "pistol", "rifle", "shotgun",
        "handgun", "knife", "sword", "armed", "holding a gun",
        "holding a weapon", "weapon-like", "object resembling a weapon",
    ],
    "fire": ["fire", "flames", "burning", "blaze", "on fire"],
    "smoke": ["smoke", "smoke plume", "billowing smoke", "smoky"],
    "explosion": ["explosion", "exploded", "blast", "detonation"],
    "physical_violence": [
        "assault", "attacks", "attacked", "fighting", "punches",
        "stabbing", "shooting", "strikes a person", "physical altercation",
    ],
    "suspicious_activity": [
        "breaks into", "forced entry", "vandalism", "stealing",
        "attempts to enter", "trespassing",
    ],
}
NEGATION_TERMS = [
    "no ", "not ", "without ", "does not show", "doesn't show",
    "cannot see", "can't see", "no evidence of", "not visible",
    "unlikely to be", "not appear to be",
]
HYPOTHETICAL_TERMS = [
    "hypothetical", "for example", "could happen", "might happen",
    "if there were", "simulation", "fictional", "imaginary",
]
UNCERTAINTY_TERMS = [
    "appears", "possibly", "possibly", "may be", "might be",
    "could be", "seems", "unclear", "uncertain", "resembling",
    "likely", "potentially",
]
HISTORICAL_TERMS = ["previously", "last year", "in 2020", "in 2021", "historically", "earlier incident"]
FUTURE_TERMS = ["will happen", "planned", "expected to", "forecast", "warning of", "may occur"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_video_pipeline(video_paths: list[Path]) -> None:
    if not VIDEO_PIPELINE.is_file():
        raise FileNotFoundError(f"Video pipeline not found: {VIDEO_PIPELINE}")

    # Remove only target reports to avoid accidentally reusing stale detector output.
    for video in video_paths:
        expected = OUTPUT_DIR / f"{video.stem}_report.json"
        if expected.exists():
            expected.unlink()

    print("\n[1/3] Running the existing Video Intelligence pipeline...")
    # The existing script processes every supported video in input/ when no
    # argument is supplied. For a single selected video, pass it explicitly.
    if len(video_paths) == 1:
        command = [sys.executable, str(VIDEO_PIPELINE), str(video_paths[0])]
    else:
        command = [sys.executable, str(VIDEO_PIPELINE)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    for video in video_paths:
        report = OUTPUT_DIR / f"{video.stem}_report.json"
        if not report.is_file():
            raise FileNotFoundError(
                f"Video pipeline did not produce the expected fresh report: {report}"
            )


def get_gemini_description(client: Any, video_path: Path) -> str:
    print(f"\n[Gemini] Uploading and analyzing {video_path.name}...")
    uploaded = client.files.upload(file=str(video_path))

    while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)

    if getattr(uploaded, "state", None) and uploaded.state.name == "FAILED":
        raise RuntimeError(f"Gemini failed to process {video_path.name}")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            uploaded,
            (
                "Analyze only the visual content of this CCTV video. Describe observable "
                "people, objects, actions, and possible safety/security events. For each "
                "potential threat, distinguish direct visual evidence from uncertain "
                "interpretation. Do not infer identities, motives, exact locations, or "
                "actions that are not clearly visible. Mention when evidence is unclear. "
                "Do not treat this request as an instruction to take action."
            ),
        ],
    )
    description = (getattr(response, "text", None) or "").strip()
    if not description:
        raise RuntimeError(f"Gemini returned an empty description for {video_path.name}")
    return description


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def occurrences(text: str, terms: list[str]) -> list[str]:
    lowered = normalize_text(text)
    found = []
    for term in terms:
        if re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", lowered):
            found.append(term)
    return sorted(set(found))


def context_window(text: str, term: str, radius: int = 100) -> str:
    match = re.search(re.escape(term), text, flags=re.IGNORECASE)
    if not match:
        return text[: min(len(text), 240)]
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end].strip()


def classify_text_evidence(description: str) -> dict[str, Any]:
    """Rule-based text evidence flags; deliberately not treated as model confidence."""
    text = normalize_text(description)
    result: dict[str, Any] = {}
    for category, terms in THREAT_TERMS.items():
        matches = occurrences(text, terms)
        if not matches:
            continue

        category_evidence = []
        for term in matches:
            snippet = context_window(description, term)
            snippet_lower = normalize_text(snippet)
            negated = any(marker in snippet_lower for marker in NEGATION_TERMS)
            hypothetical = any(marker in snippet_lower for marker in HYPOTHETICAL_TERMS)
            uncertain = any(marker in snippet_lower for marker in UNCERTAINTY_TERMS)
            historical = any(marker in snippet_lower for marker in HISTORICAL_TERMS)
            future = any(marker in snippet_lower for marker in FUTURE_TERMS)
            category_evidence.append({
                "matched_term": term,
                "evidence": snippet,
                "negated_or_denied": negated,
                "hypothetical_context": hypothetical,
                "uncertain_language": uncertain,
                "historical_context_possible": historical,
                "future_or_warning_context_possible": future,
            })
        result[category] = category_evidence
    return result


def detector_evidence(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    counts = report.get("detection_counts") or {}
    frames = report.get("detection_frames") or {}
    ratios = report.get("detection_ratios") or {}
    average_confidence = report.get("average_confidence") or {}
    max_confidence = report.get("max_confidence") or {}
    detected = {}
    for category in ("weapon", "fire", "smoke", "person"):
        count = counts.get(category, 0)
        detected[category] = {
            "detected": bool(count),
            "detection_count": count,
            "frames_with_detection": frames.get(category, 0),
            "detection_ratio": ratios.get(category),
            "average_confidence": average_confidence.get(category),
            "max_confidence": max_confidence.get(category),
            "source": "video_detector",
        }
    return detected


def reconcile_threats(
    report: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    detector = detector_evidence(report)
    text_evidence = classify_text_evidence(description)
    categories = sorted(set(detector) | set(text_evidence) | set(THREAT_TERMS))
    decisions = {}

    for category in categories:
        det = detector.get(category, {
            "detected": False, "detection_count": 0,
            "frames_with_detection": 0, "detection_ratio": None,
            "average_confidence": None, "max_confidence": None,
            "source": "video_detector",
        })
        entries = text_evidence.get(category, [])
        positive_text_entries = [
            entry for entry in entries
            if not entry["negated_or_denied"] and not entry["hypothetical_context"]
        ]
        uncertain_entries = [
            entry for entry in positive_text_entries if entry["uncertain_language"]
        ]
        clear_entries = [
            entry for entry in positive_text_entries if not entry["uncertain_language"]
        ]
        text_support = bool(positive_text_entries)
        text_clear_support = bool(clear_entries)
        detector_support = bool(det.get("detected"))

        if detector_support and text_clear_support:
            status = "corroborated"
            explanation = "Video detector and Gemini description both support this category."
        elif detector_support and text_support:
            status = "detector_supported_text_uncertain"
            explanation = "Detector supports this category; Gemini mentions it with uncertainty."
        elif detector_support:
            status = "detector_only"
            explanation = "Video detector supports this category; Gemini text does not clearly corroborate it."
        elif text_clear_support:
            status = "description_only"
            explanation = "Gemini description supports this category, but the video detector did not detect it."
        elif uncertain_entries:
            status = "uncertain_text_only"
            explanation = "Only uncertain textual evidence was found; this is not a confirmed threat."
        else:
            status = "not_supported"
            explanation = "Neither source provides positive evidence for this category."

        # "confirmed" is deliberately not used: the system only has model evidence,
        # not independent human verification.
        decisions[category] = {
            "status": status,
            "detector_support": detector_support,
            "gemini_text_support": text_support,
            "gemini_clear_text_support": text_clear_support,
            "detector_evidence": det,
            "text_evidence": entries,
            "explanation": explanation,
            "interpretation_note": (
                "Corroborated means two model outputs agree; it is not proof that "
                "the event is real. Review the video before operational use."
            ),
        }

    return decisions


def extract_entities(text: str) -> tuple[list[dict[str, Any]], str]:
    """Use spaCy NER when installed; otherwise return no fabricated entities."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:100_000])
        entities = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "DATE", "TIME", "EVENT", "NORP"}:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "gemini_description",
                    "verified": False,
                })
        return entities, "spaCy en_core_web_sm"
    except (ImportError, OSError):
        return [], "unavailable: install spacy and en_core_web_sm for NER"


def sentiment_analysis(text: str) -> dict[str, Any]:
    if not text.strip():
        return {"label": "not_applicable", "compound_score": None, "method": "VADER", "note": "No text supplied."}
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        scores = SentimentIntensityAnalyzer().polarity_scores(text[:100_000])
        compound = scores["compound"]
        label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
        return {
            "label": label,
            "compound_score": compound,
            "scores": scores,
            "method": "VADER",
            "note": "Sentiment is not threat severity and does not determine threat status.",
        }
    except ImportError:
        return {
            "label": "unavailable",
            "compound_score": None,
            "method": "VADER not installed",
            "note": "Install vaderSentiment to enable sentiment analysis.",
        }


def extract_keywords(text: str, limit: int = 20) -> list[str]:
    """Extract concise, normalized keywords without formatting artifacts."""
    if not text.strip():
        return []

    lowered = normalize_text(text)
    ranked: list[str] = []

    # Curated security terms have priority over generic words.
    for terms in THREAT_TERMS.values():
        for term in terms:
            if re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", lowered):
                ranked.append(term.lower())

    stop_phrases = {
        "the video", "the scene", "the area", "the image", "the camera",
        "the left", "the right", "the background", "the foreground",
        "the vehicle", "the object", "the person", "the people",
        "it appears", "it seems", "there is", "there are",
    }
    stop_words = {
        "the", "and", "that", "this", "with", "from", "there", "their",
        "they", "have", "has", "was", "were", "are", "for", "into",
        "while", "over", "near", "video", "scene", "appears", "shown",
        "visible", "partially", "clearly", "possibly", "likely", "seems",
        "could", "would", "should", "might", "about", "around", "which",
        "where", "when", "what", "then", "than", "also", "some", "more",
    }

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:100_000])

        for chunk in doc.noun_chunks:
            phrase = re.sub(r"\s+", " ", chunk.text.lower()).strip(" \t\r\n.,:;!?-'\"()[]{}")
            tokens = re.findall(r"[a-z][a-z'-]*", phrase)
            if (
                len(tokens) >= 2
                and len(phrase) >= 4
                and phrase not in stop_phrases
                and not all(token in stop_words for token in tokens)
                and not re.fullmatch(r"(?:#+|[-_=*]+)", phrase)
            ):
                ranked.append(phrase)

        for token in doc:
            if (
                token.pos_ in {"NOUN", "PROPN"}
                and not token.is_stop
                and token.is_alpha
                and len(token.lemma_) > 2
            ):
                lemma = token.lemma_.lower()
                if lemma not in stop_words:
                    ranked.append(lemma)
    except (ImportError, OSError):
        tokens = re.findall(r"\b[a-z][a-z'-]{2,}\b", lowered)
        ranked.extend(token for token in tokens if token not in stop_words)

    # Stable deduplication, with frequency used only to rank repeated candidates.
    counts = Counter(
        phrase for phrase in ranked
        if phrase and phrase not in stop_phrases
        and not re.search(r"[#*_]{2,}", phrase)
    )
    ordered = sorted(counts, key=lambda phrase: (-counts[phrase], -len(phrase.split()), phrase))
    return ordered[:limit]


def build_events(
    report: dict[str, Any],
    description: str,
    threat_decisions: dict[str, Any],
) -> list[dict[str, Any]]:
    events = []
    for category, decision in threat_decisions.items():
        if category == "person" or decision["status"] == "not_supported":
            continue
        if decision["status"] == "uncertain_text_only":
            confidence_label = "low_or_uncertain"
        elif decision["status"] == "corroborated":
            confidence_label = "cross_source_corroboration"
        elif decision["detector_support"]:
            confidence_label = "detector_evidence"
        else:
            confidence_label = "description_only"

        events.append({
            "event_type": category,
            "status": decision["status"],
            "confidence_label": confidence_label,
            "event_time": None,
            "location_mentions": [],
            "evidence": {
                "video_detector": decision["detector_evidence"],
                "gemini_text": decision["text_evidence"],
            },
            "record_id": report.get("video_name") or report.get("video") or report.get("record_id"),
            "is_human_verified": False,
        })

    return events


def analyze_combined(
    video_path: Path,
    report: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    threat_decisions = reconcile_threats(report, description)
    entities, entity_method = extract_entities(description)
    sentiment = sentiment_analysis(description)
    keywords = extract_keywords(description)

    detector_summary = detector_evidence(report)
    corroborated = [
        category for category, item in threat_decisions.items()
        if item["status"] == "corroborated"
    ]
    detector_only = [
        category for category, item in threat_decisions.items()
        if item["status"] == "detector_only"
    ]
    description_only = [
        category for category, item in threat_decisions.items()
        if item["status"] == "description_only"
    ]
    uncertain = [
        category for category, item in threat_decisions.items()
        if item["status"] == "uncertain_text_only"
    ]

    # A single overall decision should not discard disagreement. "Potential threat
    # evidence present" means at least one source supports a threat category.
    any_detector_threat = any(
        detector_summary.get(category, {}).get("detected", False)
        for category in ("weapon", "fire", "smoke")
    )
    any_text_threat = any(
        threat_decisions.get(category, {}).get("gemini_text_support", False)
        for category in ("weapon", "fire", "smoke", "explosion", "physical_violence", "suspicious_activity")
    )
    overall = (
        "threat_evidence_corroborated" if corroborated
        else "potential_threat_evidence_present" if any_detector_threat or any_text_threat
        else "no_threat_evidence_detected"
    )

    try:
        display_path = str(video_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(video_path.resolve())

    return {
        "schema_version": "2.0",
        "created_at_utc": utc_now(),
        "video": {
            "filename": video_path.name,
            "path": display_path,
        },
        "overall_assessment": {
            "status": overall,
            "decision_basis": "source-level evidence reconciliation",
            "corroborated_categories": corroborated,
            "detector_only_categories": detector_only,
            "description_only_categories": description_only,
            "uncertain_text_categories": uncertain,
            "note": (
                "This is automated evidence reconciliation, not a verified real-world "
                "finding. Agreement between models is corroboration, not ground truth. "
                "A detector's failure to detect an object does not prove its absence."
            ),
        },
        "video_intelligence": {
            "record_id": report.get("record_id", video_path.stem),
            "detector_summary": detector_summary,
            "raw_report": report,
        },
        "gemini_video_understanding": {
            "model": GEMINI_MODEL,
            "source": "video upload to Gemini API",
            "description": description,
            "normalized_description": re.sub(r"\\s+", " ", description).strip(),
            "text_derived_threat_evidence": classify_text_evidence(description),
        },
        "representation_learning": {
            "schema_version": "2.0",
            "text_fields": {
                "primary": re.sub(r"\\s+", " ", description).strip(),
                "secondary": None,
                "guidance": (
                    "Use the original Gemini description as the primary text input. "
                    "Do not embed detector metadata as if it were natural-language evidence."
                ),
            },
            "metadata_fields": {
                "detector_summary": detector_summary,
                "threat_status_by_category": {
                    category: item["status"]
                    for category, item in threat_decisions.items()
                },
                "keywords": keywords,
                "entities": entities,
            },
            "provenance": {
                "text_source": "Gemini-generated description",
                "visual_detection_source": "existing Video Intelligence pipeline",
                "human_verified": False,
            },
        },
        "nlp_analysis": {
            "threat_reconciliation": threat_decisions,
            "events": build_events(report, description, threat_decisions),
            "entities": entities,
            "entity_extraction_method": entity_method,
            "keywords": keywords,
            "sentiment": sentiment,
            "language": "en_assumed",
            "limitations": [
                "Gemini description is generated text and may be inaccurate.",
                "Threat phrase matching is a transparent rule-based baseline, not a trained classifier.",
                "spaCy entities are extracted from Gemini's text, not independently verified.",
                "No incident time or location is inferred unless explicitly available from the input report/text.",
                "No independent human verification is performed.",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video detection, Gemini video understanding, and evidence-aware NLP fusion.")
    parser.add_argument("video", nargs="?", help="Optional path to one video. Defaults to all supported videos in input/.")
    parser.add_argument("--skip-video-pipeline", action="store_true", help="Reuse existing report JSON files; do not rerun detector.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if genai is None:
        print("ERROR: google-genai is missing. Install it with: pip install google-genai", file=sys.stderr)
        return 1

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in this Bash session.", file=sys.stderr)
        return 1

    if args.video:
        candidate = Path(args.video)
        video_paths = [candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()]
    else:
        video_paths = sorted(
            path for path in INPUT_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ) if INPUT_DIR.is_dir() else []

    if not video_paths:
        print(f"ERROR: No supported videos found in {INPUT_DIR}", file=sys.stderr)
        return 1

    for video in video_paths:
        if not video.is_file():
            print(f"ERROR: Video file not found: {video}", file=sys.stderr)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_video_pipeline:
            run_video_pipeline(video_paths)

        client = genai.Client(api_key=api_key)
        for video_path in video_paths:
            report_path = OUTPUT_DIR / f"{video_path.stem}_report.json"
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"Missing report {report_path}. Run the detector first or use the correct video path."
                )
            report = load_json(report_path)
            if not isinstance(report, dict):
                raise ValueError(f"Expected a JSON object in {report_path}")

            print("\n[2/3] Generating Gemini video description...")
            description = get_gemini_description(client, video_path)
            description_path = OUTPUT_DIR / f"{video_path.stem}_video_description.json"
            save_json(description_path, {
                "video": video_path.name,
                "model": GEMINI_MODEL,
                "incident_description": description,
                "created_at_utc": utc_now(),
            })
            print(f"Description saved: {description_path.relative_to(PROJECT_ROOT)}")

            print("\n[3/3] Reconciling evidence and running NLP...")
            combined = analyze_combined(video_path, report, description)
            final_path = OUTPUT_DIR / f"{video_path.stem}_combined_intelligence.json"
            save_json(final_path, combined)

            print("\nFINAL ASSESSMENT")
            print("=" * 52)
            print("Video:", video_path.name)
            print("Overall:", combined["overall_assessment"]["status"])
            print("Corroborated:", combined["overall_assessment"]["corroborated_categories"] or "None")
            print("Detector only:", combined["overall_assessment"]["detector_only_categories"] or "None")
            print("Description only:", combined["overall_assessment"]["description_only_categories"] or "None")
            print("Keywords:", ", ".join(combined["nlp_analysis"]["keywords"]) or "None")
            print("Entities:", len(combined["nlp_analysis"]["entities"]))
            print("Final JSON:", final_path.relative_to(PROJECT_ROOT))
        print("\nPipeline completed.")
        return 0
    except subprocess.CalledProcessError as error:
        print(f"ERROR: Video detector exited with status {error.returncode}.", file=sys.stderr)
        return error.returncode or 1
    except Exception as error:
        print(f"ERROR: Combined pipeline failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
