import json
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parent
out = root / "representation_learning" / "outputs"

evidence_path = out / "aegis_normalized_video_evidence.json"
retrieval_path = out / "aegis_category_retrieval.json"

for path in (evidence_path, retrieval_path):
    if not path.exists():
        raise SystemExit(f"Required file missing: {path}")

evidence_data = json.loads(evidence_path.read_text(encoding="utf-8"))
retrieval_data = json.loads(retrieval_path.read_text(encoding="utf-8"))

records = evidence_data["records"]
by_id = {str(r["evidence_id"]): r for r in records}

if len(by_id) != len(records):
    raise SystemExit("Duplicate evidence IDs detected.")

def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def duration_seconds(record):
    first = record.get("first_seen_sec")
    last = record.get("last_seen_sec")
    if number(first) and number(last) and last >= first:
        return round(last - first, 3)
    return None

def quality_flags(record):
    flags = []
    confidence = record.get("confidence")
    frames = record.get("frames_detected")
    duration = duration_seconds(record)

    if confidence is None:
        flags.append("CONFIDENCE_UNAVAILABLE")
    elif confidence < 0.30:
        flags.append("LOW_CONFIDENCE_REVIEW")
    elif confidence < 0.50:
        flags.append("MODERATE_CONFIDENCE_REVIEW")

    if number(frames) and frames <= 3:
        flags.append("VERY_FEW_DETECTED_FRAMES")

    if duration is not None and duration <= 1.0:
        flags.append("SHORT_DETECTION_INTERVAL")

    if record.get("category") == "person":
        peak = record.get("peak_concurrent_persons")
        average = record.get("avg_concurrent_persons")
        if not number(peak) or not number(average):
            flags.append("PERSON_COUNT_INCOMPLETE")

    return flags

report = {
    "report_type": "AEGIS evidence-quality audit",
    "records_total": len(records),
    "disclaimer": (
        "Review flags are triage signals, not proof of false positives. "
        "Detector confidence is not a calibrated probability. "
        "Semantic similarity is not detection accuracy."
    ),
    "category_summary": {},
    "retrieval_summary": {},
    "evidence": [],
}

for category in ("weapon", "fire", "smoke", "person"):
    subset = [r for r in records if r.get("category") == category]
    positives = [r for r in subset if r.get("detected") is True]
    negatives = [r for r in subset if r.get("detected") is False]

    confidence_values = [
        r["confidence"] for r in positives
        if number(r.get("confidence"))
    ]

    report["category_summary"][category] = {
        "total": len(subset),
        "detected": len(positives),
        "not_detected": len(negatives),
        "positive_confidence_available": len(confidence_values),
        "mean_positive_confidence": (
            round(sum(confidence_values) / len(confidence_values), 4)
            if confidence_values else None
        ),
    }

for record in records:
    report["evidence"].append({
        "evidence_id": record["evidence_id"],
        "source_report": record.get("source_report"),
        "category": record.get("category"),
        "detected": record.get("detected"),
        "confidence": record.get("confidence"),
        "frames_detected": record.get("frames_detected"),
        "duration_seconds": duration_seconds(record),
        "peak_concurrent_persons": record.get("peak_concurrent_persons"),
        "avg_concurrent_persons": record.get("avg_concurrent_persons"),
        "review_flags": quality_flags(record),
    })

# Support either a category-keyed dictionary or a list of retrieval groups.
groups = retrieval_data
if isinstance(retrieval_data, dict) and "results" in retrieval_data:
    groups = retrieval_data["results"]

if isinstance(groups, dict):
    iterator = groups.items()
elif isinstance(groups, list):
    iterator = [
        (str(group.get("category", group.get("query", f"group_{i}"))), group)
        for i, group in enumerate(groups)
    ]
else:
    iterator = []

for category, group in iterator:
    items = group if isinstance(group, list) else (
        group.get("results", group.get("records", []))
        if isinstance(group, dict) else []
    )
    summarized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id", item.get("id", "")))
        source = by_id.get(evidence_id, {})
        summarized.append({
            "evidence_id": evidence_id or None,
            "similarity": item.get("similarity", item.get("score")),
            "source_category": source.get("category"),
            "detected": source.get("detected"),
            "confidence": source.get("confidence"),
            "review_flags": quality_flags(source) if source else ["EVIDENCE_ID_UNMATCHED"],
        })

    report["retrieval_summary"][str(category)] = summarized

result_path = out / "aegis_evidence_quality_audit.json"
result_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("=" * 64)
print("AEGIS EVIDENCE-QUALITY AUDIT")
print("=" * 64)
print("Total evidence records:", len(records))
print("\nCATEGORY SUMMARY")
for category, summary in report["category_summary"].items():
    print(
        f"{category.upper():8} total={summary['total']} "
        f"detected={summary['detected']} "
        f"not_detected={summary['not_detected']} "
        f"positive_confidence={summary['positive_confidence_available']}"
    )

flag_counts = Counter(
    flag
    for item in report["evidence"]
    for flag in item["review_flags"]
)
print("\nREVIEW FLAGS")
for flag, count in sorted(flag_counts.items()):
    print(f"{flag}: {count}")

print("\nRetrieval groups summarized:", len(report["retrieval_summary"]))
print("Saved:", result_path)
print("Original reports and embeddings unchanged.")
