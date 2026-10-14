from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "Output"
OUT_DIR = ROOT / "representation_learning" / "outputs"
OUTPUT_FILE = OUT_DIR / "aegis_normalized_video_evidence.json"

ALLOWED_CATEGORIES = {"person", "weapon", "fire", "smoke"}

def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None

def normalize_report(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    detections = data.get("detections")

    if not isinstance(detections, list):
        raise ValueError(f"{path.name}: detections must be a list")

    normalized = []

    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise ValueError(f"{path.name}: invalid detection at index {index}")

        category = str(detection.get("category") or "").strip().lower()
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"{path.name}: unexpected category {category!r}"
            )

        detected = detection.get("detected")
        if not isinstance(detected, bool):
            raise ValueError(
                f"{path.name}: missing/invalid detected status at index {index}"
            )

        count = detection.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            count = None

        peak = detection.get("peak_concurrent_persons")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            peak = None

        average = finite_number(detection.get("avg_concurrent_persons"))
        confidence = finite_number(detection.get("avg_confidence"))

        first_seen = finite_number(detection.get("first_seen_sec"))
        last_seen = finite_number(detection.get("last_seen_sec"))

        normalized.append({
            "evidence_id": f"{path.stem}-detection-{index}",
            "source_type": "video_detection",
            "source_report": str(Path("Output") / path.name),
            "input_video": data.get("input_video"),
            "category": category,
            "class_name": detection.get("class_name"),
            "detected": detected,
            "avg_confidence": confidence,
            "frames_detected": detection.get("frames_detected"),
            "first_seen_sec": first_seen,
            "last_seen_sec": last_seen,
            "person_count": count if category == "person" else None,
            "peak_concurrent_persons": peak if category == "person" else None,
            "avg_concurrent_persons": average if category == "person" else None,
            "message": detection.get("message"),
        })

    return normalized

def main():
    if not REPORT_DIR.is_dir():
        raise SystemExit(f"Report directory not found: {REPORT_DIR}")

    paths = sorted(REPORT_DIR.glob("*report.json"))
    if not paths:
        raise SystemExit("No report JSON files found.")

    records = []
    errors = []

    for path in paths:
        try:
            records.extend(normalize_report(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    if errors:
        print("NORMALIZATION FAILED:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    ids = [record["evidence_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("NORMALIZATION FAILED: duplicate evidence IDs.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "record_count": len(records),
        "source_reports": [path.name for path in paths],
        "records": records,
    }

    OUTPUT_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    confirmed = sum(record["detected"] for record in records)
    negative = sum(not record["detected"] for record in records)

    print("=" * 60)
    print("AEGIS VIDEO EVIDENCE NORMALIZATION")
    print("=" * 60)
    print("Reports processed:", len(paths))
    print("Normalized records:", len(records))
    print("Confirmed detections:", confirmed)
    print("Negative detections:", negative)
    print("Unique evidence IDs:", len(ids) == len(set(ids)))
    print("Output:", OUTPUT_FILE)
    print("Original reports and embeddings were not modified.")
    print("RESULT: PASS")

if __name__ == "__main__":
    main()
