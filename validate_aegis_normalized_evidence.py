from pathlib import Path
import json
import math
from collections import Counter

root = Path(__file__).resolve().parent
path = (
    root / "representation_learning" / "outputs"
    / "aegis_normalized_video_evidence.json"
)

data = json.loads(path.read_text(encoding="utf-8"))
records = data["records"]
errors = []

if data.get("record_count") != len(records):
    errors.append("record_count does not match records length")

ids = [r.get("evidence_id") for r in records]
if len(ids) != len(set(ids)):
    errors.append("Duplicate evidence IDs")

categories = Counter(r.get("category") for r in records)
statuses = Counter(r.get("detected") for r in records)

for i, r in enumerate(records):
    if r.get("category") not in {"person", "weapon", "fire", "smoke"}:
        errors.append(f"Record {i}: invalid category")

    if not isinstance(r.get("detected"), bool):
        errors.append(f"Record {i}: invalid detection status")

    confidence = r.get("avg_confidence")
    if confidence is not None:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            errors.append(f"Record {i}: invalid confidence")

    for field in ("first_seen_sec", "last_seen_sec"):
        value = r.get(field)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            errors.append(f"Record {i}: invalid {field}")

    if r.get("category") == "person":
        count = r.get("person_count")
        peak = r.get("peak_concurrent_persons")

        if count is not None and peak is not None and peak > count:
            errors.append(f"Record {i}: peak exceeds person count")

print("=" * 60)
print("AEGIS NORMALIZED DATA VALIDATION")
print("=" * 60)
print("Records:", len(records))
print("Unique IDs:", len(ids) == len(set(ids)))
print("Categories:", dict(categories))
print("Detection statuses:", dict(statuses))
print("Validation errors:", len(errors))

for error in errors:
    print("ERROR:", error)

print("RESULT:", "PASS" if not errors else "FAIL")
