import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / "representation_learning" / "outputs"

source = out / "aegis_normalized_video_evidence.json"
data = json.loads(source.read_text(encoding="utf-8"))

review_path = out / "aegis_ground_truth_review.csv"

fields = [
    "evidence_id",
    "source_report",
    "input_video",
    "category",
    "class_name",
    "detector_detected",
    "detector_confidence",
    "frames_detected",
    "first_seen_sec",
    "last_seen_sec",
    "peak_concurrent_persons",
    "avg_concurrent_persons",
    "human_verified",
    "human_category",
    "reviewer_notes",
]

with review_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for record in data["records"]:
        writer.writerow({
            "evidence_id": record.get("evidence_id"),
            "source_report": record.get("source_report"),
            "input_video": record.get("input_video"),
            "category": record.get("category"),
            "class_name": record.get("class_name"),
            "detector_detected": record.get("detected"),
            "detector_confidence": record.get("confidence"),
            "frames_detected": record.get("frames_detected"),
            "first_seen_sec": record.get("first_seen_sec"),
            "last_seen_sec": record.get("last_seen_sec"),
            "peak_concurrent_persons": record.get("peak_concurrent_persons"),
            "avg_concurrent_persons": record.get("avg_concurrent_persons"),
            "human_verified": "",
            "human_category": "",
            "reviewer_notes": "",
        })

print("RESULT: PASS")
print("Records exported:", len(data["records"]))
print("Review sheet:", review_path)
print("Human-verification fields intentionally left blank.")
