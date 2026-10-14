from pathlib import Path
import json
from collections import Counter, defaultdict

root = Path(__file__).resolve().parent
out = root / "representation_learning" / "outputs"
metadata = json.loads(
    (out / "aegis_evidence_metadata.json").read_text(encoding="utf-8")
)

records = metadata.get("records", [])

print("=" * 60)
print(" DUPLICATE EVIDENCE IDs")
print("=" * 60)
counts = Counter(r.get("evidence_id") for r in records)
for evidence_id, count in counts.items():
    if count > 1:
        print(f"{evidence_id}: {count} occurrences")

for source in ("person_detection", "crowd_detection"):
    selected = [r for r in records if r.get("source_type") == source]
    print(f"\n{'=' * 60}\n{source.upper()} RECORDS\n{'=' * 60}")

    seen = set()
    for record in selected:
        description = record.get("description", "")
        if description in seen:
            continue
        seen.add(description)

        print("\nEvidence ID:", record.get("evidence_id"))
        print("Source report:", record.get("source_report"))
        print("Description:", description)
        print("Detection metadata:")
        print(json.dumps(record.get("detection_metadata", {}), indent=2))

print("\n" + "=" * 60)
print("ACTUAL REPORT SCHEMAS")
print("=" * 60)

for folder_name in ("Output", "output"):
    folder = root / folder_name
    if not folder.exists():
        continue

    for path in sorted(folder.rglob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(report, dict):
            continue

        detections = report.get("detections")
        if not isinstance(detections, list) or not detections:
            continue

        print(f"\nReport: {path.relative_to(root)}")
        print("Top-level keys:", list(report.keys()))
        print("First detection:")
        print(json.dumps(detections[0], indent=2))
        break
