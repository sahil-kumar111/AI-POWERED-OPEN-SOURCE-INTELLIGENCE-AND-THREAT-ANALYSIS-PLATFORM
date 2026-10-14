import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
output_dir = root / "representation_learning" / "outputs"
normalized_path = output_dir / "aegis_normalized_video_evidence.json"
backup_path = output_dir / "aegis_normalized_video_evidence.pre_confidence_fix.json"

data = json.loads(normalized_path.read_text(encoding="utf-8"))

if backup_path.exists():
    raise SystemExit(f"Backup already exists; refusing to overwrite: {backup_path}")

# Preserve the exact pre-fix file.
shutil.copy2(normalized_path, backup_path)

updated = 0
missing = []

for record in data["records"]:
    source_report = record.get("source_report")
    evidence_id = record.get("evidence_id")

    if not source_report or not evidence_id:
        missing.append(evidence_id)
        continue

    report_path = root / source_report
    if not report_path.exists():
        raise SystemExit(f"Source report missing: {report_path}")

    try:
        index = int(evidence_id.rsplit("-", 1)[1])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        original = report["detections"][index]
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise SystemExit(f"Cannot map {evidence_id}: {exc}")

    if original.get("category") != record.get("category"):
        raise SystemExit(f"Category mismatch for {evidence_id}")

    value = original.get("avg_confidence")
    record["confidence"] = value

    if value is not None:
        updated += 1

if missing:
    raise SystemExit(f"Missing source identifiers; no repaired file written: {missing}")

# Validate all mapped values against the original reports.
for record in data["records"]:
    report = json.loads(
        (root / record["source_report"]).read_text(encoding="utf-8")
    )
    index = int(record["evidence_id"].rsplit("-", 1)[1])
    expected = report["detections"][index].get("avg_confidence")
    if record.get("confidence") != expected:
        raise SystemExit(f"Confidence validation failed: {record['evidence_id']}")

normalized_path.write_text(
    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("RESULT: PASS")
print("Records:", len(data["records"]))
print("Confidence values restored:", updated)
print("Backup:", backup_path)
print("Updated:", normalized_path)
print("Original reports unchanged.")
