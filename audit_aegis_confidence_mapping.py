import json
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / "Output"
normalized_path = root / "representation_learning" / "outputs" / "aegis_normalized_video_evidence.json"

normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
targets = {
    "crowd_report-detection-0",
    "crowd_report-detection-2",
    "fire_and_smoke_report-detection-0",
    "fire_and_smoke_report-detection-1",
    "fire_smoke_report-detection-1",
    "fire_smoke_report-detection-2",
    "smoke_report-detection-2",
    "weapon1_report-detection-2",
}

for record in normalized["records"]:
    evidence_id = record.get("evidence_id")
    if evidence_id not in targets:
        continue

    report_name = record["source_report"].replace("Output\\", "")
    report_path = out / report_name
    report = json.loads(report_path.read_text(encoding="utf-8"))

    try:
        index = int(evidence_id.rsplit("-", 1)[1])
        original = report["detections"][index]
    except (ValueError, IndexError, KeyError, TypeError):
        print("Could not map:", evidence_id)
        continue

    print("\n" + "=" * 64)
    print("ID:", evidence_id)
    print("Original category:", original.get("category"))
    print("Original class:", original.get("class_name"))
    print("Original detected:", original.get("detected"))
    print("Original avg_confidence:", original.get("avg_confidence"))
    print("Normalized confidence:", record.get("confidence"))
    print("Original frames:", original.get("frames_detected"))
    print("Normalized frames:", record.get("frames_detected"))
