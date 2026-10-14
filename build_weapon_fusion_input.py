import json
from pathlib import Path

source = Path(r".\Output\weapon_frame_detections.json")
destination = Path(r".\Output\weapon_fusion_input.json")

if not source.is_file():
    raise FileNotFoundError(f"Missing source evidence: {source}")

data = json.loads(source.read_text(encoding="utf-8"))
items = data.get("detections", [])

if not isinstance(items, list):
    raise ValueError("detections must be a list")

valid = []
for item in items:
    if not isinstance(item, dict):
        continue
    if not isinstance(item.get("confidence"), (int, float)):
        continue
    valid.append(item)

if not valid:
    raise ValueError("No valid frame-level detections were found")

peak = max(valid, key=lambda item: item["confidence"])
first = min(valid, key=lambda item: item["timestamp_seconds"])
last = max(valid, key=lambda item: item["timestamp_seconds"])

# The source is from the weapon detector, so these classes map to
# the fusion engine's broad weapon category.
weapon_names = {"pistol", "knife", "rifle", "shotgun", "smg"}
if not all(str(item.get("class_name", "")).lower() in weapon_names
           for item in valid):
    raise ValueError("Unexpected class name in weapon evidence")

summary = {
    "category": "weapon",
    "detected": True,
    "class_name": peak["class_name"],
    "max_confidence": float(peak["confidence"]),
    "frames_detected": len(valid),
    "detection_ratio": (
        len({x["frame_index"] for x in valid})
        / max(int(data.get("frames_processed", 1)), 1)
    ),
    "first_seen_sec": float(first["timestamp_seconds"]),
    "last_seen_sec": float(last["timestamp_seconds"]),
    "evidence_file": str(source)
}

report = {
    "input_video": data.get("source_video"),
    "fps": data.get("fps"),
    "frames_processed": data.get("frames_processed"),
    "detections": [summary],
    "threat_events": [
        {
            "timestamp_sec": item["timestamp_seconds"],
            "detections": [{
                "category": "weapon",
                "class_name": item["class_name"],
                "timestamp_sec": item["timestamp_seconds"],
                "confidence": item["confidence"],
                "frame_index": item["frame_index"],
                "bbox_xyxy": item["bbox_xyxy"]
            }]
        }
        for item in valid
    ],
    "provenance": {
        "source": str(source),
        "aggregation": "one weapon summary; frame events retained for traceability",
        "real_world_incident_verified": False
    }
}

destination.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps({
    "status": "ok",
    "output": str(destination.resolve()),
    "frames_processed": report["frames_processed"],
    "weapon_frame_detections": len(valid),
    "peak_class": peak["class_name"],
    "peak_confidence": peak["confidence"],
    "first_seen_sec": summary["first_seen_sec"],
    "last_seen_sec": summary["last_seen_sec"],
    "real_world_incident_verified": False
}, indent=2))
