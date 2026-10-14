import json
from pathlib import Path
import cv2
from ultralytics import YOLO

video_path = Path(r".\Input\weapon.mp4").resolve()
checkpoint = Path(
    r".\weapon_detection_project\ml_models\weapon\best.pt"
).resolve()
output_path = Path(r".\Output\weapon_frame_detections.json")

threshold = 0.25
model = YOLO(str(checkpoint))
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
detections = []
frame_index = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            frame, conf=threshold, verbose=False
        )[0]

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                detections.append({
                    "frame_index": frame_index,
                    "timestamp_seconds": (
                        frame_index / fps if fps > 0 else None
                    ),
                    "class_id": class_id,
                    "class_name": model.names.get(
                        class_id, str(class_id)
                    ),
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [
                        round(float(v), 2)
                        for v in box.xyxy[0].tolist()
                    ],
                    "confidence_threshold": threshold
                })

        frame_index += 1
finally:
    cap.release()

report = {
    "source_video": str(video_path),
    "checkpoint": str(checkpoint),
    "confidence_threshold": threshold,
    "fps": fps,
    "frames_processed": frame_index,
    "detections": detections,
    "summary": {
        "total_detections": len(detections),
        "frames_with_detections": len({
            item["frame_index"] for item in detections
        }),
        "by_class": {
            name: sum(
                item["class_name"] == name for item in detections
            )
            for name in sorted({
                item["class_name"] for item in detections
            })
        },
        "highest_confidence": (
            max(detections, key=lambda x: x["confidence"])
            if detections else None
        )
    }
}

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps({
    "status": "ok",
    "output": str(output_path.resolve()),
    "frames_processed": frame_index,
    "summary": report["summary"]
}, indent=2, ensure_ascii=False))
