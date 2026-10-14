from __future__ import annotations

from typing import Any
from .contracts import number, safe_dict, safe_list, read_json


def explain_video_report(report: dict[str, Any] | str) -> dict[str, Any]:
    """Explain observable output fields from an existing AEGIS video report.

    This is a faithful post-hoc explanation of reported detector outputs. It does
    not claim to reveal the detector's internal reasoning or prove an object exists.
    """
    data = read_json(report)
    if not isinstance(data, dict):
        raise ValueError("Video report must be a JSON object")

    counts = safe_dict(data.get("detection_counts"))
    frames = safe_dict(data.get("detection_frames"))
    ratios = safe_dict(data.get("detection_ratios"))
    means = safe_dict(data.get("average_confidence"))
    maxima = safe_dict(data.get("max_confidence"))
    detections = safe_list(data.get("detections"))
    threats = safe_list(data.get("threat_events"))
    fps = number(data.get("fps"))
    frames_processed = int(data.get("frames_processed") or 0)

    categories: list[dict[str, Any]] = []
    for category in sorted(set(counts) | set(frames) | set(ratios) | set(means) | set(maxima)):
        raw_count = counts.get(category, 0)
        count = int(raw_count) if isinstance(raw_count, (int, float)) and not isinstance(raw_count, bool) else None
        ratio = number(ratios.get(category))
        mean = number(means.get(category))
        maximum = number(maxima.get(category))
        frame_count = frames.get(category)
        row = next((x for x in detections if isinstance(x, dict) and str(x.get("category", "")).lower() == str(category).lower()), {})
        detected = row.get("detected") if isinstance(row, dict) else None
        if detected is None and count is not None:
            detected = count > 0
        categories.append({
            "category": str(category),
            "detector_reported_detected": detected,
            "detection_count": count,
            "frames_with_detection": frame_count if isinstance(frame_count, int) else row.get("detection_frames"),
            "detection_frame_ratio": ratio if ratio is not None else number(row.get("detection_ratio")),
            "mean_confidence": mean if mean is not None else number(row.get("average_confidence")),
            "maximum_confidence": maximum if maximum is not None else number(row.get("max_confidence")),
            "explanation": _category_summary(str(category), detected, count, frame_count, ratio, mean, maximum),
        })

    event_examples: list[dict[str, Any]] = []
    for item in threats[:12]:
        if not isinstance(item, dict):
            continue
        explained_detections = []
        for det in safe_list(item.get("detections")):
            if not isinstance(det, dict):
                continue
            confidence = number(det.get("confidence"))
            explained_detections.append({
                "category": det.get("category"),
                "class_name": det.get("class_name"),
                "confidence": confidence,
                "bounding_box_xyxy": det.get("bbox") if isinstance(det.get("bbox"), list) and len(det["bbox"]) == 4 else None,
                "track_id": det.get("track_id"),
                "evidence_note": "Bounding box and confidence are copied from the detector report; box format is xyxy in report coordinates.",
            })
        event_examples.append({
            "frame": item.get("frame"),
            "timestamp_seconds": number(item.get("timestamp_sec")),
            "threat_categories": safe_list(item.get("threats")),
            "detections": explained_detections,
        })

    results = [c for c in categories if c.get("detector_reported_detected") is True]
    summary = (
        f"The saved detector report contains {len(results)} category/categories marked detected, "
        f"across {frames_processed} processed frames and {len(threats)} threat-event record(s)."
    )
    return {
        "explanation_type": "detector_output_trace",
        "source_report": data.get("input_video"),
        "frames_processed": frames_processed,
        "fps": fps,
        "categories": categories,
        "event_examples": event_examples,
        "total_threat_event_records": len(threats),
        "summary": summary,
        "limitations": [
            "This describes saved detector outputs; it is not an internal causal explanation of the detector.",
            "Confidence scores are model scores, not calibrated probabilities unless separately validated.",
            "Repeated frame detections are not necessarily independent incidents.",
            "A detector output alone does not prove an object or incident is real; human review is required.",
        ],
    }


def _category_summary(category: str, detected: Any, count: int | None, frame_count: Any,
                      ratio: float | None, mean: float | None, maximum: float | None) -> str:
    if detected is False or (count == 0 and not detected):
        return f"The report marks {category} as not detected. This means no positive detection was recorded by this detector in this video, not proof of absence."
    parts = [f"The report marks {category} as detected"]
    if count is not None:
        parts.append(f"with {count} recorded detection(s)")
    if isinstance(frame_count, int):
        parts.append(f"across {frame_count} frame(s) containing detections")
    if ratio is not None:
        parts.append(f"(reported frame ratio {ratio:.4f})")
    if mean is not None:
        parts.append(f"with mean confidence {mean:.4f}")
    if maximum is not None:
        parts.append(f"and maximum confidence {maximum:.4f}")
    return " ".join(parts) + "."
