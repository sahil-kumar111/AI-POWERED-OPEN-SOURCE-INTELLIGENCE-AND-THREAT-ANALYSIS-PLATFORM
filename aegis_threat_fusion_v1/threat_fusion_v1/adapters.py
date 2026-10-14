"""Adapters for AEGIS video reports and Location Threat Monitor output.

Adapters preserve measured upstream confidence and never treat article publication time
as incident occurrence time unless the CLI caller explicitly opts in to a recency proxy.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any
import hashlib
import json
import math


THREAT_CATEGORIES = ("weapon", "fire", "smoke")
_WEAPON_ALIASES = {"weapon", "weapons", "gun", "firearm", "pistol", "knife", "rifle", "shotgun", "smg"}


def _iso(value: str | datetime) -> str:
    if isinstance(value, datetime):
        dt = value
    else:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("capture/event time must be provided as ISO-8601")
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 capture/event time: {value!r}") from exc
    # Naive clock times are documented as UTC for backwards compatibility; real-world
    # runs should pass an explicit timezone/UTC offset.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _location(lat: Any, lon: Any) -> dict[str, float] | None:
    if lat is None and lon is None:
        return None
    try:
        if isinstance(lat, bool) or isinstance(lon, bool):
            return None
        if lat is None or lon is None:
            return None
        latitude, longitude = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return {"lat": latitude, "lon": longitude}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return numeric


def _category(value: Any) -> str | None:
    value = str(value or "").strip().lower().replace("-", "_")
    if value in _WEAPON_ALIASES:
        return "weapon"
    if value in {"fire", "flame", "fire_detection"}:
        return "fire"
    if value in {"smoke", "smoke_detection"}:
        return "smoke"
    return None


def _source_basename(value: Any) -> str:
    raw = str(value or "video-report")
    # Reports may have been produced on Windows and loaded on Linux, or vice versa.
    return Path(PureWindowsPath(raw).as_posix()).name or "video-report"


def _first_seen_by_category(report: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    """Extract relative first-seen seconds from supported frame-event records."""
    times: dict[str, float] = {}
    classes: dict[str, str] = {}
    candidate_lists = [report.get("threat_events"), report.get("events")]
    for event_rows in candidate_lists:
        if not isinstance(event_rows, list):
            continue
        for event_row in event_rows:
            if not isinstance(event_row, dict):
                continue
            timestamp = _number(event_row.get("timestamp_sec", event_row.get("first_seen_sec")))
            detections = event_row.get("detections")
            if not isinstance(detections, list):
                detections = [event_row]
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                cat = _category(detection.get("category", detection.get("type", detection.get("class_name"))))
                if cat not in THREAT_CATEGORIES:
                    continue
                row_time = _number(detection.get("timestamp_sec", detection.get("first_seen_sec")))
                if row_time is None:
                    row_time = timestamp
                if row_time is not None and (cat not in times or row_time < times[cat]):
                    times[cat] = row_time
                    class_name = detection.get("class_name")
                    if class_name:
                        classes[cat] = str(class_name)
    return times, classes


def _dict_number(source: Any, key: str) -> Any:
    return source.get(key) if isinstance(source, dict) else None


def video_report_to_events(
    report: dict[str, Any], *, zone: str, capture_time: str | datetime,
    camera_id: str | None = None, latitude: float | None = None, longitude: float | None = None,
    report_ref: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize one saved video report into no more than one event per threat category.

    Supports both count/metric dictionaries and the summary ``detections`` list emitted
    by ``threat_video_pipeline.py``. Uses a measured max confidence when present, then
    the existing average-confidence field as an explicitly labelled fallback. Repeated
    frame events are only used to recover the earliest relative timestamp/class label;
    they are never independently scored.
    """
    if not isinstance(report, dict):
        raise ValueError("video report must be a JSON object")
    if not isinstance(zone, str) or not zone.strip():
        raise ValueError("zone is required for video-event normalization")
    start_timestamp = _iso(capture_time)
    capture_dt = datetime.fromisoformat(start_timestamp.replace("Z", "+00:00"))
    loc = _location(latitude, longitude)
    out: list[dict[str, Any]] = []
    raw_video = report.get("input_video") or report.get("video_path") or report.get("video")
    if isinstance(raw_video, dict):
        raw_video = raw_video.get("path") or raw_video.get("input_video")
    source_id = camera_id or _source_basename(raw_video or report_ref)

    counts = report.get("detection_counts") if isinstance(report.get("detection_counts"), dict) else {}
    frames = report.get("detection_frames") if isinstance(report.get("detection_frames"), dict) else {}
    ratios = report.get("detection_ratios") if isinstance(report.get("detection_ratios"), dict) else {}
    top_max = report.get("max_confidence") if isinstance(report.get("max_confidence"), dict) else {}
    top_avg = report.get("average_confidence") if isinstance(report.get("average_confidence"), dict) else {}
    first_seen, frame_classes = _first_seen_by_category(report)

    rows: dict[str, dict[str, Any]] = {}
    raw_rows = report.get("detections")
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            cat = _category(row.get("category", row.get("type")))
            if cat in THREAT_CATEGORIES:
                # In case a report has multiple summary records for one category, keep
                # a detected row over an explicit negative row; then prefer higher max conf.
                old = rows.get(cat)
                is_detected = row.get("detected") is True
                old_detected = old.get("detected") is True if old else False
                conf = row.get("max_confidence", row.get("avg_confidence", row.get("average_confidence", -1)))
                old_conf = old.get("max_confidence", old.get("avg_confidence", old.get("average_confidence", -1))) if old else -1
                choose = old is None or (is_detected and not old_detected)
                if not choose and is_detected == old_detected:
                    choose = _number(conf) is not None and (_number(old_conf) is None or float(conf) > float(old_conf))
                if choose:
                    rows[cat] = row

    # If the summary array omits a category, counts in the legacy report remain useful.
    # Never add more than one event per category for this report.
    for cat in THREAT_CATEGORIES:
        row = rows.get(cat, {})
        count_val = _dict_number(counts, cat)
        try:
            count_num = int(count_val) if count_val is not None else None
        except (TypeError, ValueError, OverflowError):
            count_num = None
        explicit_detected = row.get("detected")
        detected = explicit_detected is True or (count_num is not None and count_num > 0)
        if explicit_detected is False and (count_num is None or count_num <= 0):
            detected = False
        if not row and (count_num is None or count_num <= 0):
            continue
        if not detected:
            continue

        # Keep the strongest already-reported confidence value when available. The
        # engine still enforces confidence > 0.60; the adapter does not tune thresholds.
        row_max = row.get("max_confidence")
        top_max_value = _dict_number(top_max, cat)
        if row_max is not None:
            confidence, confidence_basis = row_max, "detections[].max_confidence"
        elif top_max_value is not None:
            confidence, confidence_basis = top_max_value, "max_confidence[category]"
        elif row.get("avg_confidence") is not None:
            confidence, confidence_basis = row.get("avg_confidence"), "detections[].avg_confidence (max unavailable)"
        elif row.get("average_confidence") is not None:
            confidence, confidence_basis = row.get("average_confidence"), "detections[].average_confidence (max unavailable)"
        elif _dict_number(top_avg, cat) is not None:
            confidence, confidence_basis = _dict_number(top_avg, cat), "average_confidence[category] (max unavailable)"
        else:
            confidence, confidence_basis = None, "missing upstream confidence"

        row_first_seen = _number(row.get("first_seen_sec"))
        rel_first = row_first_seen if row_first_seen is not None else first_seen.get(cat)
        event_timestamp = _iso(capture_dt + timedelta(seconds=rel_first)) if rel_first is not None else start_timestamp
        timestamp_basis = "capture_time_plus_first_seen_sec" if row_first_seen is not None else (
            "capture_time_plus_first_threat_event_timestamp_sec" if rel_first is not None else "capture_time (no relative event time in report)"
        )

        details: dict[str, Any] = {
            "confidence_basis": confidence_basis,
            "timestamp_basis": timestamp_basis,
            "aggregation": "one contribution per category per source video report; repeated frames are not summed",
            "source_report": str(report_ref) if report_ref else None,
            "source_video": str(raw_video) if raw_video else None,
        }
        for field in ("detection_frames", "detection_ratio", "frames_detected", "first_seen_sec", "last_seen_sec", "class_name", "avg_bbox"):
            if field in row:
                details[field] = row[field]
        if count_num is not None:
            details["detection_count_across_report"] = count_num
        if cat in frames:
            details["detection_frames_total"] = frames[cat]
        if cat in ratios:
            details["detection_ratio_total"] = ratios[cat]
        if rel_first is not None:
            details["first_seen_sec_resolved"] = rel_first
        class_name = row.get("class_name") or frame_classes.get(cat) or cat
        event = {
            "event_id": f"video:{source_id}:{cat}",
            "timestamp": event_timestamp,
            "source": {"type": "camera", "id": str(source_id), "zone": zone.strip()},
            "type": "weapon_detection" if cat == "weapon" else cat,
            "subtype": str(class_name or row.get("category") or cat),
            "confidence": confidence,
            "details": details,
            "modality": "video",
            "raw_ref": str(report_ref or raw_video or "video-report"),
        }
        if loc is not None:
            event["location"] = loc
        out.append(event)
    return out


def _canonical_event_time(row: dict[str, Any]) -> tuple[str | None, str]:
    """Choose an explicit incident/event timestamp, never publishedAt by default."""
    for key in ("incident_timestamp", "event_timestamp", "incident_time", "event_time", "occurred_at", "timestamp_utc", "timestamp"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), key
    return None, "missing_incident_timestamp"


def _deterministic_osint_id(row: dict[str, Any], index: int) -> str:
    existing = row.get("event_id") or row.get("id")
    if existing is not None and str(existing).strip():
        return str(existing).strip()
    # Deterministic fallback for source rows without IDs; documented in README.
    identity = row.get("url") or json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:20]
    return f"osint:sha256:{digest}"


def _category_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key, "") or "") for key in ("category", "severity", "title", "description", "evidence")).strip()


def osint_rows_to_events(
    rows: list[dict[str, Any]] | dict[str, Any], *, zone: str,
    default_confidence: float = 1.0, allow_publication_time_proxy: bool = False,
) -> list[dict[str, Any]]:
    """Adapt saved Location Threat Monitor output without inventing incident metadata.

    Rows missing a timestamp or coordinates are returned with explicit adapter notes so
    the fusion engine can include them in ``excluded_events`` instead of silently losing
    them. ``publishedAt`` is only accepted as an explicitly requested recency proxy.
    """
    if isinstance(rows, dict):
        if isinstance(rows.get("results"), list):
            rows = rows["results"]
        elif isinstance(rows.get("events"), list):
            rows = rows["events"]
        else:
            raise ValueError("OSINT JSON object must contain a 'results' or 'events' array")
    if not isinstance(rows, list):
        raise ValueError("OSINT JSON must be an array of result rows")
    if not isinstance(zone, str) or not zone.strip():
        raise ValueError("zone is required for OSINT-event normalization")
    if isinstance(default_confidence, bool) or not isinstance(default_confidence, (int, float)) or not 0 <= default_confidence <= 1:
        raise ValueError("default_confidence must be numeric within [0, 1]")

    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            out.append({
                "event_id": f"osint:invalid-row:{index}", "timestamp": None,
                "source": {"type": "osint", "id": f"osint-row-{index}", "zone": zone.strip()},
                "type": "osint_threat", "confidence": default_confidence,
                "details": {"adapter_exclusion_reason": "OSINT row must be a JSON object"},
                "modality": "text", "raw_ref": None,
            })
            continue
        event_id = _deterministic_osint_id(row, index)
        location = _location(
            row.get("incident_latitude", row.get("latitude", row.get("lat"))),
            row.get("incident_longitude", row.get("longitude", row.get("lon"))),
        )
        timestamp, timestamp_basis = _canonical_event_time(row)
        pub_time = row.get("publishedAt") or row.get("published_at")
        if timestamp is None and allow_publication_time_proxy and isinstance(pub_time, str) and pub_time.strip():
            timestamp, timestamp_basis = pub_time.strip(), "publishedAt_recency_proxy_not_incident_time"

        details: dict[str, Any] = {
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "text": _category_text(row),
            "category": row.get("category", ""),
            "severity": row.get("severity", ""),
            "url": row.get("url", ""),
            "publishedAt": pub_time,
            "distance_km": row.get("distance_km"),
            "geographic_classification": row.get("geographic_classification"),
            "confidence_basis": "explicit upstream confidence" if row.get("confidence") is not None else "binary upstream threat-filter acceptance; not a probability",
            "timestamp_basis": timestamp_basis,
        }
        if timestamp is None:
            details["timestamp_exclusion_reason"] = (
                "missing incident/event timestamp; publishedAt is article publication time, not confirmed incident time; "
                "pass --allow-publication-time-proxy only for an explicit recency-proxy analysis"
            )
        if location is None:
            details["location_exclusion_reason"] = "missing or invalid OSINT incident coordinates"
        if row.get("is_threat") is False or row.get("threat") is False:
            details["adapter_exclusion_reason"] = "upstream OSINT result explicitly marks this row as not a threat"

        explicit_conf = row.get("confidence")
        confidence = default_confidence if explicit_conf is None else explicit_conf
        source_id = str(row.get("source") or row.get("url") or event_id)
        event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "source": {"type": "osint", "id": source_id, "zone": zone.strip()},
            "type": "osint_threat",
            "subtype": str(row.get("category") or "reported_threat"),
            "confidence": confidence,
            "details": details,
            "modality": "text",
            "raw_ref": row.get("url") or event_id,
        }
        if location is not None:
            event["location"] = location
        out.append(event)
    return out


def crowd_count_to_event(
    count: int, capacity: int, *, zone: str, timestamp: str | datetime,
    camera_id: str = "crowd-counter", latitude: float | None = None, longitude: float | None = None,
) -> dict[str, Any]:
    """Adapt a count from a single measurement; does not sum people over a full clip."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("crowd count must be a non-negative integer from one measurement")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    if not zone or not zone.strip():
        raise ValueError("zone is required for crowd-event normalization")
    event_timestamp = _iso(timestamp)
    event = {
        "event_id": f"crowd:{camera_id}:{event_timestamp}",
        "timestamp": event_timestamp,
        "source": {"type": "camera", "id": camera_id, "zone": zone.strip()},
        "type": "crowd_alert",
        "confidence": 1.0,
        "details": {"count": count, "capacity": capacity, "count_basis": "single measurement supplied to adapter"},
        "modality": "video",
    }
    loc = _location(latitude, longitude)
    if loc is not None:
        event["location"] = loc
    return event
