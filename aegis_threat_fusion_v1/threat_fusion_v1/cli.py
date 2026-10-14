from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .adapters import crowd_count_to_event, osint_rows_to_events, video_report_to_events
from .engine import ThreatFusionEngine


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _relative_time(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _peak_crowd_measurement(report: dict[str, Any]) -> tuple[int | None, float | None]:
    """Return peak concurrent count only when a matching peak timestamp is present."""
    candidates: list[dict[str, Any]] = []
    rows = report.get("detections")
    if isinstance(rows, list):
        candidates.extend(row for row in rows if isinstance(row, dict) and str(row.get("category", "")).lower() == "person")
    stats = report.get("person_statistics")
    if isinstance(stats, dict):
        candidates.append(stats)
    candidates.append(report)
    for row in candidates:
        count = row.get("peak_concurrent_persons")
        if isinstance(count, bool) or not isinstance(count, (int, float)) or count < 0:
            continue
        offset = row.get("peak_concurrent_persons_timestamp_sec", row.get("peak_timestamp_sec", row.get("peak_time_sec")))
        return int(count), _relative_time(offset)
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse existing AEGIS video and OSINT JSON reports into an explainable threat score."
    )
    parser.add_argument("--events", help="JSON file with normalized events (array or object containing events)")
    parser.add_argument("--video-report", help="Path to one already-generated video report JSON")
    parser.add_argument("--osint-json", help="Saved Location Threat Monitor JSON (array, results, or events)")
    parser.add_argument("--zone", help="Actual shared zone identifier, not a guessed location")
    parser.add_argument("--event-time", help="Actual video recording start time in ISO-8601; required with --video-report")
    parser.add_argument("--camera-id", default=None, help="Camera/source ID; defaults to video filename")
    parser.add_argument("--latitude", type=float, help="Actual camera latitude; required for OSINT spatial matching")
    parser.add_argument("--longitude", type=float, help="Actual camera longitude; required for OSINT spatial matching")
    parser.add_argument("--crowd-count", type=int, help="Verified count from one measurement/frame, not a per-video sum")
    parser.add_argument("--capacity", type=int, help="Configured site capacity; required for a crowd-over-capacity score")
    parser.add_argument("--window-seconds", type=int, default=3600, help="Allowed distance from reference time in seconds (default 3600)")
    parser.add_argument("--osint-radius-meters", type=float, default=1000.0, help="Max camera-to-incident distance (default 1000m)")
    parser.add_argument("--reference-time", help="Optional explicit fusion reference time; defaults to --event-time")
    parser.add_argument("--allow-publication-time-proxy", action="store_true",
                        help="Explicitly use article publishedAt only as a recency proxy, never as confirmed incident time")
    parser.add_argument("--validation-only", action="store_true",
                        help="Mark output as a software-validation run, not a real-world threat assessment")
    parser.add_argument("--output", help="Write output JSON here as well as stdout")
    args = parser.parse_args()

    if not any((args.events, args.video_report, args.osint_json, args.crowd_count is not None)):
        parser.error("provide at least one of --events, --video-report, --osint-json, or --crowd-count")
    if (args.latitude is None) != (args.longitude is None):
        parser.error("--latitude and --longitude must be supplied together")
    if args.video_report and (not args.zone or not args.event_time):
        parser.error("--video-report requires --zone and the actual --event-time (absolute video capture start time)")
    if args.osint_json and (not args.zone or args.latitude is None or args.longitude is None):
        parser.error("--osint-json requires --zone and the actual camera --latitude and --longitude")
    if args.validation_only and args.osint_json:
        parser.error("--validation-only cannot be combined with --osint-json; do not correlate synthetic timestamps with OSINT")
    if args.crowd_count is not None and args.capacity is None:
        parser.error("--crowd-count requires a verified --capacity")
    if args.crowd_count is not None and (not args.zone or not args.event_time):
        parser.error("--crowd-count requires --zone and --event-time")
    if args.capacity is not None and args.capacity < 0:
        parser.error("--capacity must be non-negative")
    if args.window_seconds < 0 or args.osint_radius_meters < 0:
        parser.error("window and OSINT radius must be non-negative")

    events: list[dict[str, Any]] = []
    adapter_exclusions: list[dict[str, Any]] = []
    reference_time = args.reference_time or args.event_time
    video_report: dict[str, Any] | None = None
    video_path: Path | None = Path(args.video_report).expanduser().resolve() if args.video_report else None
    osint_path: Path | None = Path(args.osint_json).expanduser().resolve() if args.osint_json else None

    if args.events:
        raw = load_json(args.events)
        if isinstance(raw, dict):
            raw = raw.get("events")
        if not isinstance(raw, list):
            parser.error("--events file must be a JSON array or an object with an 'events' array")
        events.extend(raw)

    if video_path:
        if not video_path.is_file():
            parser.error(f"video report does not exist: {video_path}")
        loaded_report = load_json(video_path)
        if not isinstance(loaded_report, dict):
            parser.error("video report must contain a JSON object")
        video_report = loaded_report
        events.extend(video_report_to_events(
            video_report,
            zone=args.zone,
            capture_time=args.event_time,
            camera_id=args.camera_id,
            latitude=args.latitude,
            longitude=args.longitude,
            report_ref=str(video_path),
        ))

    if osint_path:
        if not osint_path.is_file():
            parser.error(f"OSINT JSON does not exist: {osint_path}")
        events.extend(osint_rows_to_events(
            load_json(osint_path),
            zone=args.zone,
            allow_publication_time_proxy=args.allow_publication_time_proxy,
        ))

    if args.crowd_count is not None:
        events.append(crowd_count_to_event(
            args.crowd_count, args.capacity, zone=args.zone, timestamp=args.event_time,
            camera_id=args.camera_id or "crowd-counter", latitude=args.latitude, longitude=args.longitude,
        ))
    elif args.capacity is not None and video_report is not None:
        peak_count, peak_offset = _peak_crowd_measurement(video_report)
        source_ref = str(video_path)
        if peak_count is None:
            adapter_exclusions.append({
                "event_id": f"crowd:peak:{video_path.name}",
                "reason": "no peak_concurrent_persons measurement exists in this report; total person detections/unique-person count are not simultaneous occupancy",
                "raw_ref": source_ref,
            })
        elif peak_offset is None:
            adapter_exclusions.append({
                "event_id": f"crowd:peak:{video_path.name}",
                "reason": "peak_concurrent_persons exists but its timestamp is unavailable; peak was not assigned a fabricated event time",
                "raw_ref": source_ref,
            })
        else:
            start = datetime.fromisoformat(args.event_time.strip().replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            peak_time = (start.astimezone(timezone.utc) + timedelta(seconds=peak_offset)).isoformat().replace("+00:00", "Z")
            events.append(crowd_count_to_event(
                peak_count, args.capacity, zone=args.zone, timestamp=peak_time,
                camera_id=args.camera_id or (video_report.get("input_video") and Path(str(video_report["input_video"])).name) or "crowd-counter",
                latitude=args.latitude, longitude=args.longitude,
            ))
    elif args.capacity is not None and video_report is None:
        parser.error("--capacity without --crowd-count requires --video-report with a timestamped peak_concurrent_persons value")

    engine = ThreatFusionEngine(
        time_window_seconds=args.window_seconds,
        osint_radius_meters=args.osint_radius_meters,
    )
    result = engine.fuse(events, zone=args.zone, reference_time=reference_time)
    result["excluded_events"].extend(adapter_exclusions)
    result["run_metadata"] = {
        "input_video_report": str(video_path) if video_path else None,
        "input_osint_json": str(osint_path) if osint_path else None,
        "explicit_publication_time_proxy_enabled": bool(args.allow_publication_time_proxy),
        "camera_coordinates_supplied": args.latitude is not None and args.longitude is not None,
        "reference_time_source": "--reference-time" if args.reference_time else ("--event-time" if args.event_time else "derived from valid events"),
        "score_semantics": "explainable weighted rule score; not a calibrated probability",
        "validation_only": bool(args.validation_only),
    }
    if args.validation_only:
        result["run_metadata"]["warning"] = (
            "Software-validation result only. The supplied reference timestamp may be synthetic; "
            "do not use this output for OSINT correlation or real-world threat decisions."
        )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
