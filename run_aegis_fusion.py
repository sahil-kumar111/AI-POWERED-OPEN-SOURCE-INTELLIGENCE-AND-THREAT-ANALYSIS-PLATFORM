"""Run Threat Fusion against saved AEGIS JSON reports without re-running inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def locate_fusion_root(project_root: Path) -> Path:
    candidates = [project_root / "aegis_threat_fusion_v1", project_root]
    for candidate in candidates:
        if (candidate / "threat_fusion_v1" / "__main__.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the fusion package. Expected "
        "<project>/aegis_threat_fusion_v1/threat_fusion_v1 or <project>/threat_fusion_v1."
    )


def resolve_project_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fuse existing AEGIS video/OSINT JSON outputs; does not run detectors, collectors, or embeddings."
    )
    parser.add_argument("--video-report", required=True,
                        help="Path to one real, already-generated video report JSON (required; no placeholder default)")
    parser.add_argument("--osint-json", default="output/osint_results.json",
                        help="Existing Location Threat Monitor export (default: output/osint_results.json)")
    parser.add_argument("--video-only", action="store_true",
                        help="Skip OSINT; useful when incident/camera correlation inputs are unavailable")
    parser.add_argument("--validation-only", action="store_true",
                        help="Explicitly label output as a software-validation run, not a real threat assessment")
    parser.add_argument("--zone", required=True, help="Actual zone ID shared by the evidence")
    parser.add_argument("--event-time", required=True,
                        help="Actual recording start time in ISO-8601, with timezone (e.g. 2026-10-13T12:00:00Z)")
    parser.add_argument("--latitude", type=float, help="Actual camera latitude; required when including OSINT")
    parser.add_argument("--longitude", type=float, help="Actual camera longitude; required when including OSINT")
    parser.add_argument("--camera-id", default=None, help="Camera/source ID; defaults to video filename")
    parser.add_argument("--window-seconds", type=int, default=3600, help="Temporal tolerance in seconds (default 3600)")
    parser.add_argument("--osint-radius-meters", type=float, default=1000.0,
                        help="Maximum incident-to-camera distance in metres (default 1000)")
    parser.add_argument("--allow-publication-time-proxy", action="store_true",
                        help="Use publishedAt only as an explicit recency proxy, not confirmed incident time")
    parser.add_argument("--crowd-count", type=int, help="Verified count from one measurement only")
    parser.add_argument("--capacity", type=int, help="Verified configured capacity for the same venue/measurement")
    parser.add_argument("--output", default="output/threat_score.json", help="Fusion result path")
    args = parser.parse_args()

    video_path = resolve_project_path(root, args.video_report)
    osint_path = resolve_project_path(root, args.osint_json)
    output_path = resolve_project_path(root, args.output)
    if not video_path.is_file():
        parser.error(f"video report does not exist: {video_path}")
    if not args.video_only and not osint_path.is_file():
        parser.error(f"OSINT JSON does not exist: {osint_path}. Run Location Threat Monitor and click Analyze Location, or pass --video-only.")
    if args.validation_only and not args.video_only:
        parser.error("--validation-only requires --video-only so synthetic timestamps cannot be correlated with OSINT")
    if not args.video_only and (args.latitude is None or args.longitude is None):
        parser.error("--latitude and --longitude must be the actual camera coordinates when OSINT fusion is enabled")
    if (args.latitude is None) != (args.longitude is None):
        parser.error("provide both --latitude and --longitude, or neither")
    if (args.crowd_count is None) != (args.capacity is None) and args.crowd_count is not None:
        parser.error("--crowd-count requires --capacity")
    if args.window_seconds < 0 or args.osint_radius_meters < 0:
        parser.error("window and radius must be non-negative")

    fusion_root = locate_fusion_root(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "threat_fusion_v1",
        "--video-report", str(video_path),
        "--zone", args.zone,
        "--event-time", args.event_time,
        "--window-seconds", str(args.window_seconds),
        "--osint-radius-meters", str(args.osint_radius_meters),
        "--output", str(output_path),
    ]
    if args.camera_id:
        command.extend(["--camera-id", args.camera_id])
    if args.latitude is not None:
        command.extend(["--latitude", str(args.latitude), "--longitude", str(args.longitude)])
    if not args.video_only:
        command.extend(["--osint-json", str(osint_path)])
    if args.allow_publication_time_proxy:
        command.append("--allow-publication-time-proxy")
    if args.validation_only:
        command.append("--validation-only")
    if args.crowd_count is not None:
        command.extend(["--crowd-count", str(args.crowd_count), "--capacity", str(args.capacity)])
    elif args.capacity is not None:
        command.extend(["--capacity", str(args.capacity)])

    print("Threat Fusion is reusing saved JSON inputs; no video or NLP inference will run.")
    print(f"Video report: {video_path}")
    if not args.video_only:
        print(f"OSINT results: {osint_path}")
    print(f"Fusion output: {output_path}")
    return subprocess.run(command, cwd=fusion_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
