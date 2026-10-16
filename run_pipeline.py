#!/usr/bin/env python3
"""AEGIS - AI-Powered Open-Source Intelligence & Threat Analysis Pipeline.

Single Unified Orchestrator implementing the full End-to-End System Architecture:
1. Location & GIS Agent
2. Video Intelligence Agent (YOLOv11 Person [ByteTrack], Weapon, Fire/Smoke)
3. Gemini Video Understanding
4. OSINT Intelligence Agent (NewsAPI + Geographic Relevance)
5. NLP Intelligence Agent (spaCy NER, VADER, Cross-Source Reconciliation)
6. Representation Learning (384-dim Multimodal Threat Embedding)
7. Threat Assessment & Evidence Fusion (AEGIS Threat Index, Dominant Threat)
8. Explainability & Reporting (SHAP Attribution, EigenCAM heatmap, 7-Question Report)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import DEFAULT_OUTPUT_DIR, run_complete_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AEGIS - Run the complete multimodal threat intelligence pipeline."
    )
    parser.add_argument(
        "--location",
        type=str,
        default="Connaught Place, New Delhi",
        help="Search location or event address (default: 'Connaught Place, New Delhi')",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="input/YOUTUBE_YouTubeCCTV196_gun_2.mp4",
        help="Path to video file or live stream (default: 'input/YOUTUBE_YouTubeCCTV196_gun_2.mp4')",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=15,
        help="Expected venue / sector crowd capacity (default: 15)",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="zone-alpha-1",
        help="Operational zone identifier (default: 'zone-alpha-1')",
    )
    parser.add_argument(
        "--event-time",
        type=str,
        default=None,
        help="Event timestamp in ISO-8601 (default: current UTC time)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Video frame skip interval for faster inference (default: 1)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Run OSINT and Geospatial pipeline without video analysis",
    )

    args = parser.parse_args()
    video_target = None if args.no_video else args.video

    def cli_progress(stage: str, pct: int):
        print(f"[{pct:3d}%] {stage}")

    print("\n" + "=" * 78)
    print(" 🛡️  AEGIS - MULTIMODAL THREAT INTELLIGENCE & ANALYSIS PLATFORM")
    print("=" * 78)
    print(f" Target Location : {args.location}")
    print(f" Video Feed      : {'None (OSINT+GIS Intelligence Mode)' if args.no_video else video_target}")
    print(f" Venue Capacity  : {args.capacity} persons")
    print(f" Operational Zone: {args.zone}")
    print("=" * 78 + "\n")

    result = run_complete_pipeline(
        location=args.location,
        video_path=video_target,
        venue_capacity=args.capacity,
        zone=args.zone,
        event_time=args.event_time,
        frame_skip=args.frame_skip,
        skip_video=args.no_video,
        progress_callback=cli_progress,
    )

    overview = result["event_overview"]
    ati = overview["aegis_threat_index"]
    level = overview["threat_level"]
    classification = overview["threat_classification"]
    dominant = result["threat_detection"]["dominant_threat_category"]

    color_bar = (
        "🔴 [CRITICAL ALERT]" if level == "CRITICAL"
        else "🟠 [HIGH THREAT]" if level == "HIGH"
        else "🟡 [MEDIUM ADVISORY]" if level == "MEDIUM"
        else "🟢 [LOW RISK]"
    )
    meter_filled = int(round(ati / 5.0))
    meter = "█" * meter_filled + "░" * (20 - meter_filled)

    print("\n" + "=" * 78)
    print("                    🎯 MASTER OUTPUT OF THREATS")
    print("=" * 78)
    print(f" {color_bar}")
    print(f" AEGIS THREAT INDEX (ATI) : [{meter}] {ati}/100")
    print(f" OFFICIAL THREAT LEVEL    : {level}")
    print(f" CLASSIFICATION           : {classification}")
    print(f" DOMINANT THREAT CATEGORY : {dominant.upper()}")
    print(f" LOCATION                 : {overview['location']}")
    print(f" PIPELINE EXECUTION TIME  : {result['execution_time_seconds']}s")
    print("-" * 78)

    print(" 📊 MULTIMODAL FEATURE ATTRIBUTION (SHAP DECOMPOSITION):")
    for fa in result["evidence_fusion"].get("feature_attribution", []):
        print(f"   • {fa['factor'].upper():<32} | Impact: {fa['percentage_importance']:>5.1f}% | {fa['evidence_reason']}")

    print("-" * 78)
    print(" 📋 TOP IMMEDIATE OPERATIONAL DIRECTIVES (0-5 MIN):")
    for idx, d in enumerate(result["final_intelligence_report"]["recommendations"].get("immediate_directives_0_to_5_min", []), 1):
        print(f"   {idx}. {d}")

    print("-" * 78)
    print(" 📁 GENERATED ARTIFACTS IN output/:")
    for fname in [
        "threat_score.json",
        "daily_intelligence_report.md",
        "daily_intelligence_report.json",
        "xai_explanation.json",
        "gis_layer.geojson",
        "multimodal_threat_embedding.npz",
        "multimodal_features.json",
        "osint_results.json",
        "combined_intelligence.json",
        "nlp_intelligence.json",
        "threat_summary.json",
    ]:
        fpath = DEFAULT_OUTPUT_DIR / fname
        if fpath.exists():
            print(f"   ✓ {fname:<34} ({fpath.stat().st_size:,} bytes)")
    if (DEFAULT_OUTPUT_DIR / "threat_highlight.jpg").exists():
        print(f"   ✓ {'threat_highlight.jpg':<34} (EigenCAM Spatial Heatmap Overlay)")

    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
