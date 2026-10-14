from __future__ import annotations

import argparse
import json
from pathlib import Path
from .report import build_xai_report
from .visual_explainer import generate_eigencam_on_video_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain existing AEGIS video/NLP/fusion outputs without rerunning detectors.")
    parser.add_argument("--video-report", help="Existing video detector report JSON")
    parser.add_argument("--combined-intelligence", help="Existing combined intelligence JSON (NLP and source reconciliation)")
    parser.add_argument("--threat-score", help="Existing Phase 4 threat_score.json")
    parser.add_argument("--osint-results", help="Saved OSINT result array/object JSON (optional)")
    parser.add_argument("--visual-video", help="Optional video for one-frame EigenCAM; only the chosen frame is inferred")
    parser.add_argument("--visual-frame-index", type=int, default=0, help="Zero-based frame index for optional visual XAI (default 0)")
    parser.add_argument("--weapon-checkpoint", help="Existing YOLO checkpoint; required with --visual-video")
    parser.add_argument("--visual-output", default="output/xai_eigencam_overlay.jpg", help="Visual overlay output path")
    parser.add_argument("--output", default="output/xai_explanation.json", help="Output explanation JSON")
    args = parser.parse_args()
    json_input = any((args.video_report, args.combined_intelligence, args.threat_score, args.osint_results))
    visual_input = bool(args.visual_video or args.weapon_checkpoint)
    if not json_input and not visual_input:
        parser.error("provide at least one input JSON or --visual-video with --weapon-checkpoint")
    if bool(args.visual_video) != bool(args.weapon_checkpoint):
        parser.error("--visual-video and --weapon-checkpoint must be supplied together")
    for arg_name in ("video_report", "combined_intelligence", "threat_score", "osint_results"):
        value = getattr(args, arg_name)
        if value and not Path(value).expanduser().is_file(): parser.error(f"file not found for --{arg_name.replace('_','-')}: {value}")
    visual = None
    target = Path(args.output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    base_kwargs = dict(video_report=args.video_report, combined_intelligence=args.combined_intelligence,
                       threat_score=args.threat_score, osint_results=args.osint_results)
    if args.visual_video:
        try:
            visual = generate_eigencam_on_video_frame(
                args.visual_video, args.weapon_checkpoint, args.visual_output,
                frame_index=args.visual_frame_index,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            # Do not throw away usable event/NLP/fusion explanations merely because the optional CAM dependency is absent.
            if json_input:
                base_report = build_xai_report(**base_kwargs)
                target.write_text(json.dumps(base_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                parser.error(f"non-visual explanation report saved to {target.resolve()}; visual XAI failed: {exc}")
            parser.error(f"visual XAI failed: {exc}")
    report = build_xai_report(**base_kwargs, visual_explanation=visual)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(target.resolve()), "modules": list(report["explanations"].keys()),
                      "visual_overlay": str(Path(args.visual_output).resolve()) if visual else None}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
