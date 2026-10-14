"""CLI: Phase 2 pipeline smoke test + label-free measurements for DINOv2 / CLIP.

Examples (run from the repository root):

  # synthetic fixtures + tiny RANDOM checkpoints: checks the code path only
  python -m representation_learning.evaluation.run_eval --source synthetic --tiny-random \
      --output representation_learning/evaluation/runs/smoke_tiny.json

  # committed video + existing report, REAL pinned local weights (CPU), weapon/fire/smoke crops
  python -m representation_learning.evaluation.run_eval --source video \
      --video input/YOUTUBE_YouTubeCCTV196_gun_2.mp4 \
      --report output/YOUTUBE_YouTubeCCTV196_gun_2_report.json \
      --manifest representation_learning/models.manifest.json \
      --output representation_learning/evaluation/runs/real_video_smoke.json

  # a folder of target-object crops (labels are NOT used in Phase 2)
  python -m representation_learning.evaluation.run_eval --source images --images path/to/dir --manifest ...

The report never ranks encoders and never claims representation quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..encoders import ClipEncoder, Dinov2Encoder, EncoderConfig
from ..encoders.errors import EncoderError
from .inputs import directory_inputs, synthetic_inputs, video_report_inputs
from .measure import hardware_context, measure_diagnostic_frames, measure_encoder

NOT_EVALUATED = (
    "This report is a PIPELINE SMOKE TEST with label-free measurements. Representation "
    "quality (semantic usefulness, retrieval accuracy, robustness, generalisation) was NOT "
    "evaluated, no encoder ranking is produced, and DINOv2 and CLIP vectors are in different "
    "spaces and were never compared with each other. A suitable labelled evaluation set is "
    "required (docs/evaluation_data_requirements.md).")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "video", "images"], default="synthetic")
    p.add_argument("--video", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--images", type=Path)
    p.add_argument("--max-items", type=int, default=12)
    p.add_argument("--encoders", default="dinov2,clip")
    p.add_argument("--manifest", type=Path, default=None,
                   help="pinned weights manifest (models.manifest.json); entries must be fully "
                        "pinned and verified. Preferred over --dinov2-dir/--clip-dir.")
    p.add_argument("--include-diagnostic-frames", action="store_true",
                   help="also encode whole frames as a separate, clearly-labelled pipeline "
                        "diagnostic (never part of target-object statistics)")
    p.add_argument("--dinov2-variant", default="small")
    p.add_argument("--clip-variant", default="vit-b-32")
    p.add_argument("--dinov2-dir", type=Path, default=None)
    p.add_argument("--dinov2-sha256", default=None)
    p.add_argument("--clip-dir", type=Path, default=None)
    p.add_argument("--clip-sha256", default=None)
    p.add_argument("--device", default="cpu", help="cpu (default) | auto | cuda[:N]")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch-sizes", default="1,4")
    p.add_argument("--tiny-random", action="store_true",
                   help="use tiny RANDOM checkpoints (NOT pretrained) to test the real code path")
    p.add_argument("--allow-unverified", action="store_true",
                   help="load without an expected SHA-256 (the computed hash is still recorded)")
    p.add_argument("--output", type=Path)
    return p


def build_inputs(a):
    if a.source == "synthetic":
        return synthetic_inputs(a.max_items), {
            "source": "synthetic", "label": "generated fixtures; pipeline correctness only",
            "semantic_validity": False}
    if a.source == "video":
        if not a.video or not a.report:
            raise SystemExit("--source video needs --video and --report")
        return video_report_inputs(a.video, a.report, a.max_items,
                                   include_frames=a.include_diagnostic_frames), {
            "source": "committed_video_via_existing_report",
            "video": a.video.name, "report": a.report.name,
            "label": "pipeline smoke test on ONE video; not evidence of quality or generalisation",
            "categories_used": ["weapon", "fire", "smoke"], "person_excluded": True,
            "target_inputs": "weapon/fire/smoke crops with bbox + evidence provenance",
            "whole_frames": "diagnostics only" if a.include_diagnostic_frames else "not used",
            "semantic_validity": False}
    if not a.images:
        raise SystemExit("--source images needs --images DIR")
    return directory_inputs(a.images, a.max_items), {
        "source": "image_directory", "directory": str(a.images),
        "label": "ad-hoc images, no provenance or labels used", "semantic_validity": False}


def build_encoders(a, tmp: Path) -> Dict[str, Any]:
    names = [n.strip() for n in a.encoders.split(",") if n.strip()]
    out: Dict[str, Any] = {}
    for n in names:
        if n not in ("dinov2", "clip"):
            raise SystemExit(f"unknown encoder {n!r}")
        if a.tiny_random:
            from ..testing.tiny_checkpoints import make_tiny_clip, make_tiny_dinov2
            from ..encoders.integrity import sha256_file
            if n == "dinov2":
                d = make_tiny_dinov2(tmp / "dinov2_tiny")
                variant = "small"          # 384-d, matches the tiny checkpoint
            else:
                d = make_tiny_clip(tmp / "clip_tiny")
                variant = "vit-b-32"
            cfg = EncoderConfig(weights_dir=d, device=a.device, checkpoint_id="tiny-random-NOT-PRETRAINED",
                                expected_weights_sha256=sha256_file(d / "model.safetensors"))
        elif a.manifest:
            from ..encoders.manifest import encoder_from_manifest
            out[n] = encoder_from_manifest("dinov2-small" if n == "dinov2" else "clip-vit-b-32",
                                           a.manifest, device=a.device)
            continue
        else:
            d = a.dinov2_dir if n == "dinov2" else a.clip_dir
            sha = a.dinov2_sha256 if n == "dinov2" else a.clip_sha256
            variant = a.dinov2_variant if n == "dinov2" else a.clip_variant
            cfg = EncoderConfig(weights_dir=Path(d) if d else None, device=a.device,
                                expected_weights_sha256=sha, allow_unverified=a.allow_unverified,
                                checkpoint_id=f"local:{Path(d).name}" if d else "UNSPECIFIED")
        out[n] = (Dinov2Encoder if n == "dinov2" else ClipEncoder)(variant, cfg)
    return out


def run(a) -> Dict[str, Any]:
    all_inputs, input_desc = build_inputs(a)
    inputs = [i for i in all_inputs if i.kind != "frame"]
    frames = [i for i in all_inputs if i.kind == "frame"]
    kinds: Dict[str, int] = {}
    for i in all_inputs:
        kinds[i.kind] = kinds.get(i.kind, 0) + 1
    input_desc.update({"n_inputs": len(inputs), "by_kind": {k: v for k, v in kinds.items()
                                                            if k != "frame"},
                       "n_diagnostic_frames": len(frames)})
    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        encs = build_encoders(a, Path(td))
        for name, enc in encs.items():
            res = measure_encoder(
                enc, inputs, repeats=a.repeats,
                batch_sizes=[int(x) for x in a.batch_sizes.split(",") if x])
            if frames and res.get("load", {}).get("ok"):
                res["diagnostics_whole_frames"] = measure_diagnostic_frames(enc, frames)
            results.append(res)
    report = {
        "report_type": "pipeline_smoke_test",
        "representation_quality_evaluated": False,
        "winner_declared": False,
        "statement": NOT_EVALUATED,
        "checkpoint_kind": "tiny_random_NOT_PRETRAINED" if a.tiny_random else "local_pretrained_checkpoint",
        "real_pretrained_inference_executed": not a.tiny_random,
        "model_selection_note": "DINOv2-small and CLIP ViT-B/32 are a resource-based CPU baseline, "
                                "not a quality ranking.",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": hardware_context(),
        "inputs": input_desc,
        "encoders": results,
        "side_by_side_non_semantic": [
            {"encoder": r["encoder"], "dim": r.get("validity", {}).get("dim"),
             "median_ms_single_end_to_end": r.get("latency_single_item_end_to_end", {}).get("median_ms"),
             "n_ok": r.get("extraction", {}).get("n_ok"), "n_failed": r.get("extraction", {}).get("n_failed")}
            for r in results],
        "side_by_side_caveat": "Dimension and latency are properties of the model/hardware, not "
                               "indicators of embedding quality.",
    }
    return report


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    try:
        rep = run(a)
    except EncoderError as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    txt = json.dumps(rep, indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(txt)
        print(f"wrote {a.output}")
    else:
        print(txt)
    failed_load = [r["encoder"] for r in rep["encoders"] if not r.get("load", {}).get("ok")]
    if failed_load:
        print(f"warning: failed to load: {failed_load}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
