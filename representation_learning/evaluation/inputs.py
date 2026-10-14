"""Evaluation inputs. Three sources, all explicit about what they can/can't show.

- synthetic_inputs(): generated fixtures (pipeline correctness ONLY, no semantics).
- video_report_inputs(): Phase 1 VisualEvidenceItem dicts built from an existing
  pipeline report + the ORIGINAL video (weapon/fire/smoke only; person excluded).
  This is an evaluation-time helper, not the future production adapter.
- directory_inputs(): ad-hoc image files (no provenance beyond the file hash).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..encoders.errors import InvalidInputError, UnsupportedEvidenceError
from ..encoders.integrity import sha256_file

ALLOWED_CATEGORIES = ("weapon", "fire", "smoke")   # person intentionally absent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EvalInput:
    name: str
    image: Any                      # ndarray or PIL.Image (the full frame or the image)
    color_order: Optional[str]      # 'BGR' for OpenCV frames, 'RGB' for ndarray-RGB, None for PIL/gray
    evidence: Optional[Dict[str, Any]] = None
    kind: str = "image"             # image | frame | crop | synthetic
    extra: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ synthetic
def synthetic_inputs(n: int = 12, seed: int = 0) -> List[EvalInput]:
    """Deterministic generated images. Variety of size/mode exercises validation and
    preprocessing paths. They carry no semantic content."""
    rng = np.random.default_rng(seed)
    out: List[EvalInput] = []
    sizes = [(240, 320), (480, 640), (96, 96), (300, 120), (120, 300), (720, 1280)]
    for i in range(n):
        h, w = sizes[i % len(sizes)]
        yy, xx = np.mgrid[0:h, 0:w]
        kind = i % 4
        if kind == 0:      # gradient
            img = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1),
                            np.full_like(xx, (i * 37) % 256)], -1).astype(np.uint8)
        elif kind == 1:    # noise
            img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        elif kind == 2:    # shapes on flat background
            img = np.full((h, w, 3), (i * 53) % 256, np.uint8)
            cy, cx, r = h // 2, w // 2, min(h, w) // 4
            img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = (255, 64, 0)
            img[: h // 5, :] = (20, 20, 180)
        else:              # checkerboard
            c = ((yy // 16 + xx // 16) % 2 * 255).astype(np.uint8)
            img = np.stack([c, c, c], -1)
        if i % 6 == 4:     # a grayscale 2-D variant
            out.append(EvalInput(f"synthetic_{i:02d}_gray", img[..., 0].copy(), None,
                                 kind="synthetic"))
        elif i % 6 == 5:   # a PIL RGBA variant
            rgba = np.concatenate([img, np.full((h, w, 1), 255, np.uint8)], -1)
            out.append(EvalInput(f"synthetic_{i:02d}_rgba", Image.fromarray(rgba, "RGBA"), None,
                                 kind="synthetic"))
        else:
            out.append(EvalInput(f"synthetic_{i:02d}", img, "RGB", kind="synthetic"))
    return out


# ------------------------------------------------------------------ directory
def directory_inputs(directory: Path, max_items: Optional[int] = None) -> List[EvalInput]:
    d = Path(directory)
    if not d.is_dir():
        raise InvalidInputError(f"image directory not found: {d}")
    files = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if max_items:
        files = files[:max_items]
    if not files:
        raise InvalidInputError(f"no images ({sorted(IMAGE_EXTS)}) found in {d}")
    out = []
    for p in files:
        try:
            with Image.open(p) as im:
                im.load()
                out.append(EvalInput(str(p.relative_to(d)), im.copy(), None, kind="image",
                                     extra={"file_sha256": sha256_file(p)}))
        except Exception as e:      # unreadable file: keep as a recorded failure input
            out.append(EvalInput(str(p.relative_to(d)), None, None, kind="image",
                                 extra={"load_error": f"{type(e).__name__}: {e}"}))
    return out


# --------------------------------------------------- video + existing report
def _evidence_id(video_sha: str, frame: int, bbox, kind: str) -> str:
    h = hashlib.sha256(json.dumps([video_sha, frame, bbox, kind]).encode()).hexdigest()
    return "ve_" + h[:12]


def video_report_inputs(video: Path, report: Path, max_items: int = 12,
                        include_frames: bool = False) -> List[EvalInput]:
    """Build Phase 1 VisualEvidenceItem inputs from a pipeline report.

    Reads the ORIGINAL video by frame index (never the annotated output). Only
    weapon/fire/smoke detections are used; person evidence is excluded. Whole frames
    (`include_frames=True`) are returned only for pipeline diagnostics, never as target-object inputs. The video's
    real fps/resolution must match the report, otherwise the report belongs to a
    different video and we refuse.
    """
    import cv2

    video, report = Path(video), Path(report)
    if video.name.lower().endswith("_analyzed.mp4"):
        raise UnsupportedEvidenceError("annotated video must not be used as evidence source")
    rep = json.loads(report.read_text())
    if Path(rep.get("input_video", "")).name != video.name:
        raise InvalidInputError(
            f"report is for {Path(rep.get('input_video', '')).name!r}, not {video.name!r}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise InvalidInputError(f"cannot open video {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if [w, h] != list(rep["resolution"]) or abs(fps - float(rep["fps"])) > 1e-3:
        raise InvalidInputError("video fps/resolution do not match the report")
    vsha = sha256_file(video)
    ms = rep.get("model_settings", {})

    dets = []
    for ev in rep["threat_events"]:
        for d in ev["detections"]:
            if d["category"] in ALLOWED_CATEGORIES:
                dets.append((ev["frame"], d))
    dets.sort(key=lambda t: (t[0], -t[1]["confidence"]))
    if not dets:
        raise InvalidInputError("report has no weapon/fire/smoke detections")
    pick = [dets[i] for i in sorted({int(round(x)) for x in np.linspace(0, len(dets) - 1,
                                                                         min(max_items, len(dets)))})]
    out: List[EvalInput] = []
    frames_cache: Dict[int, np.ndarray] = {}

    def read(fi: int) -> np.ndarray:
        if fi not in frames_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok or fr is None:
                raise InvalidInputError(f"cannot read frame {fi} of {video.name}")
            frames_cache[fi] = fr
        return frames_cache[fi]

    for fi, d in pick:
        if not 0 <= fi < n_frames:
            continue
        frame = read(fi)
        cat = d["category"]
        det = {"category": cat, "class_name": d["class_name"], "confidence": d["confidence"],
               "track_id": d.get("track_id"), "detector_name": "threat_video_pipeline (report)",
               "model_sha256": None,
               "settings": ms.get("weapon" if cat == "weapon" else "fire_smoke",
                                  {"imgsz": 0, "conf": 0.0, "iou": 0.0})}
        base = {"schema_version": "0.1", "video_uri": str(video), "video_sha256": vsha,
                "frame_index": int(fi), "timestamp_sec": fi / fps, "fps": fps,
                "resolution": [w, h], "report_uri": str(report),
                "sensitivity": {"may_contain_persons": True, "retain_pixels": False}}
        box = [float(v) for v in d["bbox"]]
        crop_ev = dict(base, source_kind="crop", bbox_xyxy=box, detector=det,
                       evidence_id=_evidence_id(vsha, fi, box, "crop"))
        out.append(EvalInput(f"f{fi:04d}_{cat}_{d['class_name']}", frame, "BGR", crop_ev,
                             kind="crop"))
        if include_frames and not any(o.evidence["frame_index"] == fi and o.kind == "frame"
                                       for o in out):
            fr_ev = dict(base, source_kind="frame",
                         evidence_id=_evidence_id(vsha, fi, None, "frame"))
            out.append(EvalInput(f"f{fi:04d}_frame", frame, "BGR", fr_ev, kind="frame"))
    cap.release()
    return out
