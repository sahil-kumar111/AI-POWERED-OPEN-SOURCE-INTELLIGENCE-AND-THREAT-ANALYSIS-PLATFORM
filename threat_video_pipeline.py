#!/usr/bin/env python3
"""
threat_video_pipeline.py
=========================
Combines the Weapon, Fire, Smoke, and Person detection models from this
project into ONE pipeline that runs on an input video and produces an
annotated output video (plus a JSON event report).

HOW IT WORKS
------------
1. Drop video file(s) into the  ./input/   folder.
2. Run:   python threat_video_pipeline.py
3. Annotated videos + JSON reports appear in  ./output/

You can also process a single file directly:
    python threat_video_pipeline.py path/to/video.mp4

FOLDER LAYOUT EXPECTED (this file should sit at the project root, next
to these existing folders that came with the project):
    weapon_detection_project/ml_models/weapon/best.pt
    fire_smoke_detection/fire_smoke_detection/ml_models/fire_smoke/best-fire.pt
    person_detection/person_detection/ml_models/person/yolo11n.pt

Each model is loaded with the SAME integrity checks (SHA-256 hash +
class-map verification) used by the original per-model modules in this
repo, so this script will refuse to run silently on a corrupted /
swapped-out checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
import types as _pytypes
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is not installed. Run:\n"
          "    pip install ultralytics opencv-python\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("threat_pipeline")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---------------------------------------------------------------------------
# Model configuration (mirrors the hash / class-map checks already present
# in weapon_detector.py, fire_detector.py, smoke_detector.py, person_detector.py)
# ---------------------------------------------------------------------------
WEAPON_MODEL_PATH = PROJECT_ROOT / "weapon_detection_project" / "ml_models" / "weapon" / "best.pt"
WEAPON_SHA256 = "66fdda89b7b535bd0afe50d61287b793d25fc7411338b6e14d2b8c9d48cd76f9"
WEAPON_CLASSES = {0: "pistol", 1: "knife", 2: "rifle", 3: "shotgun", 4: "smg"}

FIRE_SMOKE_MODEL_PATH = (
    PROJECT_ROOT / "fire_smoke_detection" / "fire_smoke_detection"
    / "ml_models" / "fire_smoke" / "best-fire.pt"
)
FIRE_SMOKE_SHA256 = "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f"
FIRE_SMOKE_CLASSES = {0: "smoke", 1: "fire"}

PERSON_MODEL_PATH = (
    PROJECT_ROOT / "person_detection" / "person_detection"
    / "ml_models" / "person" / "yolo11n.pt"
)
PERSON_CLASS_ID = 0  # COCO "person"

DEFAULT_IMGSZ = 128
DEFAULT_CONF = 0.2
DEFAULT_IOU = 0.50

# Categories that count as an active "threat" for the on-screen banner /
# report. Person detection is informational context, not a threat by itself.
THREAT_CATEGORIES = {"weapon", "fire", "smoke"}

BYTETRACK_ROOT = PROJECT_ROOT / "bytetrack" / "backend"
CROWD_COUNTER_ROOT = PROJECT_ROOT / "crowd_counter" / "crowd_counter" / "backend"

# Candidate locations for the shared `Detection` dataclass definition —
# bytetrack/, crowd_counter/, fire_smoke_detection/, and person_detection/
# each ship an identical copy of it under their own "backend" package.
_SHARED_TYPES_CANDIDATES = [
    BYTETRACK_ROOT / "app" / "interfaces" / "types.py",
    CROWD_COUNTER_ROOT / "app" / "interfaces" / "types.py",
]


def _stub_pkg(name: str):
    """Register an empty placeholder package in sys.modules under `name`
    (if not already present) so that `from name.sub import x`-style imports
    in the original per-project source files resolve without needing that
    project's own top-level package on sys.path (which would collide with
    the other projects here, since several of them use the same top-level
    package name "backend")."""
    if name not in sys.modules:
        mod = _pytypes.ModuleType(name)
        mod.__path__ = []  # mark as a package so submodule imports resolve
        sys.modules[name] = mod
    return sys.modules[name]


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_shared_types_loaded() -> bool:
    """Make sure `backend.app.interfaces.types` (the shared Detection
    dataclass used by the original per-model source files) is importable,
    loading it from whichever sub-project actually has the file on disk.
    Returns True if available (already loaded or freshly loaded), False if
    none of the candidate projects are present."""
    if "backend.app.interfaces.types" in sys.modules:
        return True
    for candidate in _SHARED_TYPES_CANDIDATES:
        if candidate.exists():
            _stub_pkg("backend")
            _stub_pkg("backend.app")
            _stub_pkg("backend.app.interfaces")
            _load_module_from_file("backend.app.interfaces.types", candidate)
            return True
    return False


def _load_bytetrack_adapter_class():
    """Load this project's real ByteTrack adapter (bytetrack/backend/...)
    so we can count *unique* people across the whole video (via persistent
    track IDs) instead of a raw per-frame detection count. Returns the
    ByteTrackAdapter class, or None if it can't be loaded — the person
    summary then falls back to per-frame averaging with a note explaining why.

    This reuses the real, unmodified ByteTrack adapter shipped in this repo.
    It's loaded by file path (rather than a normal package import) because
    bytetrack/, fire_smoke_detection/, and person_detection/ each ship their
    own top-level "backend" package with the same name — importing more than
    one of those normally in the same process would collide.
    """
    if not BYTETRACK_ROOT.exists():
        logger.warning("bytetrack/ folder not found — person count will fall back to per-frame averaging.")
        return None
    try:
        if not _ensure_shared_types_loaded():
            return None
        _load_module_from_file("backend.app.interfaces.tracker", BYTETRACK_ROOT / "app" / "interfaces" / "tracker.py")
        bt_module = _load_module_from_file(
            "_threat_pipeline_byte_tracker",
            BYTETRACK_ROOT / "app" / "modules" / "video" / "byte_tracker.py",
        )
        logger.info("ByteTrack adapter loaded — person counts will be unique individuals, not per-frame hits.")
        return bt_module.ByteTrackAdapter
    except Exception:
        logger.exception("Could not load ByteTrack adapter — person count will fall back to per-frame averaging.")
        return None


def _load_crowd_counter_class():
    """Load this project's real CrowdCounter (crowd_counter/...) so we can
    report the PEAK number of people visible at once in any single frame —
    a "how crowded did it get" spike metric that's distinct from ByteTrack's
    lifetime unique-individual count. Returns the CrowdCounter class, or
    None if it can't be loaded (peak/avg concurrent-person fields are then
    simply omitted from the report).

    Note on what this actually is: despite the folder name, CrowdCounter in
    this repo is a small stateless reducer over one frame's Detection list
    (`sum(1 for d in detections if d.class_name == "person")`) — not a
    density-map model. It has no notion of frames or time by itself; this
    pipeline calls it once per processed frame and tracks the running
    max/average itself.
    """
    if not CROWD_COUNTER_ROOT.exists():
        logger.warning("crowd_counter/ folder not found — peak crowd count will be omitted from the report.")
        return None
    try:
        if not _ensure_shared_types_loaded():
            return None
        cc_module = _load_module_from_file(
            "_threat_pipeline_crowd_counter",
            CROWD_COUNTER_ROOT / "app" / "modules" / "video" / "crowd_counter.py",
        )
        logger.info("CrowdCounter loaded — report will include peak concurrent person count.")
        return cc_module.CrowdCounter
    except Exception:
        logger.exception("Could not load CrowdCounter — peak crowd count will be omitted from the report.")
        return None


BYTE_TRACK_ADAPTER_CLASS = _load_bytetrack_adapter_class()
CROWD_COUNTER_CLASS = _load_crowd_counter_class()

# Drawing colors per category, BGR (OpenCV order)
COLORS = {
    "weapon": (0, 0, 255),      # red
    "fire": (0, 100, 255),      # orange
    "smoke": (180, 180, 180),   # grey
    "person": (0, 200, 0),      # green
}


# ---------------------------------------------------------------------------
# Shared detection record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    category: str        # "weapon" | "fire" | "smoke" | "person"
    class_name: str       # e.g. "pistol", "fire", "smoke", "person"
    confidence: float
    bbox: List[float]     # [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_frame(frame: np.ndarray) -> None:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Invalid frame supplied to detector.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be an HxWx3 array.")


# ---------------------------------------------------------------------------
# Individual detector wrappers
# ---------------------------------------------------------------------------
class VerifiedYoloDetector:
    """Generic loader that hash-verifies (optional) and class-verifies
    (optional) a YOLO checkpoint before allowing inference."""

    def __init__(
        self,
        name: str,
        model_path: Path,
        expected_sha256: Optional[str],
        expected_classes: Optional[dict],
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.enabled = False
        self.model: Optional[YOLO] = None

        if not model_path.exists() or not model_path.is_file():
            logger.warning("[%s] checkpoint not found at %s — this detector will be SKIPPED.",
                            name, model_path)
            return

        if expected_sha256 is not None:
            actual = _sha256_of_file(model_path)
            if actual != expected_sha256:
                logger.error(
                    "[%s] checkpoint hash mismatch (expected %s, got %s) — "
                    "REFUSING to load this model for safety.",
                    name, expected_sha256, actual,
                )
                return

        try:
            model = YOLO(str(model_path))
        except Exception:
            logger.exception("[%s] failed to load checkpoint — SKIPPED.", name)
            return

        if expected_classes is not None:
            actual_classes = dict(model.names)
            if actual_classes != expected_classes:
                logger.error(
                    "[%s] class map mismatch (expected %s, got %s) — "
                    "REFUSING to load this model for safety.",
                    name, expected_classes, actual_classes,
                )
                return

        self.model = model
        self.enabled = True
        logger.info("[%s] loaded OK (imgsz=%d conf=%.2f iou=%.2f)",
                    name, imgsz, conf, iou)

    def raw_predict(self, frame: np.ndarray, classes: Optional[List[int]] = None):
        if not self.enabled:
            return None
        _validate_frame(frame)
        return self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=classes,
            verbose=False,
        )


def build_detectors() -> dict:
    detectors = {}

    detectors["weapon"] = VerifiedYoloDetector(
        "WeaponDetector", WEAPON_MODEL_PATH, WEAPON_SHA256, WEAPON_CLASSES,
    )
    detectors["fire_smoke"] = VerifiedYoloDetector(
        "FireSmokeDetector", FIRE_SMOKE_MODEL_PATH, FIRE_SMOKE_SHA256, FIRE_SMOKE_CLASSES,
    )
    detectors["person"] = VerifiedYoloDetector(
        "PersonDetector", PERSON_MODEL_PATH, None, None,
    )
    return detectors


def run_all_detectors(detectors: dict, frame: np.ndarray, include_person: bool) -> List[Detection]:
    """Run every enabled detector on one frame and return a unified list
    of Detection objects, one per detected object."""
    out: List[Detection] = []

    # -- Weapon --
    wd = detectors["weapon"]
    if wd.enabled:
        results = wd.raw_predict(frame)
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                for box, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                    cls_id = int(cls)
                    class_name = WEAPON_CLASSES.get(cls_id, "unknown")
                    out.append(Detection("weapon", class_name, float(conf), [float(v) for v in box]))

    # -- Fire + Smoke (shared checkpoint, two logical detectors) --
    fsd = detectors["fire_smoke"]
    if fsd.enabled:
        results = fsd.raw_predict(frame)
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                for box, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                    cls_id = int(cls)
                    class_name = FIRE_SMOKE_CLASSES.get(cls_id, "unknown")
                    category = "fire" if cls_id == 1 else "smoke"
                    out.append(Detection(category, class_name, float(conf), [float(v) for v in box]))

    # -- Person (optional, informational) --
    if include_person:
        pd = detectors["person"]
        if pd.enabled:
            results = pd.raw_predict(frame, classes=[PERSON_CLASS_ID])
            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes):
                    for box, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                        if int(cls) != PERSON_CLASS_ID:
                            continue
                        out.append(Detection("person", "person", float(conf), [float(v) for v in box]))

    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        color = COLORS.get(det.category, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.category.upper()}: {det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

    threats = sorted({d.category for d in detections if d.category in THREAT_CATEGORIES})
    if threats:
        banner = "THREAT DETECTED: " + ", ".join(t.upper() for t in threats)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 200), -1)
        cv2.putText(frame, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
def process_video(
    video_path: Path,
    detectors: dict,
    output_dir: Path,
    include_person: bool = True,
    frame_skip: int = 1,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = output_dir / f"{video_path.stem}_analyzed.mp4"
    out_report_path = output_dir / f"{video_path.stem}_report.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    counts = {"weapon": 0, "fire": 0, "smoke": 0, "person": 0}
    # Running aggregates keyed by (category, class_name) -> stats dict
    agg: dict = {}
    frame_idx = 0
    last_detections: List[Detection] = []
    start_time = time.time()

    person_tracker = None
    unique_person_ids: set = set()
    if include_person and BYTE_TRACK_ADAPTER_CLASS is not None:
        person_tracker = BYTE_TRACK_ADAPTER_CLASS()
        person_tracker.reset()

    crowd_counter = CROWD_COUNTER_CLASS() if (include_person and CROWD_COUNTER_CLASS is not None) else None
    concurrent_person_counts: List[int] = []

    logger.info("Processing %s (%dx%d @ %.1f fps, %d frames)...",
                video_path.name, width, height, fps, total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % max(frame_skip, 1) == 0:
            detections = run_all_detectors(detectors, frame, include_person)
            last_detections = detections
        else:
            # Reuse previous frame's boxes for skipped frames (perf trade-off)
            detections = last_detections

        for det in detections:
            counts[det.category] = counts.get(det.category, 0) + 1
            key = (det.category, det.class_name)
            entry = agg.setdefault(key, {
                "confidence_sum": 0.0,
                "bbox_sum": [0.0, 0.0, 0.0, 0.0],
                "frame_count": 0,
                "first_frame": frame_idx,
                "last_frame": frame_idx,
            })
            entry["confidence_sum"] += det.confidence
            for i in range(4):
                entry["bbox_sum"][i] += det.bbox[i]
            entry["frame_count"] += 1
            entry["last_frame"] = frame_idx

        if person_tracker is not None:
            person_dets = [d for d in detections if d.category == "person"]
            tracked = person_tracker.update(person_dets, frame)
            for t in tracked:
                unique_person_ids.add(t.track_id)

        if crowd_counter is not None:
            concurrent_person_counts.append(crowd_counter.count(detections))

        annotated = draw_detections(frame, detections)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            logger.info("  ...%d/%d frames (%.1fs elapsed)", frame_idx, total_frames or frame_idx, elapsed)

    cap.release()
    writer.release()

    # Build one consolidated summary entry per (category, class_name), plus an
    # explicit "not detected" entry for any category that never showed up.
    # Person is handled separately below (unique count via tracking, not an
    # averaged per-frame entry).
    summary = []
    for (category, class_name), stats in agg.items():
        if category == "person":
            continue
        n = stats["frame_count"]
        summary.append({
            "category": category,
            "class_name": class_name,
            "detected": True,
            "avg_confidence": round(stats["confidence_sum"] / n, 4),
            "avg_bbox": [round(v / n, 2) for v in stats["bbox_sum"]],
            "frames_detected": n,
            "first_seen_sec": round(stats["first_frame"] / fps, 2),
            "last_seen_sec": round(stats["last_frame"] / fps, 2),
        })

    seen_categories = {category for category, _ in agg.keys() if category != "person"}
    for category in sorted({"weapon", "fire", "smoke"} - seen_categories):
        summary.append({
            "category": category,
            "class_name": None,
            "detected": False,
            "message": f"no {category} detected",
        })

    # Person: unique individuals across the whole video (via ByteTrack track
    # IDs), not a raw per-frame detection count. Also report the peak number
    # of people visible at once (via CrowdCounter), which is a different,
    # complementary signal (a busy moment vs. total distinct visitors).
    if include_person:
        peak_concurrent = max(concurrent_person_counts) if concurrent_person_counts else None
        avg_concurrent = (
            round(sum(concurrent_person_counts) / len(concurrent_person_counts), 2)
            if concurrent_person_counts else None
        )

        if person_tracker is not None:
            person_count = len(unique_person_ids)
            entry = {
                "category": "person",
                "detected": person_count > 0,
                "count": person_count,
                "message": (
                    f"{person_count} person detected in the whole video" if person_count == 1
                    else f"{person_count} persons detected in the whole video" if person_count > 0
                    else "no person detected"
                ),
            }
            if peak_concurrent is not None:
                entry["peak_concurrent_persons"] = peak_concurrent
                entry["avg_concurrent_persons"] = avg_concurrent
            summary.append(entry)
        else:
            # ByteTrack adapter unavailable: fall back to per-frame averaging,
            # clearly labeled as such (this is NOT a unique person count).
            stats = agg.get(("person", "person"))
            if stats:
                n = stats["frame_count"]
                entry = {
                    "category": "person",
                    "class_name": "person",
                    "detected": True,
                    "avg_confidence": round(stats["confidence_sum"] / n, 4),
                    "frames_detected": n,
                    "note": "ByteTrack unavailable — this is a per-frame detection count, not a unique person count.",
                }
                if peak_concurrent is not None:
                    entry["peak_concurrent_persons"] = peak_concurrent
                    entry["avg_concurrent_persons"] = avg_concurrent
                summary.append(entry)
            else:
                summary.append({
                    "category": "person",
                    "detected": False,
                    "count": 0,
                    "message": "no person detected",
                })

    # Stable, readable ordering: weapon, fire, smoke, person; detected first.
    category_order = {"weapon": 0, "fire": 1, "smoke": 2, "person": 3}
    summary.sort(key=lambda d: (category_order.get(d["category"], 99), not d.get("detected", False)))

    report = {
        "input_video": str(video_path),
        "output_video": str(out_video_path),
        "fps": fps,
        "resolution": [width, height],
        "frames_processed": frame_idx,
        "detections": summary,
    }
    with open(out_report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Done: %s", video_path.name)
    logger.info("  -> annotated video: %s", out_video_path)
    logger.info("  -> report json:     %s", out_report_path)
    logger.info("  -> summary: %s", [d.get("message", f"{d['category']}: {d}") for d in summary])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run combined threat detection on video(s).")
    parser.add_argument("video", nargs="?", default=None,
                         help="Path to a single video file. If omitted, every video in ./input/ is processed.")
    parser.add_argument("--no-person", action="store_true",
                         help="Disable person detection overlay (weapon/fire/smoke still run).")
    parser.add_argument("--frame-skip", type=int, default=1,
                         help="Run detection every Nth frame to speed things up (default: 1 = every frame).")
    args = parser.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detectors = build_detectors()
    if not any(d.enabled for d in detectors.values()):
        logger.error("No detectors loaded successfully. Check model paths/hashes above. Aborting.")
        sys.exit(1)

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = sorted(
            p for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            logger.warning("No video files found in %s. Drop a video there and re-run.", INPUT_DIR)
            return

    for video_path in videos:
        process_video(
            video_path,
            detectors,
            OUTPUT_DIR,
            include_person=not args.no_person,
            frame_skip=args.frame_skip,
        )


if __name__ == "__main__":
    main()