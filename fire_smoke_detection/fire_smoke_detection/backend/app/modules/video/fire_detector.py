"""
FireDetector
============
Frame-based fire detection wrapper around a verified, fine-tuned YOLOv8n
fire/smoke checkpoint. Implements the VideoDetector contract. The
checkpoint detects both "smoke" (class 0) and "fire" (class 1);
FireDetector filters its output down to "fire" only. No tracking, no
video I/O beyond simple frame iteration, no integration with any other
module.

Checkpoint provenance (as recorded in the checkpoint's own metadata and
supplied verification context -- not independently re-verified here):
    Source: rabahdev/fire-smoke-yolov8n (D-Fire fine-tune)
    License: AGPL-3.0
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
from ultralytics import YOLO

from backend.app.interfaces.types import Detection

logger = logging.getLogger("video_intelligence.FireDetector")

# ---------------------------------------------------------------------------
# Fixed, trusted configuration (no runtime-supplied paths/URLs accepted)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = Path(__file__).parents[3] / "ml_models" / "fire_smoke" / "best-fire.pt"
APPROVED_MODEL_SHA256 = (
    "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f"
)
EXPECTED_CLASS_MAP = {
    0: "smoke",
    1: "fire",
}

FIRE_CLASS_ID = 1
FIRE_CLASS_NAME = "fire"

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.40
DEFAULT_IOU = 0.50


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class FireDetectorError(Exception):
    """Base exception for FireDetector failures."""


class ModelHashMismatchError(FireDetectorError):
    """Raised when the checkpoint's SHA-256 does not match the approved hash."""


class ModelClassMappingError(FireDetectorError):
    """Raised when the loaded model's class map does not match the approved map."""


class InvalidFrameError(FireDetectorError):
    """Raised when the supplied frame fails input validation."""


class InferenceError(FireDetectorError):
    """Raised when YOLO inference itself fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_frame(frame) -> None:
    if frame is None:
        raise InvalidFrameError("Frame is None.")
    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError("Frame must be a numpy.ndarray.")
    if frame.size == 0:
        raise InvalidFrameError("Frame is empty.")
    if frame.ndim != 3:
        raise InvalidFrameError("Frame must be a HxWxC array (3 dimensions).")
    h, w, c = frame.shape
    if h <= 0 or w <= 0:
        raise InvalidFrameError("Frame has invalid height/width.")
    if c != 3:
        raise InvalidFrameError("Frame must have exactly 3 channels.")


# ---------------------------------------------------------------------------
# FireDetector
# ---------------------------------------------------------------------------
class FireDetector:
    """
    Loads the verified fire/smoke YOLOv8n checkpoint and exposes the
    VideoDetector contract's `.detect(frame, config)` API, filtered to
    "fire" (class id 1) detections only. Every load is hash- and
    class-verified before the model is trusted for inference.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        expected_sha256: str = APPROVED_MODEL_SHA256,
        expected_classes: dict = None,
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> None:
        self._model_path = Path(model_path)
        self._expected_sha256 = expected_sha256
        self._expected_classes = expected_classes or EXPECTED_CLASS_MAP
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou

        self._verify_checkpoint_exists()
        self._verify_checkpoint_hash()
        self._model = self._load_model()
        self._verify_class_mapping()

        logger.info(
            "FireDetector initialized: model=%s classes=%d imgsz=%d conf=%.2f iou=%.2f",
            self._model_path.name,
            len(self._expected_classes),
            self._imgsz,
            self._conf,
            self._iou,
        )

    # -- initialization steps -------------------------------------------------
    def _verify_checkpoint_exists(self) -> None:
        if not self._model_path.exists() or not self._model_path.is_file():
            logger.error("Configured fire/smoke model checkpoint is missing.")
            raise FireDetectorError(
                "Fire/smoke model checkpoint not found at configured path. "
                "No automatic download is performed."
            )

    def _verify_checkpoint_hash(self) -> None:
        actual = _sha256_of_file(self._model_path)
        if actual != self._expected_sha256:
            logger.error("Fire/smoke model checkpoint failed hash verification.")
            raise ModelHashMismatchError(
                "Fire/smoke model checkpoint hash does not match the approved checkpoint."
            )

    def _load_model(self) -> YOLO:
        try:
            return YOLO(str(self._model_path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load fire/smoke model checkpoint.")
            raise FireDetectorError(
                "Failed to load fire/smoke model checkpoint."
            ) from exc

    def _verify_class_mapping(self) -> None:
        actual = dict(self._model.names)
        if actual != self._expected_classes:
            logger.error("Fire/smoke model class mapping does not match approved mapping.")
            raise ModelClassMappingError(
                "Fire/smoke model class mapping does not match the approved "
                "{0:'smoke', 1:'fire'} map."
            )

    # -- public API (VideoDetector contract) -------------------------------------
    def detect(self, frame: np.ndarray, config: Any = None) -> List[Detection]:
        """
        Run fire detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            HxWx3 image array (e.g. as read by OpenCV, BGR or RGB).
        config : Any, optional
            Unused placeholder to satisfy the VideoDetector contract's
            signature. FireDetector currently uses its own fixed
            imgsz/conf/iou set at construction time.

        Returns
        -------
        List[Detection]
            One entry per detected "fire" instance above the confidence
            threshold. All "smoke" detections are discarded. Empty list
            if there are no fire detections.
        """
        _validate_frame(frame)

        try:
            results = self._model.predict(
                frame,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                classes=[FIRE_CLASS_ID],
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Fire detection inference failed.")
            raise InferenceError("Fire detection inference failed.") from exc

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        classes = boxes.cls.tolist()

        for box, confidence, cls in zip(xyxy, confs, classes):
            # Defense in depth: `classes=[FIRE_CLASS_ID]` already restricts
            # predict() output, but we still filter explicitly so a
            # Detection is never emitted for "smoke" even if that upstream
            # filter behaves unexpectedly.
            if int(cls) != FIRE_CLASS_ID:
                continue
            detections.append(
                Detection(
                    bbox=[float(v) for v in box],
                    class_name=FIRE_CLASS_NAME,
                    confidence=float(confidence),
                )
            )

        logger.info("FireDetector.detect: %d fire detection(s)", len(detections))
        return detections
