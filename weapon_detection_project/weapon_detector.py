"""
WeaponDetector
==============
Frame-based weapon detection wrapper around a verified, fine-tuned YOLO11n
checkpoint. No tracking, no video I/O beyond simple frame iteration, no
integration with any other module. Scope: single-frame inference only.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("weapon_detection.WeaponDetector")

# ---------------------------------------------------------------------------
# Fixed, trusted configuration (no runtime-supplied paths/URLs accepted)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = Path(__file__).parent / "ml_models" / "weapon" / "best.pt"
APPROVED_MODEL_SHA256 = (
    "66fdda89b7b535bd0afe50d61287b793d25fc7411338b6e14d2b8c9d48cd76f9"
)
EXPECTED_CLASS_MAP = {
    0: "pistol",
    1: "knife",
    2: "rifle",
    3: "shotgun",
    4: "smg",
}
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.40
# Ultralytics' own unset default is 0.70, which was found (via controlled
# experiment) to be too permissive: it let a geometrically near-duplicate box
# (IoU 0.66 with the top detection) survive NMS on a single-object test image.
# 0.50 removes that near-duplicate. It does NOT guarantee exactly one box per
# object in general (a second, more loosely-overlapping candidate box can still
# survive at IoU as low as ~0.34) -- this is a partial mitigation, not a fix
# for all duplicate detections. See NMS investigation notes.
DEFAULT_IOU = 0.50


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class WeaponDetectorError(Exception):
    """Base exception for WeaponDetector failures."""


class ModelHashMismatchError(WeaponDetectorError):
    """Raised when the checkpoint's SHA-256 does not match the approved hash."""


class ModelClassMappingError(WeaponDetectorError):
    """Raised when the loaded model's class map does not match the approved map."""


class InvalidFrameError(WeaponDetectorError):
    """Raised when the supplied frame fails input validation."""


class InferenceError(WeaponDetectorError):
    """Raised when YOLO inference itself fails."""


# ---------------------------------------------------------------------------
# Detection result structure
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeaponDetection:
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


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
# WeaponDetector
# ---------------------------------------------------------------------------
class WeaponDetector:
    """
    Loads a verified YOLO11n weapon-detection checkpoint and exposes a single
    frame-based `.detect(frame)` API. Every load is hash- and class-verified
    before the model is trusted for inference.
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
            "WeaponDetector initialized: model=%s classes=%d imgsz=%d conf=%.2f iou=%.2f",
            self._model_path.name,
            len(self._expected_classes),
            self._imgsz,
            self._conf,
            self._iou,
        )

    # -- initialization steps -------------------------------------------------
    def _verify_checkpoint_exists(self) -> None:
        if not self._model_path.exists() or not self._model_path.is_file():
            logger.error("Configured weapon model checkpoint is missing.")
            raise WeaponDetectorError(
                "Weapon model checkpoint not found at configured path."
            )

    def _verify_checkpoint_hash(self) -> None:
        actual = _sha256_of_file(self._model_path)
        if actual != self._expected_sha256:
            logger.error("Weapon model checkpoint failed hash verification.")
            raise ModelHashMismatchError(
                "Weapon model checkpoint hash does not match the approved checkpoint."
            )

    def _load_model(self) -> YOLO:
        try:
            return YOLO(str(self._model_path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load weapon model checkpoint.")
            raise WeaponDetectorError(
                "Failed to load weapon model checkpoint."
            ) from exc

    def _verify_class_mapping(self) -> None:
        actual = dict(self._model.names)
        if actual != self._expected_classes:
            logger.error("Weapon model class mapping does not match approved mapping.")
            raise ModelClassMappingError(
                "Weapon model class mapping does not match the approved 5-class map."
            )

    # -- public API -------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[WeaponDetection]:
        """
        Run weapon detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            HxWx3 image array (e.g. as read by OpenCV, BGR or RGB).

        Returns
        -------
        List[WeaponDetection]
            One entry per detected weapon above the confidence threshold.
            Empty list if there are no detections.
        """
        _validate_frame(frame)

        try:
            results = self._model.predict(
                frame,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Weapon detection inference failed.")
            raise InferenceError("Weapon detection inference failed.") from exc

        detections: List[WeaponDetection] = []
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
            cls_id = int(cls)
            class_name = self._expected_classes.get(cls_id, "unknown")
            detections.append(
                WeaponDetection(
                    class_name=class_name,
                    class_id=cls_id,
                    confidence=float(confidence),
                    bbox=[float(v) for v in box],
                )
            )

        logger.info("WeaponDetector.detect: %d detection(s)", len(detections))
        return detections
