"""
SmokeDetector
=============

Frame-based smoke detection wrapper around a verified YOLO fire/smoke
checkpoint.

The model contains two classes:

    0 -> Fire
    1 -> Smoke

SmokeDetector filters the model output to smoke detections only.

The detector:
- validates the checkpoint
- verifies SHA-256
- verifies the model class mapping
- validates input frames
- performs YOLO inference
- filters smoke detections
- returns the shared Detection type

No video I/O, tracking, or external integration is performed here.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
from ultralytics import YOLO

from backend.app.interfaces.types import Detection

logger = logging.getLogger("video_intelligence.SmokeDetector")


DEFAULT_MODEL_PATH = (
    Path(__file__).parents[3]
    / "ml_models"
    / "fire_smoke"
    / "best_nano_111.pt"
)

APPROVED_MODEL_SHA256 = (
    "3cbeb9569d8c4057734a81437b90c3e95e55ebdd40a5c5ea7dd7873a247cef13"
)

EXPECTED_CLASS_MAP = {
    0: "Fire",
    1: "Smoke",
}

SMOKE_CLASS_ID = 1
SMOKE_CLASS_NAME = "smoke"

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.40
DEFAULT_IOU = 0.50


class SmokeDetectorError(Exception):
    """Base exception for SmokeDetector failures."""


class ModelHashMismatchError(SmokeDetectorError):
    """Raised when the checkpoint SHA-256 does not match."""


class ModelClassMappingError(SmokeDetectorError):
    """Raised when the model class mapping does not match."""


class InvalidFrameError(SmokeDetectorError):
    """Raised when the supplied frame is invalid."""


class InferenceError(SmokeDetectorError):
    """Raised when YOLO inference fails."""


def _sha256_of_file(path: Path) -> str:

    hash_object = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hash_object.update(chunk)

    return hash_object.hexdigest()


def _validate_frame(frame) -> None:

    if frame is None:
        raise InvalidFrameError(
            "Frame is None."
        )

    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError(
            "Frame must be a numpy.ndarray."
        )

    if frame.size == 0:
        raise InvalidFrameError(
            "Frame is empty."
        )

    if frame.ndim != 3:
        raise InvalidFrameError(
            "Frame must be a HxWxC array."
        )

    height, width, channels = frame.shape

    if height <= 0 or width <= 0:
        raise InvalidFrameError(
            "Frame has invalid height/width."
        )

    if channels != 3:
        raise InvalidFrameError(
            "Frame must have exactly 3 channels."
        )


class SmokeDetector:

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

        self._expected_classes = (
            expected_classes or EXPECTED_CLASS_MAP
        )

        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou

        self._verify_checkpoint_exists()

        self._verify_checkpoint_hash()

        self._model = self._load_model()

        self._verify_class_mapping()

        logger.info(
            "SmokeDetector initialized: model=%s classes=%d imgsz=%d conf=%.2f iou=%.2f",
            self._model_path.name,
            len(self._expected_classes),
            self._imgsz,
            self._conf,
            self._iou,
        )

    def _verify_checkpoint_exists(self) -> None:

        if (
            not self._model_path.exists()
            or not self._model_path.is_file()
        ):

            logger.error(
                "Configured fire/smoke model checkpoint is missing."
            )

            raise SmokeDetectorError(
                "Fire/smoke model checkpoint not found at "
                f"{self._model_path}. "
                "No automatic download is performed."
            )

    def _verify_checkpoint_hash(self) -> None:

        actual_hash = _sha256_of_file(
            self._model_path
        )

        if actual_hash != self._expected_sha256:

            logger.error(
                "Fire/smoke model checkpoint failed hash verification."
            )

            raise ModelHashMismatchError(
                "Fire/smoke model checkpoint hash does not "
                "match the approved checkpoint."
            )

    def _load_model(self) -> YOLO:

        try:

            return YOLO(
                str(self._model_path)
            )

        except Exception as exc:

            logger.error(
                "Failed to load fire/smoke model checkpoint."
            )

            raise SmokeDetectorError(
                "Failed to load fire/smoke model checkpoint."
            ) from exc

    def _verify_class_mapping(self) -> None:

        actual_classes = dict(
            self._model.names
        )

        if actual_classes != self._expected_classes:

            logger.error(
                "Fire/smoke model class mapping does not "
                "match approved mapping."
            )

            raise ModelClassMappingError(
                "Fire/smoke model class mapping does not "
                "match the approved "
                "{0:'Fire', 1:'Smoke'} map. "
                f"Actual mapping: {actual_classes}"
            )

    def detect(
        self,
        frame: np.ndarray,
        config: Any = None,
    ) -> List[Detection]:

        _validate_frame(frame)

        try:

            results = self._model.predict(
                frame,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                classes=[SMOKE_CLASS_ID],
                verbose=False,
            )

        except Exception as exc:

            logger.error(
                "Smoke detection inference failed."
            )

            raise InferenceError(
                "Smoke detection inference failed."
            ) from exc

        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.tolist()
        confidences = boxes.conf.tolist()
        classes = boxes.cls.tolist()

        for box, confidence, class_id in zip(
            xyxy,
            confidences,
            classes,
        ):

            class_id = int(class_id)

            if class_id != SMOKE_CLASS_ID:
                continue

            detections.append(
                Detection(
                    bbox=[
                        float(value)
                        for value in box
                    ],
                    class_name=SMOKE_CLASS_NAME,
                    confidence=float(confidence),
                )
            )

        logger.info(
            "SmokeDetector.detect: %d smoke detection(s)",
            len(detections),
        )

        return detections