"""
PersonDetector
==============
Frame-based person detection wrapper around a local YOLO11n COCO-pretrained
checkpoint. Implements the VideoDetector contract. Filters COCO output down
to the "person" class only (COCO class id 0). No tracking, no video I/O
beyond simple frame iteration, no integration with any other module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import numpy as np
from ultralytics import YOLO

from backend.app.interfaces.types import Detection

logger = logging.getLogger("video_intelligence.PersonDetector")

# ---------------------------------------------------------------------------
# Fixed, trusted configuration (no runtime-supplied paths/URLs accepted)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = Path(__file__).parents[3] / "ml_models" / "person" / "yolo11n.pt"

# Standard COCO class id for "person" in the stock Ultralytics COCO models.
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.40
DEFAULT_IOU = 0.50


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PersonDetectorError(Exception):
    """Base exception for PersonDetector failures."""


class InvalidFrameError(PersonDetectorError):
    """Raised when the supplied frame fails input validation."""


class InferenceError(PersonDetectorError):
    """Raised when YOLO inference itself fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
# PersonDetector
# ---------------------------------------------------------------------------
class PersonDetector:
    """
    Loads a local YOLO11n COCO checkpoint and exposes the VideoDetector
    contract's `.detect(frame, config)` API, filtered to person-only
    detections.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> None:
        self._model_path = Path(model_path)
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou

        self._verify_checkpoint_exists()
        self._model = self._load_model()

        logger.info(
            "PersonDetector initialized: model=%s imgsz=%d conf=%.2f iou=%.2f",
            self._model_path.name,
            self._imgsz,
            self._conf,
            self._iou,
        )

    # -- initialization steps -------------------------------------------------
    def _verify_checkpoint_exists(self) -> None:
        if not self._model_path.exists() or not self._model_path.is_file():
            logger.error("Configured person model checkpoint is missing.")
            raise PersonDetectorError(
                "Person model checkpoint not found at configured path. "
                "No automatic download is performed."
            )

    def _load_model(self) -> YOLO:
        try:
            return YOLO(str(self._model_path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load person model checkpoint.")
            raise PersonDetectorError(
                "Failed to load person model checkpoint."
            ) from exc

    # -- public API (VideoDetector contract) -------------------------------------
    def detect(self, frame: np.ndarray, config: Any = None) -> List[Detection]:
        """
        Run person detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            HxWx3 image array (e.g. as read by OpenCV, BGR or RGB).
        config : Any, optional
            Unused placeholder to satisfy the VideoDetector contract's
            signature. PersonDetector currently uses its own fixed
            imgsz/conf/iou set at construction time.

        Returns
        -------
        List[Detection]
            One entry per detected person above the confidence threshold.
            All non-person COCO classes are discarded. Empty list if there
            are no person detections.
        """
        _validate_frame(frame)

        try:
            results = self._model.predict(
                frame,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Person detection inference failed.")
            raise InferenceError("Person detection inference failed.") from exc

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
            # Defense in depth: the `classes=[PERSON_CLASS_ID]` predict()
            # argument should already restrict output to person, but we
            # still filter explicitly so a Detection is never emitted for
            # any other class even if that upstream filter behaves
            # unexpectedly.
            if int(cls) != PERSON_CLASS_ID:
                continue
            detections.append(
                Detection(
                    bbox=[float(v) for v in box],
                    class_name=PERSON_CLASS_NAME,
                    confidence=float(confidence),
                )
            )

        logger.info("PersonDetector.detect: %d person detection(s)", len(detections))
        return detections
