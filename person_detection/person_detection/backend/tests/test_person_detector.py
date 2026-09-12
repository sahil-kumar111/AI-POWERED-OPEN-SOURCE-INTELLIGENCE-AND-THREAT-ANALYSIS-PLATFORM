"""
Unit tests for PersonDetector.

Mocks the underlying YOLO model so tests run with no real inference and no
internet access, mirroring the pattern used in tests/test_weapon_detector.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Allow `backend.app...` imports when run directly from the module root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.interfaces.types import Detection
from backend.app.interfaces.video_detector import VideoDetector
from backend.app.modules.video.person_detector import (
    PersonDetector,
    PersonDetectorError,
    InvalidFrameError,
    InferenceError,
    PERSON_CLASS_ID,
    PERSON_CLASS_NAME,
    DEFAULT_MODEL_PATH,
)


def _make_fake_boxes(xyxy, confs, clses):
    fake = MagicMock()
    fake.xyxy = MagicMock(tolist=lambda: xyxy)
    fake.conf = MagicMock(tolist=lambda: confs)
    fake.cls = MagicMock(tolist=lambda: clses)
    fake.__len__ = lambda self=fake: len(xyxy)
    return fake


def _make_fake_result(xyxy, confs, clses):
    result = MagicMock()
    result.boxes = _make_fake_boxes(xyxy, confs, clses)
    return result


@pytest.fixture
def valid_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_yolo():
    """Patch ultralytics.YOLO (as imported into person_detector) so
    PersonDetector loads a fake model without touching disk/network."""
    with patch("backend.app.modules.video.person_detector.YOLO") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield mock_cls, instance


@pytest.fixture
def detector(mock_yolo, tmp_path):
    """A PersonDetector pointed at a dummy checkpoint file so the
    existence check passes, with YOLO itself mocked."""
    fake_checkpoint = tmp_path / "yolo11n.pt"
    fake_checkpoint.write_bytes(b"not a real checkpoint")
    return PersonDetector(model_path=fake_checkpoint)


# ---------------------------------------------------------------------------
# Construction / checkpoint handling
# ---------------------------------------------------------------------------
def test_missing_checkpoint_raises(mock_yolo, tmp_path):
    missing_path = tmp_path / "does_not_exist.pt"
    with pytest.raises(PersonDetectorError):
        PersonDetector(model_path=missing_path)


def test_default_model_path_points_at_local_checkpoint():
    assert DEFAULT_MODEL_PATH.name == "yolo11n.pt"
    assert "ml_models" in DEFAULT_MODEL_PATH.parts
    assert "person" in DEFAULT_MODEL_PATH.parts


def test_no_network_fallback_on_load_failure(mock_yolo, tmp_path):
    """If YOLO() itself raises (e.g. corrupt file), PersonDetector must
    surface an error rather than silently downloading a substitute."""
    mock_cls, _instance = mock_yolo
    mock_cls.side_effect = RuntimeError("simulated load failure")
    fake_checkpoint = tmp_path / "yolo11n.pt"
    fake_checkpoint.write_bytes(b"not a real checkpoint")
    with pytest.raises(PersonDetectorError):
        PersonDetector(model_path=fake_checkpoint)


# ---------------------------------------------------------------------------
# detect(): input validation
# ---------------------------------------------------------------------------
def test_detect_none_frame_raises(detector):
    with pytest.raises(InvalidFrameError):
        detector.detect(None)


def test_detect_non_ndarray_raises(detector):
    with pytest.raises(InvalidFrameError):
        detector.detect([[1, 2, 3]])


def test_detect_wrong_dims_raises(detector):
    with pytest.raises(InvalidFrameError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))


def test_detect_wrong_channels_raises(detector):
    with pytest.raises(InvalidFrameError):
        detector.detect(np.zeros((10, 10, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# detect(): empty detections
# ---------------------------------------------------------------------------
def test_detect_empty_result_returns_empty_list(detector, valid_frame):
    _mock_cls, instance = detector._model, None  # noqa: F841 (documented below)
    detector._model.predict.return_value = [_make_fake_result([], [], [])]
    result = detector.detect(valid_frame)
    assert result == []


def test_detect_no_results_returns_empty_list(detector, valid_frame):
    detector._model.predict.return_value = []
    result = detector.detect(valid_frame)
    assert result == []


def test_detect_none_boxes_returns_empty_list(detector, valid_frame):
    fake_result = MagicMock()
    fake_result.boxes = None
    detector._model.predict.return_value = [fake_result]
    result = detector.detect(valid_frame)
    assert result == []


# ---------------------------------------------------------------------------
# detect(): person detection -> Detection conversion
# ---------------------------------------------------------------------------
def test_detect_person_converts_to_detection(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[10.0, 20.0, 30.0, 40.0]],
            confs=[0.87],
            clses=[PERSON_CLASS_ID],
        )
    ]
    result = detector.detect(valid_frame)
    assert len(result) == 1
    d = result[0]
    assert isinstance(d, Detection)
    assert d.class_name == PERSON_CLASS_NAME
    assert d.bbox == [10.0, 20.0, 30.0, 40.0]
    assert d.confidence == pytest.approx(0.87)


def test_detect_bbox_and_confidence_preserved_exactly(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[1.5, 2.5, 300.25, 400.75]],
            confs=[0.6789],
            clses=[PERSON_CLASS_ID],
        )
    ]
    result = detector.detect(valid_frame)
    assert result[0].bbox == [1.5, 2.5, 300.25, 400.75]
    assert result[0].confidence == pytest.approx(0.6789)


def test_detect_multiple_persons(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]],
            confs=[0.9, 0.55],
            clses=[PERSON_CLASS_ID, PERSON_CLASS_ID],
        )
    ]
    result = detector.detect(valid_frame)
    assert len(result) == 2
    assert all(d.class_name == PERSON_CLASS_NAME for d in result)


# ---------------------------------------------------------------------------
# detect(): non-person classes are filtered
# ---------------------------------------------------------------------------
def test_detect_filters_non_person_classes(detector, valid_frame):
    # Simulate a result containing a person (class 0) plus a car (class 2)
    # and a dog (class 16) -- only the person should survive, exercising the
    # detector's own defense-in-depth filter regardless of what the
    # `classes=[PERSON_CLASS_ID]` predict() argument already restricted.
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 50.0, 50.0]],
            confs=[0.9, 0.8, 0.7],
            clses=[PERSON_CLASS_ID, 2, 16],
        )
    ]
    result = detector.detect(valid_frame)
    assert len(result) == 1
    assert result[0].class_name == PERSON_CLASS_NAME
    assert result[0].bbox == [0.0, 0.0, 5.0, 5.0]


def test_detect_all_non_person_returns_empty(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[0.0, 0.0, 5.0, 5.0]],
            confs=[0.9],
            clses=[2],  # car
        )
    ]
    result = detector.detect(valid_frame)
    assert result == []


def test_predict_called_with_person_class_filter(detector, valid_frame):
    detector._model.predict.return_value = [_make_fake_result([], [], [])]
    detector.detect(valid_frame)
    _, kwargs = detector._model.predict.call_args
    assert kwargs["classes"] == [PERSON_CLASS_ID]


# ---------------------------------------------------------------------------
# detect(): inference failure
# ---------------------------------------------------------------------------
def test_inference_failure_raises_inference_error(detector, valid_frame):
    detector._model.predict.side_effect = RuntimeError("simulated inference failure")
    with pytest.raises(InferenceError):
        detector.detect(valid_frame)


# ---------------------------------------------------------------------------
# VideoDetector contract compatibility
# ---------------------------------------------------------------------------
def test_person_detector_satisfies_video_detector_protocol(detector):
    assert isinstance(detector, VideoDetector)


def test_detect_accepts_frame_and_config_positional(detector, valid_frame):
    detector._model.predict.return_value = [_make_fake_result([], [], [])]
    # VideoDetector.detect(frame, config) -- config is accepted (and
    # currently unused) so PersonDetector is structurally compatible.
    result = detector.detect(valid_frame, config={"some": "config"})
    assert result == []
