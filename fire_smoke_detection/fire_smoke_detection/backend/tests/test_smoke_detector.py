"""
Unit tests for SmokeDetector.

Mocks the underlying YOLO model so tests run with no real inference and no
internet access, mirroring the pattern used in tests/test_weapon_detector.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.interfaces.video_detector import VideoDetector
from backend.app.modules.video.smoke_detector import (
    SmokeDetector,
    SmokeDetectorError,
    ModelHashMismatchError,
    ModelClassMappingError,
    InvalidFrameError,
    InferenceError,
    EXPECTED_CLASS_MAP,
    APPROVED_MODEL_SHA256,
    SMOKE_CLASS_ID,
    SMOKE_CLASS_NAME,
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
def real_checkpoint_bytes(tmp_path):
    p = tmp_path / "best-fire.pt"
    p.write_bytes(b"fire-smoke-checkpoint-bytes")
    return p


@pytest.fixture
def mock_yolo_valid():
    with patch("backend.app.modules.video.smoke_detector.YOLO") as mock_cls:
        instance = MagicMock()
        instance.names = dict(EXPECTED_CLASS_MAP)
        mock_cls.return_value = instance
        yield mock_cls, instance


@pytest.fixture
def detector(mock_yolo_valid, real_checkpoint_bytes):
    import hashlib

    actual_hash = hashlib.sha256(real_checkpoint_bytes.read_bytes()).hexdigest()
    return SmokeDetector(model_path=real_checkpoint_bytes, expected_sha256=actual_hash)


# ---------------------------------------------------------------------------
# Checkpoint existence / hash / class-map validation
# ---------------------------------------------------------------------------
def test_missing_checkpoint_raises(mock_yolo_valid, tmp_path):
    missing_path = tmp_path / "does_not_exist.pt"
    with pytest.raises(SmokeDetectorError):
        SmokeDetector(model_path=missing_path, expected_sha256=APPROVED_MODEL_SHA256)


def test_hash_mismatch_raises(mock_yolo_valid, real_checkpoint_bytes):
    with pytest.raises(ModelHashMismatchError):
        SmokeDetector(
            model_path=real_checkpoint_bytes,
            expected_sha256="0" * 64,
        )


def test_hash_match_allows_load(detector):
    assert detector is not None


def test_wrong_class_mapping_raises(mock_yolo_valid, real_checkpoint_bytes):
    import hashlib

    _mock_cls, instance = mock_yolo_valid
    instance.names = {0: "not_smoke", 1: "fire"}
    actual_hash = hashlib.sha256(real_checkpoint_bytes.read_bytes()).hexdigest()
    with pytest.raises(ModelClassMappingError):
        SmokeDetector(model_path=real_checkpoint_bytes, expected_sha256=actual_hash)


def test_no_network_fallback_on_load_failure(mock_yolo_valid, real_checkpoint_bytes):
    import hashlib

    mock_cls, _instance = mock_yolo_valid
    mock_cls.side_effect = RuntimeError("simulated load failure")
    actual_hash = hashlib.sha256(real_checkpoint_bytes.read_bytes()).hexdigest()
    with pytest.raises(SmokeDetectorError):
        SmokeDetector(model_path=real_checkpoint_bytes, expected_sha256=actual_hash)


# ---------------------------------------------------------------------------
# detect(): input validation
# ---------------------------------------------------------------------------
def test_detect_none_frame_raises(detector):
    with pytest.raises(InvalidFrameError):
        detector.detect(None)


def test_detect_wrong_dims_raises(detector):
    with pytest.raises(InvalidFrameError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))


# ---------------------------------------------------------------------------
# detect(): empty detections
# ---------------------------------------------------------------------------
def test_detect_empty_result_returns_empty_list(detector, valid_frame):
    detector._model.predict.return_value = [_make_fake_result([], [], [])]
    assert detector.detect(valid_frame) == []


def test_detect_none_boxes_returns_empty_list(detector, valid_frame):
    fake_result = MagicMock()
    fake_result.boxes = None
    detector._model.predict.return_value = [fake_result]
    assert detector.detect(valid_frame) == []


# ---------------------------------------------------------------------------
# detect(): smoke filtering / conversion
# ---------------------------------------------------------------------------
def test_detect_smoke_converts_to_detection(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[10.0, 20.0, 30.0, 40.0]], confs=[0.71], clses=[SMOKE_CLASS_ID]
        )
    ]
    result = detector.detect(valid_frame)
    assert len(result) == 1
    assert result[0].class_name == SMOKE_CLASS_NAME
    assert result[0].bbox == [10.0, 20.0, 30.0, 40.0]
    assert result[0].confidence == pytest.approx(0.71)


def test_detect_filters_fire_from_mixed_results(detector, valid_frame):
    # Simulate a result containing both smoke (class 0) and fire (class 1);
    # only smoke should survive.
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 20.0, 20.0]],
            confs=[0.9, 0.6],
            clses=[SMOKE_CLASS_ID, 1],  # smoke, fire
        )
    ]
    result = detector.detect(valid_frame)
    assert len(result) == 1
    assert result[0].class_name == "smoke"
    assert result[0].bbox == [0.0, 0.0, 5.0, 5.0]


def test_detect_all_fire_returns_empty(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(xyxy=[[0.0, 0.0, 5.0, 5.0]], confs=[0.9], clses=[1])
    ]
    assert detector.detect(valid_frame) == []


def test_bbox_and_confidence_preserved_exactly(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[1.5, 2.5, 300.25, 400.75]], confs=[0.4321], clses=[SMOKE_CLASS_ID]
        )
    ]
    result = detector.detect(valid_frame)
    assert result[0].bbox == [1.5, 2.5, 300.25, 400.75]
    assert result[0].confidence == pytest.approx(0.4321)


def test_predict_called_with_smoke_class_filter(detector, valid_frame):
    detector._model.predict.return_value = [_make_fake_result([], [], [])]
    detector.detect(valid_frame)
    _, kwargs = detector._model.predict.call_args
    assert kwargs["classes"] == [SMOKE_CLASS_ID]


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
def test_smoke_detector_satisfies_video_detector_protocol(detector):
    assert isinstance(detector, VideoDetector)


def test_detect_accepts_frame_and_config_positional(detector, valid_frame):
    detector._model.predict.return_value = [_make_fake_result([], [], [])]
    result = detector.detect(valid_frame, config={"some": "config"})
    assert result == []
