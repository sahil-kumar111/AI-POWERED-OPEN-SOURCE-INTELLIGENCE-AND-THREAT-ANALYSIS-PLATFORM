"""
Unit tests for FireDetector.

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
from backend.app.modules.video.fire_detector import (
    FireDetector,
    FireDetectorError,
    ModelHashMismatchError,
    ModelClassMappingError,
    InvalidFrameError,
    InferenceError,
    EXPECTED_CLASS_MAP,
    APPROVED_MODEL_SHA256,
    FIRE_CLASS_ID,
    FIRE_CLASS_NAME,
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
    """A dummy file whose bytes we control, so hash checks are testable."""
    p = tmp_path / "best-fire.pt"
    p.write_bytes(b"fire-smoke-checkpoint-bytes")
    return p


@pytest.fixture
def mock_yolo_valid():
    """Patch ultralytics.YOLO so FireDetector loads a fake, correctly
    configured model without touching disk/network."""
    with patch("backend.app.modules.video.fire_detector.YOLO") as mock_cls:
        instance = MagicMock()
        instance.names = dict(EXPECTED_CLASS_MAP)
        mock_cls.return_value = instance
        yield mock_cls, instance


@pytest.fixture
def detector(mock_yolo_valid, real_checkpoint_bytes):
    """A FireDetector pointed at a dummy checkpoint whose hash matches
    what we tell the detector to expect, with YOLO mocked."""
    import hashlib

    actual_hash = hashlib.sha256(real_checkpoint_bytes.read_bytes()).hexdigest()
    return FireDetector(model_path=real_checkpoint_bytes, expected_sha256=actual_hash)


# ---------------------------------------------------------------------------
# Checkpoint existence / hash / class-map validation
# ---------------------------------------------------------------------------
def test_missing_checkpoint_raises(mock_yolo_valid, tmp_path):
    missing_path = tmp_path / "does_not_exist.pt"
    with pytest.raises(FireDetectorError):
        FireDetector(model_path=missing_path, expected_sha256=APPROVED_MODEL_SHA256)


def test_hash_mismatch_raises(mock_yolo_valid, real_checkpoint_bytes):
    with pytest.raises(ModelHashMismatchError):
        FireDetector(
            model_path=real_checkpoint_bytes,
            expected_sha256="0" * 64,  # deliberately wrong
        )


def test_hash_match_allows_load(detector):
    assert detector is not None


def test_wrong_class_mapping_raises(mock_yolo_valid, real_checkpoint_bytes):
    import hashlib

    _mock_cls, instance = mock_yolo_valid
    instance.names = {0: "smoke", 1: "not_fire"}  # tampered mapping
    actual_hash = hashlib.sha256(real_checkpoint_bytes.read_bytes()).hexdigest()
    with pytest.raises(ModelClassMappingError):
        FireDetector(model_path=real_checkpoint_bytes, expected_sha256=actual_hash)


def test_no_network_fallback_on_load_failure(mock_yolo_valid, real_checkpoint_bytes):
    import hashlib

    mock_cls, _instance = mock_yolo_valid
    mock_cls.side_effect = RuntimeError("simulated load failure")
    actual_hash = hashlib.sha256(real_checkpoint_bytes.read_bytes()).hexdigest()
    with pytest.raises(FireDetectorError):
        FireDetector(model_path=real_checkpoint_bytes, expected_sha256=actual_hash)


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
# detect(): fire filtering / conversion
# ---------------------------------------------------------------------------
def test_detect_fire_converts_to_detection(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[10.0, 20.0, 30.0, 40.0]], confs=[0.83], clses=[FIRE_CLASS_ID]
        )
    ]
    result = detector.detect(valid_frame)
    assert len(result) == 1
    assert result[0].class_name == FIRE_CLASS_NAME
    assert result[0].bbox == [10.0, 20.0, 30.0, 40.0]
    assert result[0].confidence == pytest.approx(0.83)


def test_detect_filters_smoke_from_mixed_results(detector, valid_frame):
    # Simulate a result containing both fire (class 1) and smoke (class 0);
    # only fire should survive.
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 20.0, 20.0]],
            confs=[0.9, 0.6],
            clses=[FIRE_CLASS_ID, 0],  # fire, smoke
        )
    ]
    result = detector.detect(valid_frame)
    assert len(result) == 1
    assert result[0].class_name == "fire"
    assert result[0].bbox == [0.0, 0.0, 5.0, 5.0]


def test_detect_all_smoke_returns_empty(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(xyxy=[[0.0, 0.0, 5.0, 5.0]], confs=[0.9], clses=[0])
    ]
    assert detector.detect(valid_frame) == []


def test_bbox_and_confidence_preserved_exactly(detector, valid_frame):
    detector._model.predict.return_value = [
        _make_fake_result(
            xyxy=[[1.5, 2.5, 300.25, 400.75]], confs=[0.6789], clses=[FIRE_CLASS_ID]
        )
    ]
    result = detector.detect(valid_frame)
    assert result[0].bbox == [1.5, 2.5, 300.25, 400.75]
    assert result[0].confidence == pytest.approx(0.6789)


def test_predict_called_with_fire_class_filter(detector, valid_frame):
    detector._model.predict.return_value = [_make_fake_result([], [], [])]
    detector.detect(valid_frame)
    _, kwargs = detector._model.predict.call_args
    assert kwargs["classes"] == [FIRE_CLASS_ID]


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
def test_fire_detector_satisfies_video_detector_protocol(detector):
    assert isinstance(detector, VideoDetector)


def test_detect_accepts_frame_and_config_positional(detector, valid_frame):
    detector._model.predict.return_value = [_make_fake_result([], [], [])]
    result = detector.detect(valid_frame, config={"some": "config"})
    assert result == []
