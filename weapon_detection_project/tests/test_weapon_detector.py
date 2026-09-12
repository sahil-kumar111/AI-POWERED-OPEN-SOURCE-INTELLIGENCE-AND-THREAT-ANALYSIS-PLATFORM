"""
Unit + integration tests for WeaponDetector.

Unit tests mock the underlying YOLO model so they run with no real inference
and no internet access. Integration tests are marked and use the real
verified checkpoint + a local CC0 fixture image only.
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from weapon_detector import (
    WeaponDetector,
    WeaponDetection,
    WeaponDetectorError,
    ModelHashMismatchError,
    ModelClassMappingError,
    InvalidFrameError,
    InferenceError,
    EXPECTED_CLASS_MAP,
    APPROVED_MODEL_SHA256,
    DEFAULT_MODEL_PATH,
    DEFAULT_IOU,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
M1911_IMAGE = FIXTURES_DIR / "M1911_pistol.jpg"


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
def mock_yolo_valid():
    """Patch ultralytics.YOLO so WeaponDetector loads a fake, correctly
    configured model without touching disk/network."""
    with patch("weapon_detector.YOLO") as mock_cls:
        instance = MagicMock()
        instance.names = dict(EXPECTED_CLASS_MAP)
        mock_cls.return_value = instance
        yield mock_cls, instance


# ---------------------------------------------------------------------------
# 1. Detector initialization / valid model loading
# ---------------------------------------------------------------------------
def test_initialization_success(tmp_path, mock_yolo_valid):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"fake-weights")
    expected_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    detector = WeaponDetector(model_path=model_path, expected_sha256=expected_hash)

    assert detector is not None
    mock_cls, instance = mock_yolo_valid
    mock_cls.assert_called_once_with(str(model_path))


# ---------------------------------------------------------------------------
# 3. Missing model failure
# ---------------------------------------------------------------------------
def test_missing_model_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.pt"
    with pytest.raises(WeaponDetectorError):
        WeaponDetector(model_path=missing_path, expected_sha256="deadbeef")


# ---------------------------------------------------------------------------
# 2 / 5. Model hash verification
# ---------------------------------------------------------------------------
def test_hash_mismatch_raises(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"some-weights")
    wrong_hash = "0" * 64

    with pytest.raises(ModelHashMismatchError):
        WeaponDetector(model_path=model_path, expected_sha256=wrong_hash)


def test_hash_match_allows_load(tmp_path, mock_yolo_valid):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"correct-weights")
    correct_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    detector = WeaponDetector(model_path=model_path, expected_sha256=correct_hash)
    assert detector is not None


# ---------------------------------------------------------------------------
# 4. Invalid model configuration (class mapping mismatch)
# ---------------------------------------------------------------------------
def test_wrong_class_mapping_raises(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")
    correct_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    with patch("weapon_detector.YOLO") as mock_cls:
        instance = MagicMock()
        instance.names = {0: "guns", 1: "knife"}  # wrong mapping, wrong size
        mock_cls.return_value = instance

        with pytest.raises(ModelClassMappingError):
            WeaponDetector(model_path=model_path, expected_sha256=correct_hash)


# ---------------------------------------------------------------------------
# 12 / 13. Invalid input handling / safe error behavior
# ---------------------------------------------------------------------------
def test_detect_none_frame_raises(tmp_path, mock_yolo_valid):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(InvalidFrameError):
        detector.detect(None)


def test_detect_empty_frame_raises(tmp_path, mock_yolo_valid):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(InvalidFrameError):
        detector.detect(np.array([]))


def test_detect_wrong_dims_raises(tmp_path, mock_yolo_valid):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(InvalidFrameError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))  # missing channel dim


def test_detect_wrong_channels_raises(tmp_path, mock_yolo_valid):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(InvalidFrameError):
        detector.detect(np.zeros((10, 10, 4), dtype=np.uint8))  # 4 channels


def test_detect_non_ndarray_raises(tmp_path, mock_yolo_valid):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(InvalidFrameError):
        detector.detect([[1, 2, 3]])  # not a numpy array


# ---------------------------------------------------------------------------
# 6 / 7. Valid frame inference, zero-detection frame
# ---------------------------------------------------------------------------
def test_detect_zero_detections(tmp_path, mock_yolo_valid, valid_frame):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    mock_cls, instance = mock_yolo_valid
    instance.predict.return_value = [_make_fake_result([], [], [])]

    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    result = detector.detect(valid_frame)
    assert result == []


# ---------------------------------------------------------------------------
# 8 / 9 / 10 / 11. YOLO result -> Detection conversion, all 5 classes,
# confidence propagation, bbox propagation
# ---------------------------------------------------------------------------
def test_detect_all_five_classes_conversion(tmp_path, mock_yolo_valid, valid_frame):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    mock_cls, instance = mock_yolo_valid

    xyxy = [
        [10.0, 20.0, 30.0, 40.0],
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    confs = [0.91, 0.55, 0.33, 0.72, 0.60]
    clses = [0, 1, 2, 3, 4]  # pistol, knife, rifle, shotgun, smg
    instance.predict.return_value = [_make_fake_result(xyxy, confs, clses)]

    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    detections = detector.detect(valid_frame)

    assert len(detections) == 5
    names = {d.class_name for d in detections}
    assert names == {"pistol", "knife", "rifle", "shotgun", "smg"}

    pistol_det = next(d for d in detections if d.class_name == "pistol")
    assert pistol_det.class_id == 0
    assert pistol_det.confidence == pytest.approx(0.91)
    assert pistol_det.bbox == [10.0, 20.0, 30.0, 40.0]
    assert isinstance(pistol_det, WeaponDetection)


# ---------------------------------------------------------------------------
# Confidence filtering is delegated to YOLO's `conf=` kwarg — verify it's passed
# ---------------------------------------------------------------------------
def test_detect_passes_confidence_and_imgsz_to_model(
    tmp_path, mock_yolo_valid, valid_frame
):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    mock_cls, instance = mock_yolo_valid
    instance.predict.return_value = [_make_fake_result([], [], [])]

    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        imgsz=640,
        conf=0.40,
        iou=0.50,
    )
    detector.detect(valid_frame)

    _, kwargs = instance.predict.call_args
    assert kwargs["imgsz"] == 640
    assert kwargs["conf"] == 0.40
    assert kwargs["iou"] == 0.50
    assert kwargs["verbose"] is False


def test_default_iou_is_050(tmp_path, mock_yolo_valid, valid_frame):
    """NMS investigation result: default iou must be 0.50, not Ultralytics'
    unset default of 0.70, which was shown to let a near-duplicate box
    (IoU 0.66 with the top detection) survive on the M1911 fixture."""
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    mock_cls, instance = mock_yolo_valid
    instance.predict.return_value = [_make_fake_result([], [], [])]

    assert DEFAULT_IOU == 0.50

    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    detector.detect(valid_frame)

    _, kwargs = instance.predict.call_args
    assert kwargs["iou"] == 0.50


# ---------------------------------------------------------------------------
# Inference failure handling
# ---------------------------------------------------------------------------
def test_inference_failure_raises_inference_error(
    tmp_path, mock_yolo_valid, valid_frame
):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    mock_cls, instance = mock_yolo_valid
    instance.predict.side_effect = RuntimeError("boom")

    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(InferenceError):
        detector.detect(valid_frame)


# ---------------------------------------------------------------------------
# 14. Interface compatibility: detector = WeaponDetector(); detector.detect(frame)
# ---------------------------------------------------------------------------
def test_public_api_shape(tmp_path, mock_yolo_valid, valid_frame):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"w")
    mock_cls, instance = mock_yolo_valid
    instance.predict.return_value = [
        _make_fake_result([[1.0, 2.0, 3.0, 4.0]], [0.8], [0])
    ]

    detector = WeaponDetector(
        model_path=model_path,
        expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    detections = detector.detect(valid_frame)
    assert isinstance(detections, list)
    d = detections[0]
    assert hasattr(d, "class_name")
    assert hasattr(d, "class_id")
    assert hasattr(d, "confidence")
    assert hasattr(d, "bbox")


# ---------------------------------------------------------------------------
# REAL INTEGRATION TEST — actual checkpoint, actual fixture image, no mocking.
# Skipped automatically if the approved checkpoint or fixture is absent.
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(), reason="Approved checkpoint not present"
)
@pytest.mark.skipif(
    not M1911_IMAGE.exists(), reason="M1911 fixture image not present"
)
def test_real_inference_on_m1911_fixture():
    import cv2

    detector = WeaponDetector()  # uses real default path + real approved hash
    frame = cv2.imread(str(M1911_IMAGE))
    assert frame is not None, "Failed to read fixture image"

    detections = detector.detect(frame)

    assert len(detections) >= 1, "Expected at least one detection on M1911 fixture"
    pistol_detections = [d for d in detections if d.class_name == "pistol"]
    assert pistol_detections, "Expected a pistol detection on the M1911 fixture"

    top = max(pistol_detections, key=lambda d: d.confidence)
    assert 0.0 < top.confidence <= 1.0
    assert len(top.bbox) == 4
