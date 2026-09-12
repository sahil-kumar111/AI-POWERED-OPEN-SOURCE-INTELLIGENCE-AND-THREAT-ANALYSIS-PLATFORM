"""
Focused tests for the restored project contracts:
- Detection
- VideoDetector (Protocol)
- Tracker / TrackedDetection (Protocol)

These tests only check the shape/behavior of the contracts themselves.
They do not test any concrete detector or tracker implementation.
"""

import sys
from pathlib import Path

# Allow `backend.app...` imports when run directly from the module root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from backend.app.interfaces.types import Detection
from backend.app.interfaces.video_detector import VideoDetector
from backend.app.interfaces.tracker import Tracker, TrackedDetection


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_detection_has_exact_fields():
    d = Detection(bbox=[1.0, 2.0, 3.0, 4.0], class_name="pistol", confidence=0.91)
    assert d.bbox == [1.0, 2.0, 3.0, 4.0]
    assert d.class_name == "pistol"
    assert d.confidence == pytest.approx(0.91)


def test_detection_is_frozen():
    d = Detection(bbox=[0.0, 0.0, 1.0, 1.0], class_name="knife", confidence=0.5)
    with pytest.raises(Exception):
        d.confidence = 0.9  # dataclass(frozen=True) must reject mutation


def test_detection_fields_match_weapon_detection_shape():
    """WeaponDetection (in weapon_detector.py) is not wired to Detection yet,
    but its bbox/class_name/confidence fields must be compatible so a future
    adapter can construct a Detection from a WeaponDetection without loss."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from weapon_detector import WeaponDetection

    wd = WeaponDetection(
        class_name="rifle", class_id=2, confidence=0.77, bbox=[1.0, 2.0, 3.0, 4.0]
    )
    d = Detection(bbox=wd.bbox, class_name=wd.class_name, confidence=wd.confidence)
    assert d.bbox == wd.bbox
    assert d.class_name == wd.class_name
    assert d.confidence == wd.confidence


# ---------------------------------------------------------------------------
# VideoDetector Protocol
# ---------------------------------------------------------------------------
class _FakeDetector:
    """Minimal structural implementation of VideoDetector for contract testing."""

    def detect(self, frame, config):
        return [Detection(bbox=[0.0, 0.0, 1.0, 1.0], class_name="person", confidence=0.9)]


def test_conforming_detector_satisfies_protocol():
    detector = _FakeDetector()
    assert isinstance(detector, VideoDetector)


def test_non_conforming_object_does_not_satisfy_protocol():
    class NotADetector:
        pass

    assert not isinstance(NotADetector(), VideoDetector)


def test_fake_detector_returns_list_of_detection():
    detector = _FakeDetector()
    result = detector.detect(frame=None, config=None)
    assert isinstance(result, list)
    assert all(isinstance(d, Detection) for d in result)


# ---------------------------------------------------------------------------
# Tracker Protocol / TrackedDetection
# ---------------------------------------------------------------------------
class _FakeTracker:
    """Minimal structural implementation of Tracker for contract testing."""

    def __init__(self):
        self._next_id = 1

    def update(self, detections, frame=None):
        out = []
        for d in detections:
            out.append(
                TrackedDetection(
                    bbox=d.bbox,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    track_id=self._next_id,
                )
            )
            self._next_id += 1
        return out

    def reset(self):
        self._next_id = 1


def test_conforming_tracker_satisfies_protocol():
    assert isinstance(_FakeTracker(), Tracker)


def test_non_conforming_object_does_not_satisfy_tracker_protocol():
    class NotATracker:
        pass

    assert not isinstance(NotATracker(), Tracker)


def test_tracked_detection_preserves_detection_fields_plus_track_id():
    tracker = _FakeTracker()
    detections = [Detection(bbox=[1.0, 1.0, 2.0, 2.0], class_name="knife", confidence=0.6)]
    tracked = tracker.update(detections)
    assert len(tracked) == 1
    t = tracked[0]
    assert isinstance(t, TrackedDetection)
    assert t.bbox == detections[0].bbox
    assert t.class_name == detections[0].class_name
    assert t.confidence == detections[0].confidence
    assert isinstance(t.track_id, int)


def test_tracker_handles_empty_detections():
    tracker = _FakeTracker()
    assert tracker.update([]) == []


def test_tracked_detection_is_frozen():
    t = TrackedDetection(bbox=[0.0, 0.0, 1.0, 1.0], class_name="smg", confidence=0.8, track_id=1)
    with pytest.raises(Exception):
        t.track_id = 2
