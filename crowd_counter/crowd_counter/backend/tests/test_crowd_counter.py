"""
Unit tests for CrowdCounter.

Pure reduction over List[Detection] -- no model, no mocking required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.interfaces.types import Detection
from backend.app.modules.video.crowd_counter import CrowdCounter


def _person(bbox=None, confidence=0.9):
    return Detection(bbox=bbox or [0.0, 0.0, 1.0, 1.0], class_name="person", confidence=confidence)


def _other(class_name, bbox=None, confidence=0.9):
    return Detection(bbox=bbox or [0.0, 0.0, 1.0, 1.0], class_name=class_name, confidence=confidence)


def test_zero_detections_returns_zero():
    counter = CrowdCounter()
    assert counter.count([]) == 0


def test_empty_list_returns_zero():
    counter = CrowdCounter()
    detections: list = []
    assert counter.count(detections) == 0


def test_one_person_returns_one():
    counter = CrowdCounter()
    assert counter.count([_person()]) == 1


def test_multiple_people_correct_count():
    counter = CrowdCounter()
    detections = [_person(), _person(), _person(), _person()]
    assert counter.count(detections) == 4


def test_non_person_detections_ignored():
    counter = CrowdCounter()
    detections = [_other("pistol"), _other("knife"), _other("fire"), _other("smoke")]
    assert counter.count(detections) == 0


def test_mixed_person_and_weapon_fire_smoke_only_people_counted():
    counter = CrowdCounter()
    detections = [
        _person(),
        _other("pistol"),
        _person(),
        _other("fire"),
        _other("smoke"),
        _person(),
        _other("knife"),
    ]
    assert counter.count(detections) == 3


def test_input_detections_not_mutated():
    counter = CrowdCounter()
    detections = [_person(), _other("rifle"), _person()]
    original_ids = [id(d) for d in detections]
    original_values = list(detections)

    result = counter.count(detections)

    assert result == 2
    # Same list contents, same objects, same order -- nothing mutated or reordered.
    assert [id(d) for d in detections] == original_ids
    assert detections == original_values
    assert len(detections) == 3


def test_count_returns_plain_int():
    counter = CrowdCounter()
    result = counter.count([_person(), _person()])
    assert isinstance(result, int)
