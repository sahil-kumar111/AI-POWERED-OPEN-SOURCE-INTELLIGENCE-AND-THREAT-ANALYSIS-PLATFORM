"""
Tests for ByteTrackAdapter.

Uses the REAL Ultralytics BYTETracker (installed in this environment) with
synthetic Detection objects -- no detector model inference is required or
used. These tests exercise the actual ByteTrack algorithm end-to-end
through this project's Tracker contract.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.interfaces.types import Detection
from backend.app.interfaces.tracker import Tracker, TrackedDetection
from backend.app.modules.video.byte_tracker import ByteTrackAdapter


def _det(bbox, class_name="person", confidence=0.9):
    return Detection(bbox=list(bbox), class_name=class_name, confidence=confidence)


def _shift(bbox, dx=0.0, dy=0.0):
    x1, y1, x2, y2 = bbox
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


# ---------------------------------------------------------------------------
# 1. Empty detections
# ---------------------------------------------------------------------------
def test_update_empty_detections_does_not_crash():
    tracker = ByteTrackAdapter()
    result = tracker.update([])
    assert result == []


def test_update_empty_after_prior_activity_does_not_crash():
    tracker = ByteTrackAdapter()
    tracker.update([_det([10, 10, 50, 50])])
    result = tracker.update([])
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 2. Single object
# ---------------------------------------------------------------------------
def test_single_object_receives_track_id():
    tracker = ByteTrackAdapter()
    result = tracker.update([_det([10, 10, 50, 50])])
    assert len(result) == 1
    assert isinstance(result[0], TrackedDetection)
    assert isinstance(result[0].track_id, int)
    assert result[0].track_id > 0


# ---------------------------------------------------------------------------
# 3. Multiple objects
# ---------------------------------------------------------------------------
def test_multiple_objects_receive_separate_ids():
    tracker = ByteTrackAdapter()
    result = tracker.update(
        [
            _det([10, 10, 50, 50]),
            _det([200, 200, 260, 260]),
            _det([400, 50, 460, 110]),
        ]
    )
    assert len(result) == 3
    ids = [t.track_id for t in result]
    assert len(set(ids)) == 3  # all distinct


# ---------------------------------------------------------------------------
# 4. Persistent IDs across consecutive frames
# ---------------------------------------------------------------------------
def test_same_object_keeps_same_track_id_across_frames():
    tracker = ByteTrackAdapter()
    frame1 = tracker.update([_det([10, 10, 50, 50])])
    assert len(frame1) == 1
    track_id = frame1[0].track_id

    # Small motion between frames, same object.
    frame2 = tracker.update([_det(_shift([10, 10, 50, 50], dx=2, dy=1))])
    assert len(frame2) == 1
    assert frame2[0].track_id == track_id

    frame3 = tracker.update([_det(_shift([10, 10, 50, 50], dx=4, dy=2))])
    assert len(frame3) == 1
    assert frame3[0].track_id == track_id


# ---------------------------------------------------------------------------
# 5. Separate objects receive different IDs
# ---------------------------------------------------------------------------
def test_spatially_separate_objects_get_different_ids():
    tracker = ByteTrackAdapter()
    result = tracker.update(
        [
            _det([0, 0, 20, 20]),
            _det([500, 500, 520, 520]),
        ]
    )
    assert len(result) == 2
    assert result[0].track_id != result[1].track_id


# ---------------------------------------------------------------------------
# 6. Class-aware tracking
# ---------------------------------------------------------------------------
def test_different_classes_do_not_share_track_pool_even_with_overlapping_boxes():
    tracker = ByteTrackAdapter()
    # Identical/heavily-overlapping bbox but different classes -- must not be
    # matched to each other or forced to share an ID.
    result = tracker.update(
        [
            _det([10, 10, 50, 50], class_name="person"),
            _det([10, 10, 50, 50], class_name="pistol"),
        ]
    )
    assert len(result) == 2
    by_class = {t.class_name: t for t in result}
    assert "person" in by_class and "pistol" in by_class
    assert by_class["person"].track_id != by_class["pistol"].track_id


def test_class_specific_ids_stay_globally_unique_even_when_created_lazily():
    """A class's tracker may be created many frames after another class's
    tracker already has active tracks. Ultralytics' BYTETracker.__init__
    resets a shared, process-global ID counter as a side effect; the adapter
    must prevent that from producing a colliding track_id across classes."""
    tracker = ByteTrackAdapter()
    person_out = tracker.update([_det([10, 10, 50, 50], class_name="person")])
    person_id = person_out[0].track_id

    # "fire" never appeared before this frame -- its tracker is created now,
    # well after the person tracker already has an active track.
    fire_out = tracker.update(
        [
            _det(_shift([10, 10, 50, 50], dx=1, dy=1), class_name="person"),
            _det([300, 300, 340, 340], class_name="fire"),
        ]
    )
    fire_track = [t for t in fire_out if t.class_name == "fire"][0]
    assert fire_track.track_id != person_id


def test_a_detection_cannot_inherit_a_different_class_track_id():
    """A person overlapping a pistol's box must never appear with the
    pistol's track_id (or vice versa) -- classes are matched in fully
    separate pools."""
    tracker = ByteTrackAdapter()
    frame1 = tracker.update([_det([10, 10, 50, 50], class_name="pistol")])
    pistol_id = frame1[0].track_id

    frame2 = tracker.update(
        [
            _det(_shift([10, 10, 50, 50], dx=1), class_name="pistol"),
            _det([10, 10, 50, 50], class_name="person"),  # same box, new class
        ]
    )
    by_class = {t.class_name: t for t in frame2}
    assert by_class["pistol"].track_id == pistol_id
    assert by_class["person"].track_id != pistol_id


# ---------------------------------------------------------------------------
# 7/8/9. bbox / confidence / class_name preservation
# ---------------------------------------------------------------------------
def test_bbox_preserved_exactly():
    tracker = ByteTrackAdapter()
    bbox = [12.5, 33.25, 88.75, 140.0]
    result = tracker.update([_det(bbox)])
    assert result[0].bbox == bbox


def test_confidence_preserved_exactly():
    tracker = ByteTrackAdapter()
    result = tracker.update([_det([10, 10, 50, 50], confidence=0.7321)])
    assert result[0].confidence == pytest.approx(0.7321)


def test_class_name_preserved():
    tracker = ByteTrackAdapter()
    result = tracker.update([_det([10, 10, 50, 50], class_name="knife")])
    assert result[0].class_name == "knife"


def test_multi_class_frame_preserves_each_detection_correctly():
    tracker = ByteTrackAdapter()
    dets = [
        _det([10, 10, 50, 50], class_name="person", confidence=0.81),
        _det([200, 200, 260, 260], class_name="rifle", confidence=0.65),
        _det([400, 50, 460, 110], class_name="fire", confidence=0.92),
    ]
    result = tracker.update(dets)
    assert len(result) == 3
    by_class = {t.class_name: t for t in result}
    assert by_class["person"].bbox == [10, 10, 50, 50]
    assert by_class["person"].confidence == pytest.approx(0.81)
    assert by_class["rifle"].bbox == [200, 200, 260, 260]
    assert by_class["rifle"].confidence == pytest.approx(0.65)
    assert by_class["fire"].bbox == [400, 50, 460, 110]
    assert by_class["fire"].confidence == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# 10. reset()
# ---------------------------------------------------------------------------
def test_reset_clears_state_and_restarts_ids():
    tracker = ByteTrackAdapter()
    first = tracker.update([_det([10, 10, 50, 50])])
    first_id = first[0].track_id

    tracker.reset()

    second = tracker.update([_det([10, 10, 50, 50])])
    assert len(second) == 1
    # reset() explicitly zeroes Ultralytics' shared track-ID counter, so the
    # very next track issued by ANY tracker right after a reset() must be 1 --
    # this is deterministic precisely because nothing else can have advanced
    # the counter between reset() and this update() call.
    assert second[0].track_id == 1


def test_reset_drops_all_per_class_trackers():
    tracker = ByteTrackAdapter()
    tracker.update(
        [
            _det([10, 10, 50, 50], class_name="person"),
            _det([100, 100, 140, 140], class_name="knife"),
        ]
    )
    assert len(tracker._trackers) == 2
    tracker.reset()
    assert len(tracker._trackers) == 0


# ---------------------------------------------------------------------------
# 11. Temporary disappearance (lost-track buffering) -- genuine recovery
# ---------------------------------------------------------------------------
def test_temporary_disappearance_recovers_same_track_id_via_lost_buffer():
    """Proves genuine ByteTrack lost-track recovery: the SAME track_id must
    come back after a temporary disappearance, not merely *a* valid track_id
    (which a freshly-created track would also satisfy).

    Mechanism being exercised: BYTETracker keeps a disappeared-but-not-yet-
    expired track in `self.lost_stracks`, and `strack_pool = joint_stracks(
    tracked_stracks, self.lost_stracks)` includes lost tracks in the very
    next frame's first-stage IoU association. If the reappearing detection's
    (Kalman-predicted) IoU with that lost track clears `match_thresh`, it is
    matched via `_apply_match` -> `STrack.re_activate(new_track, frame_id,
    new_id=False)`, which explicitly reuses the existing track_id instead of
    calling `next_id()`. This is upstream Ultralytics behavior; this test
    exercises the real BYTETracker, not a mock.
    """
    tracker = ByteTrackAdapter()
    bbox = [10.0, 10.0, 50.0, 50.0]

    # Frame 1: create the object and record its track_id.
    frame1 = tracker.update([_det(bbox, confidence=0.9)])
    assert len(frame1) == 1
    original_track_id = frame1[0].track_id

    # Frames 2-4: object disappears. Well under ByteTrack's default
    # track_buffer of 30 frames, so the track must still be sitting in
    # lost_stracks (not yet expired by _remove_stale_lost) when it reappears.
    for _ in range(3):
        empty_result = tracker.update([])
        assert empty_result == []

    # Reappearance: same object, small positional drift (as a real detector
    # would produce frame-to-frame), same confidence.
    reappearance = tracker.update([_det(_shift(bbox, dx=2, dy=1), confidence=0.9)])

    assert len(reappearance) == 1
    recovered_track_id = reappearance[0].track_id

    # The critical assertion: this must be the SAME track, not a new one.
    # A buggy or naive tracker that simply started a fresh track on
    # reappearance would produce a *different* track_id here and this
    # assertion would (correctly) fail.
    assert recovered_track_id == original_track_id

    # bbox/confidence for the recovered track must still be the exact
    # reappearance Detection's own values (adapter passthrough), not the
    # frame-1 values and not a Kalman-smoothed reconstruction.
    assert reappearance[0].bbox == _shift(bbox, dx=2, dy=1)
    assert reappearance[0].confidence == pytest.approx(0.9)


def test_disappearance_recovery_holds_across_multiple_disappearance_lengths():
    """Same recovery property, exercised at a couple of different (still well
    under track_buffer=30) disappearance lengths, so the previous test isn't
    a one-off coincidence of a specific frame count."""
    for gap_frames in (1, 5):
        tracker = ByteTrackAdapter()
        bbox = [100.0, 100.0, 140.0, 140.0]

        frame1 = tracker.update([_det(bbox, class_name="knife")])
        original_track_id = frame1[0].track_id

        for _ in range(gap_frames):
            assert tracker.update([]) == []

        reappearance = tracker.update([_det(_shift(bbox, dx=1, dy=1), class_name="knife")])
        assert len(reappearance) == 1
        assert reappearance[0].track_id == original_track_id, (
            f"expected recovery of track_id {original_track_id} after a "
            f"{gap_frames}-frame gap, got {reappearance[0].track_id}"
        )


# ---------------------------------------------------------------------------
# 12. Multi-frame stability
# ---------------------------------------------------------------------------
def test_ids_remain_stable_across_several_synthetic_frames():
    tracker = ByteTrackAdapter()
    bbox = [50.0, 50.0, 90.0, 90.0]
    track_id = None
    for i in range(8):
        frame = tracker.update([_det(_shift(bbox, dx=i * 1.5, dy=i * 0.5))])
        assert len(frame) == 1
        if track_id is None:
            track_id = frame[0].track_id
        else:
            assert frame[0].track_id == track_id


# ---------------------------------------------------------------------------
# VideoDetector-produced Detection compatibility / Tracker contract
# ---------------------------------------------------------------------------
def test_adapter_satisfies_tracker_protocol():
    assert isinstance(ByteTrackAdapter(), Tracker)


def test_accepts_plain_detection_objects_as_any_detector_would_produce():
    """No detector-specific format required -- any List[Detection], as
    WeaponDetector/PersonDetector/FireDetector/SmokeDetector all produce,
    is accepted directly."""
    tracker = ByteTrackAdapter()
    weapon_style = Detection(bbox=[1.0, 2.0, 3.0, 4.0], class_name="smg", confidence=0.55)
    person_style = Detection(bbox=[5.0, 6.0, 7.0, 8.0], class_name="person", confidence=0.77)
    result = tracker.update([weapon_style, person_style])
    assert len(result) == 2
