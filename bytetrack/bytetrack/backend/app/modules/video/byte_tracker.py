"""
ByteTrack adapter
==================
Concrete Tracker implementation that adapts the official Ultralytics
BYTETracker (ultralytics.trackers.byte_tracker.BYTETracker) to this
project's Tracker contract (backend/app/interfaces/tracker.py).

This module does not reimplement ByteTrack. It uses the real algorithm
as shipped by Ultralytics:
    - Kalman-filter motion prediction (KalmanFilterXYAH, via STrack)
    - two-stage association: high-confidence detections first, then
      low-confidence detections against the remaining unmatched tracks
      (BYTETracker._first_association / _second_association)
    - IoU-based cost matrices + Hungarian assignment
      (ultralytics.trackers.utils.matching)
    - track lifecycle state machine: New -> Tracked -> Lost -> Removed
      (ultralytics.trackers.basetrack.TrackState, BYTETracker.update)

Design: Detection -> adapter -> Ultralytics BYTETracker -> TrackedDetection
--------------------------------------------------------------------------
BYTETracker's `update()` expects a "Results-like" object exposing
`.xywh`/`.xyxy`, `.conf`, `.cls` (as numpy arrays) and supporting
`len()` and boolean-mask indexing (`results[mask]`). `_ResultsView`
below is that adapter object, built directly from our Detection list --
no Ultralytics model output is required.

Class-aware tracking
---------------------
The stock BYTETracker performs class-agnostic IoU matching: nothing in
its own association logic checks `cls`. To guarantee that a detection of
one semantic class (e.g. "person") can never be matched onto, or inherit
the track_id of, a different class's track (e.g. "pistol"), this adapter
keeps one independent BYTETracker instance per class_name and routes
each frame's detections only to the tracker for their own class. Two
objects of different classes are therefore never in the same matching
pool and can never be confused for one another, regardless of bbox
overlap. This is implemented entirely in this adapter -- the Ultralytics
BYTETracker class itself is used unmodified.

One subtlety this adapter has to handle: `BYTETracker.__init__` resets
Ultralytics' shared, process-global track-ID counter
(`ultralytics.trackers.basetrack.BaseTrack._count`) to 0 as a side
effect. Since per-class trackers are created lazily (a class's tracker
only needs to exist once a detection of that class is first seen), a
naive implementation would have every newly-created per-class tracker
reset numbering back to 1, causing two different classes' tracks to
collide on the same track_id even though they are matched in completely
separate pools. This adapter snapshots and restores that shared counter
around each new per-class tracker's construction (after the first) so
track IDs stay globally unique across all classes, without touching any
Ultralytics source file.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from backend.app.interfaces.types import Detection
from backend.app.interfaces.tracker import TrackedDetection

# ---------------------------------------------------------------------------
# Official Ultralytics ByteTrack hyperparameters (unmodified defaults, as
# shipped in ultralytics/cfg/trackers/bytetrack.yaml). Loaded once and
# shared across all per-class BYTETracker instances -- this mirrors how
# Ultralytics itself shares a single cfg object across multiple tracker
# instances when batch size > 1 (see ultralytics.trackers.track.on_predict_start).
# ---------------------------------------------------------------------------
_BYTETRACK_CFG = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))


class _ResultsView:
    """Minimal "Results-like" object satisfying BYTETracker.update()'s input
    contract: `.xywh`, `.xyxy`, `.conf`, `.cls` as numpy arrays, `len()`, and
    boolean-mask `__getitem__`. Built directly from Detection objects --
    no Ultralytics model inference involved.
    """

    __slots__ = ("xywh", "xyxy", "conf", "cls")

    def __init__(self, xywh: np.ndarray, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_ResultsView":
        return _ResultsView(self.xywh[mask], self.xyxy[mask], self.conf[mask], self.cls[mask])


def _empty_results_view() -> _ResultsView:
    z4 = np.zeros((0, 4), dtype=np.float32)
    z1 = np.zeros((0,), dtype=np.float32)
    return _ResultsView(z4, z4, z1, z1)


def _detections_to_results_view(detections: List[Detection]) -> _ResultsView:
    if not detections:
        return _empty_results_view()

    xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
    xywh = xyxy.copy()
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

    conf = np.array([d.confidence for d in detections], dtype=np.float32)
    # cls is unused for our own class_name reconstruction (the adapter already
    # knows which per-class tracker produced each row), but is required by
    # BYTETracker's input contract, so a constant placeholder is supplied.
    cls = np.zeros((len(detections),), dtype=np.float32)

    return _ResultsView(xywh, xyxy, conf, cls)


class ByteTrackAdapter:
    """Concrete Tracker implementation backed by the official Ultralytics
    BYTETracker, with class-aware isolation implemented in this adapter.

    Implements the Tracker Protocol from backend/app/interfaces/tracker.py:
        update(detections, frame=None) -> List[TrackedDetection]
        reset() -> None
    """

    def __init__(self) -> None:
        self._trackers: Dict[str, BYTETracker] = {}

    # -- internal: per-class tracker management -------------------------------
    def _get_or_create_tracker(self, class_name: str) -> BYTETracker:
        tracker = self._trackers.get(class_name)
        if tracker is not None:
            return tracker

        # BYTETracker.__init__ unconditionally resets the shared, process-global
        # BaseTrack._count ID counter to 0. Since per-class trackers are created
        # lazily (whenever a class is first seen, which may be well after other
        # classes already have active tracks), that reset must not be allowed to
        # collide numbering across classes. Snapshot/restore keeps IDs globally
        # unique without touching Ultralytics' own source.
        prior_count = BaseTrack._count
        tracker = BYTETracker(args=_BYTETRACK_CFG)
        BaseTrack._count = prior_count

        self._trackers[class_name] = tracker
        return tracker

    # -- public API (Tracker contract) -----------------------------------------
    def update(self, detections: List[Detection], frame: Any = None) -> List[TrackedDetection]:
        """
        Advance every known per-class tracker by one frame.

        Parameters
        ----------
        detections : List[Detection]
            Detections for the current frame, from any VideoDetector
            (WeaponDetector, PersonDetector, FireDetector, SmokeDetector).
            May be empty. May contain multiple classes.
        frame : Any, optional
            Unused by BYTETracker's default (non-appearance, non-GMC)
            configuration; accepted only for Tracker contract compatibility.

        Returns
        -------
        List[TrackedDetection]
            One entry per detection that produced (or was matched onto) an
            activated track this frame. bbox, class_name, and confidence
            are the exact values from the corresponding input Detection;
            track_id is newly assigned/maintained by ByteTrack. Note: per
            genuine ByteTrack behavior, a brand-new object first seen after
            a class tracker's very first frame is held as "unconfirmed" and
            will not appear in the output until it is matched again on the
            following frame -- this is upstream ByteTrack confirmation
            logic, not a defect in this adapter.
        """
        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)

        # Every class with an existing tracker must still be advanced this
        # frame (with an empty result if it has no detections now), so that
        # ByteTrack's internal frame counter and lost-track buffering (which
        # governs handling of temporary disappearance) stay correct.
        classes_to_update = set(self._trackers.keys()) | set(by_class.keys())

        results: List[TrackedDetection] = []
        for class_name in classes_to_update:
            class_detections = by_class.get(class_name, [])
            tracker = self._get_or_create_tracker(class_name)
            results_view = _detections_to_results_view(class_detections)

            output = tracker.update(results_view, frame)
            if output is None or len(output) == 0:
                continue

            for row in output:
                track_id = int(row[4])
                idx = int(row[7])
                source_detection = class_detections[idx]
                results.append(
                    TrackedDetection(
                        bbox=list(source_detection.bbox),
                        class_name=source_detection.class_name,
                        confidence=source_detection.confidence,
                        track_id=track_id,
                    )
                )

        return results

    def reset(self) -> None:
        """Completely clear all tracker state across every class.

        Drops every per-class BYTETracker instance and resets Ultralytics'
        shared track-ID counter, so tracking that follows a reset() starts
        from a genuinely clean state (next new track anywhere gets ID 1).
        """
        self._trackers.clear()
        BYTETracker.reset_id()
