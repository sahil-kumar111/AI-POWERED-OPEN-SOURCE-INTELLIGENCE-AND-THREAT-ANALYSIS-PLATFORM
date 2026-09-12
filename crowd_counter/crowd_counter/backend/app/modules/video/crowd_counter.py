"""
CrowdCounter
============
Per-frame crowd counting as a lightweight reduction over an existing
List[Detection] (e.g. as produced by PersonDetector). No model inference,
no new Detection type, no tracking. Counts only detections whose
class_name == "person".

NOTE: This is per-frame counting only. It does not deduplicate the same
physical person appearing across multiple frames -- that requires track
identity, which will come from a future ByteTrack-based Tracker
implementation and is explicitly out of scope here.
"""

from __future__ import annotations

from typing import List

from backend.app.interfaces.types import Detection

PERSON_CLASS_NAME = "person"


class CrowdCounter:
    """Counts "person" detections in a single frame's Detection list."""

    def count(self, detections: List[Detection]) -> int:
        """
        Count person detections in a single frame.

        Parameters
        ----------
        detections : List[Detection]
            Detections for one frame, from any VideoDetector (e.g.
            PersonDetector). May contain non-person classes (weapon,
            fire, smoke, etc.), which are ignored. May be empty.

        Returns
        -------
        int
            The number of detections with class_name == "person" in this
            frame. 0 if there are no person detections (including an
            empty input list).
        """
        if not detections:
            return 0
        return sum(1 for d in detections if d.class_name == PERSON_CLASS_NAME)
