"""
Tracker contract.

This defines the interface that a multi-object tracker (e.g. a future
ByteTrack-based implementation) must satisfy to sit between a
VideoDetector's per-frame Detections and downstream components
(event fusion, XAI, etc.) that need persistent object identity across
frames.

NOTE: This file defines the contract only. No tracker is implemented
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable

from backend.app.interfaces.types import Detection


@dataclass(frozen=True)
class TrackedDetection:
    """A Detection augmented with a persistent track identity.

    Carries the same fields as Detection (bbox, class_name, confidence)
    plus a track_id that a Tracker implementation guarantees stays
    stable for the same physical object across consecutive frames.
    """

    bbox: List[float]
    class_name: str
    confidence: float
    track_id: int


@runtime_checkable
class Tracker(Protocol):
    """Structural contract for a multi-object, multi-frame tracker."""

    def update(
        self, detections: List[Detection], frame: Any = None
    ) -> List[TrackedDetection]:
        """Advance the tracker by one frame's worth of detections.

        Parameters
        ----------
        detections : List[Detection]
            Detections produced by a VideoDetector for the current frame.
            May be empty.
        frame : Any, optional
            The current video frame, if the tracker implementation needs
            pixel data (e.g. for appearance-based association). Optional
            because motion-only trackers (e.g. ByteTrack) don't require it.

        Returns
        -------
        List[TrackedDetection]
            Zero or more tracked detections, each carrying a persistent
            track_id. Must preserve bbox, class_name, and confidence
            from the corresponding input Detection.
        """
        ...

    def reset(self) -> None:
        """Clear all internal track state (e.g. when starting a new video)."""
        ...
