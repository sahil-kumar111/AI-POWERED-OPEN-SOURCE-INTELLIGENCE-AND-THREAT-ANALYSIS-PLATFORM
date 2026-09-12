"""
VideoDetector contract.

Any frame-level detector in the Video Intelligence module (PersonDetector,
WeaponDetector, future detectors) is expected to satisfy this Protocol.
This file defines the contract only -- no implementation.
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

from backend.app.interfaces.types import Detection


@runtime_checkable
class VideoDetector(Protocol):
    """Structural contract for a single-frame object detector."""

    def detect(self, frame: Any, config: Any) -> List[Detection]:
        """Run detection on a single frame.

        Parameters
        ----------
        frame : Any
            A single video frame (e.g. HxWx3 numpy array).
        config : Any
            Detector-specific configuration (thresholds, etc.).

        Returns
        -------
        List[Detection]
            Zero or more detections found in the frame.
        """
        ...
