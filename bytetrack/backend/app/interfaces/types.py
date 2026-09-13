"""
Canonical shared types for the Video Intelligence module.

This is the single, minimal contract that all frame-level detectors
(PersonDetector, WeaponDetector, future detectors) are expected to
produce, and that downstream components (trackers, event fusion, etc.)
are expected to consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Detection:
    """A single object detection in one video frame.

    Attributes
    ----------
    bbox : List[float]
        Bounding box as [x1, y1, x2, y2] in pixel coordinates.
    class_name : str
        Human-readable class label (e.g. "person", "pistol", "knife").
    confidence : float
        Detector confidence score for this detection, in [0.0, 1.0].
    """

    bbox: List[float]
    class_name: str
    confidence: float
