"""Input validation and normalisation to RGB uint8 (H, W, 3).

Colour handling is explicit: OpenCV frames are BGR, PIL images are RGB. A bare
3/4-channel ndarray therefore REQUIRES `color_order`; there is no silent default.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .errors import InvalidInputError

MAX_SIDE_PX = 16384
MIN_RECOMMENDED_SIDE_PX = 32
_PIL_OK_MODES = {"1", "L", "P", "RGB", "RGBA", "LA", "PA"}
_ALPHA_PIL_MODES = {"RGBA", "LA", "PA"}


def _composite_on_white(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha.astype(np.uint16)[..., None]
    out = (rgb.astype(np.uint16) * a + 255 * (255 - a) + 127) // 255
    return out.astype(np.uint8)


def to_rgb_uint8(image: Any, color_order: Optional[str] = None) -> np.ndarray:
    """Return a C-contiguous (H, W, 3) uint8 RGB array or raise InvalidInputError.

    Supported: PIL.Image (modes 1, L, P, RGB, RGBA, LA, PA); uint8 ndarray with
    shape (H,W), (H,W,1), (H,W,3) or (H,W,4). Alpha is composited on white.
    """
    if image is None:
        raise InvalidInputError("image is None")

    if isinstance(image, Image.Image):
        if image.mode not in _PIL_OK_MODES:
            raise InvalidInputError(f"unsupported PIL mode {image.mode!r}")
        if image.width == 0 or image.height == 0:
            raise InvalidInputError("image is empty (zero width or height)")
        has_alpha = image.mode in _ALPHA_PIL_MODES or (
            image.mode == "P" and "transparency" in image.info)
        if has_alpha:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            arr = _composite_on_white(rgba[..., :3], rgba[..., 3])
        else:
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return _check_size(np.ascontiguousarray(arr))

    if not isinstance(image, np.ndarray):
        raise InvalidInputError(
            f"image must be a PIL.Image or numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise InvalidInputError("image is empty")
    if image.dtype != np.uint8:
        raise InvalidInputError(f"ndarray must be uint8, got {image.dtype}")

    if image.ndim == 2:
        arr = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        arr = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        if color_order not in ("RGB", "BGR"):
            raise InvalidInputError(
                "color_order must be 'RGB' or 'BGR' for 3/4-channel ndarray input "
                "(OpenCV frames are BGR)")
        ch = image[..., [2, 1, 0]] if color_order == "BGR" else image[..., :3]
        if image.shape[2] == 4:
            alpha = image[..., 3]
            arr = _composite_on_white(ch, alpha)
        else:
            arr = ch
    else:
        raise InvalidInputError(f"unsupported ndarray shape {image.shape}")
    return _check_size(np.ascontiguousarray(arr))


def _check_size(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < 1 or w < 1:
        raise InvalidInputError("image is empty")
    if max(h, w) > MAX_SIDE_PX:
        raise InvalidInputError(f"image side {max(h, w)} exceeds {MAX_SIDE_PX}px")
    return arr


def crop_bbox(rgb: np.ndarray, bbox_xyxy: Sequence[float]) -> Tuple[np.ndarray, List[int]]:
    """Crop `rgb` with a pixel [x1,y1,x2,y2] box (as produced by the existing
    detectors). The box is clamped to the image; returns (crop, clamped_int_box).
    Raises InvalidInputError if the clamped box has no area or contains non-finite values.
    """
    if len(bbox_xyxy) != 4 or not all(np.isfinite(v) for v in bbox_xyxy):
        raise InvalidInputError(f"bbox must be 4 finite numbers, got {bbox_xyxy!r}")
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    ix1 = int(max(0, np.floor(x1)))
    iy1 = int(max(0, np.floor(y1)))
    ix2 = int(min(w, np.ceil(x2)))
    iy2 = int(min(h, np.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        raise InvalidInputError(f"bbox {list(bbox_xyxy)} has no area inside {w}x{h} image")
    return np.ascontiguousarray(rgb[iy1:iy2, ix1:ix2]), [ix1, iy1, ix2, iy2]


def laplacian_variance(rgb: np.ndarray) -> float:
    """Heuristic sharpness (variance of a 4-neighbour Laplacian on luma).
    Uncalibrated; only a relative quality hint."""
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float64)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())
