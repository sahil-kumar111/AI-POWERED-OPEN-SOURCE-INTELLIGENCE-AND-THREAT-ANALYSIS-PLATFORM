"""Deterministic preprocessing (PIL + numpy; no torchvision dependency)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .errors import EncoderIntegrityError
from .datatypes import PreprocessSpec

# Published HF preprocessing for the supported families.
DINOV2_PREPROCESS = PreprocessSpec(
    id="dinov2-imagenet-resize256-bicubic-centercrop224-v1",
    resize_shortest_edge=256, crop_size=224,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

CLIP_PREPROCESS = PreprocessSpec(
    id="clip-resize224-bicubic-centercrop224-v1",
    resize_shortest_edge=224, crop_size=224,
    mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))


def _target_size(w: int, h: int, shortest: int):
    if w <= h:
        return shortest, max(1, int(shortest * h / w))
    return max(1, int(shortest * w / h)), shortest


def preprocess(rgb: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (3,crop,crop) float32, normalised."""
    h, w = rgb.shape[:2]
    nw, nh = _target_size(w, h, spec.resize_shortest_edge)
    im = Image.fromarray(rgb, "RGB").resize((nw, nh), Image.BICUBIC)
    c = spec.crop_size
    left = (nw - c) // 2
    top = (nh - c) // 2
    im = im.crop((left, top, left + c, top + c))
    x = np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0)
    x = (x - np.asarray(spec.mean, np.float32)) / np.asarray(spec.std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def _size_int(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, dict):
        for k in ("shortest_edge", "height"):
            if k in v:
                return int(v[k])
    return None


def check_checkpoint_preprocessing(weights_dir: Path, spec: PreprocessSpec) -> List[str]:
    """Cross-check the checkpoint's preprocessor_config.json against `spec`.

    Returns warnings (e.g. file absent). Raises EncoderIntegrityError on a mismatch,
    so a checkpoint never silently runs with different preprocessing than recorded.
    """
    p = Path(weights_dir) / "preprocessor_config.json"
    if not p.is_file():
        return ["preprocessor_config.json_not_found: preprocessing not cross-checked"]
    try:
        cfg: Dict[str, Any] = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise EncoderIntegrityError(f"unreadable preprocessor_config.json: {e}") from e
    problems = []
    if cfg.get("do_resize", True) is False:
        problems.append("do_resize is false")
    if cfg.get("do_center_crop", True) is False:
        problems.append("do_center_crop is false")
    if cfg.get("do_convert_rgb", True) is False:
        problems.append("do_convert_rgb is false")
    rs = _size_int(cfg.get("size"))
    if rs is not None and rs != spec.resize_shortest_edge:
        problems.append(f"resize {rs} != {spec.resize_shortest_edge}")
    cs = _size_int(cfg.get("crop_size"))
    if cs is not None and cs != spec.crop_size:
        problems.append(f"crop_size {cs} != {spec.crop_size}")
    for key, want in (("image_mean", spec.mean), ("image_std", spec.std)):
        got = cfg.get(key)
        if got is not None and not np.allclose(got, want, atol=1e-6):
            problems.append(f"{key} {got} != {list(want)}")
    rsm = cfg.get("resample")
    if rsm is not None and rsm != 3:  # PIL BICUBIC == 3
        problems.append(f"resample {rsm} != 3 (bicubic)")
    if problems:
        raise EncoderIntegrityError(
            "checkpoint preprocessor_config.json disagrees with applied preprocessing: "
            + "; ".join(problems))
    return []
