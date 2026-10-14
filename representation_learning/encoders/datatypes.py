"""Data types. Output records follow the Phase 1 `EmbeddingRecord` contract
(docs/data_contracts.md section 4); no competing schema is defined here."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PreprocessSpec:
    """Deterministic image preprocessing applied before the network.

    Steps: RGB uint8 -> resize shortest edge (bicubic, aspect preserved) ->
    center crop -> scale to [0,1] -> (x-mean)/std -> CHW float32.
    """

    id: str
    resize_shortest_edge: int
    crop_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    interpolation: str = "bicubic"
    color_order: str = "RGB"
    alpha_policy: str = "composite_on_white"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "color_order": self.color_order,
            "alpha_policy": self.alpha_policy,
            "input_dtype": "uint8",
            "resize": {"mode": "shortest_edge", "size": self.resize_shortest_edge,
                       "interpolation": self.interpolation},
            "crop": {"mode": "center", "size": self.crop_size},
            "rescale": 1.0 / 255.0,
            "normalize": {"mean": list(self.mean), "std": list(self.std)},
            "output_layout": "CHW",
            "output_dtype": "float32",
        }

    def with_overrides(self, **kw) -> "PreprocessSpec":
        return replace(self, **kw)


@dataclass(frozen=True)
class EncoderSpec:
    """What an encoder family extracts. Fixed by code, not by the caller."""

    name: str                 # e.g. "dinov2-small"
    family: str               # "DINOv2" | "CLIP"
    feature_name: str         # exactly which tensor is the embedding
    expected_dim: int
    preprocess: PreprocessSpec
    license: str
    source_ref: str
    metric: str = "cosine"


@dataclass(frozen=True)
class EncoderConfig:
    """Runtime configuration (paths, device, integrity)."""

    weights_dir: Optional[Path] = None      # LOCAL directory only; never a hub id
    checkpoint_id: str = "UNSPECIFIED"      # e.g. "facebook/dinov2-small@<revision>"
    expected_weights_sha256: Optional[str] = None
    allow_unverified: bool = False          # permit loading without an expected hash
    device: str = "auto"                    # "auto" | "cpu" | "cuda" | "cuda:N"
    precision: str = "float32"              # only float32 in Phase 2
    pipeline_version: Optional[str] = None  # overrides auto-detected value
    default_may_contain_persons: bool = True  # conservative default for ad-hoc images
    access_label: str = "restricted"
    run_id: Optional[str] = None
    max_batch_size: int = 8                 # chunk size used by encode_batch
    license: Optional[str] = None           # pinned licence string (from the weights manifest)
    revision: Optional[str] = None          # pinned upstream revision/commit (from the manifest)


@dataclass(frozen=True)
class BackendInfo:
    """Facts established when the model is loaded."""

    weights_sha256: str
    device: str
    precision: str
    dim: int
    framework_versions: Dict[str, str]
    extra: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualRepresentation:
    """One embedding + its Phase 1 `EmbeddingRecord` (the vector is held
    in memory; `record["vector_ref"]` is only an in-memory placeholder)."""

    vector: np.ndarray        # float32, shape (dim,), read-only, L2-normalised
    record: Dict[str, Any]

    @property
    def embedding_id(self) -> str:
        return self.record["embedding_id"]

    @property
    def space_id(self) -> str:
        return self.record["space_id"]

    @property
    def dim(self) -> int:
        return self.record["dim"]


@dataclass(frozen=True)
class EncodeFailure:
    """Per-item failure reported by `encode_batch(on_error="collect")`."""

    index: int
    error_type: str
    message: str
