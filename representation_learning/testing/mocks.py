"""Deterministic MOCK backends. They are NOT DINOv2/CLIP and carry no semantics:
the output is a seeded random projection of a coarse pixel grid, so identical
inputs give identical vectors and different inputs give different vectors."""

from __future__ import annotations

import hashlib
from typing import Callable, Optional

import numpy as np

from ..encoders.datatypes import BackendInfo


class MockBackend:
    def __init__(self, dim: int, seed: int = 0, *, fault: Optional[str] = None,
                 device: str = "cpu"):
        rng = np.random.default_rng(seed)
        self._w = rng.standard_normal((3 * 8 * 8, dim)).astype(np.float32)
        self._fault = fault  # None | "nan" | "inf" | "zeros" | "wrong_shape" | "raise" | "int_dtype"
        self.calls = 0
        self.batch_sizes = []
        self.info = BackendInfo(
            weights_sha256=hashlib.sha256(f"mock-{dim}-{seed}".encode()).hexdigest(),
            device=device, precision="float32", dim=dim,
            framework_versions={"mock": "1"}, extra={"mock": True})

    def forward(self, batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.batch_sizes.append(int(batch.shape[0]))
        n, c, h, w = batch.shape
        # average-pool to an 8x8 grid
        g = batch.reshape(n, c, 8, h // 8, 8, w // 8).mean(axis=(3, 5)).reshape(n, -1)
        out = g @ self._w
        f = self._fault
        if f == "nan":
            out[:, 0] = np.nan
        elif f == "inf":
            out[:, 0] = np.inf
        elif f == "zeros":
            out[:] = 0.0
        elif f == "wrong_shape":
            out = out[:, :-1]
        elif f == "raise":
            raise RuntimeError("simulated backend failure")
        elif f == "int_dtype":
            out = (out * 100).astype(np.int32)
        return out.astype(np.float32) if f != "int_dtype" else out


def mock_encoder(family: str = "dinov2", seed: int = 0, fault: Optional[str] = None, **cfg):
    """Build a real encoder class around a MockBackend (no weights, no network)."""
    from ..encoders import ClipEncoder, Dinov2Encoder
    from ..encoders.datatypes import EncoderConfig

    config = EncoderConfig(checkpoint_id="MOCK", **cfg)
    if family == "dinov2":
        enc = Dinov2Encoder("small", config, backend=MockBackend(384, seed, fault=fault))
    elif family == "clip":
        enc = ClipEncoder("vit-b-32", config, backend=MockBackend(512, seed, fault=fault))
    else:
        raise ValueError(family)
    return enc
