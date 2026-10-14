"""Weights integrity helpers (mirrors the SHA-256 pattern of the existing detectors)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Optional

from .errors import EncoderIntegrityError, EncoderLoadError

WEIGHTS_FILENAME = "model.safetensors"


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def locate_weights(weights_dir: Optional[Path]) -> Path:
    """Return <weights_dir>/model.safetensors. Only a local directory with a single
    safetensors file is accepted (no hub ids, no pickle .bin, no sharded files)."""
    if weights_dir is None:
        raise EncoderLoadError("weights_dir is not configured")
    d = Path(weights_dir)
    if not d.exists():
        raise EncoderLoadError(f"weights directory not found: {d}")
    if not d.is_dir():
        raise EncoderLoadError(f"weights_dir must be a directory: {d}")
    if (d / "model.safetensors.index.json").exists():
        raise EncoderLoadError("sharded checkpoints are not supported in Phase 2")
    f = d / WEIGHTS_FILENAME
    if not f.is_file():
        raise EncoderLoadError(f"{WEIGHTS_FILENAME} not found in {d}")
    if not (d / "config.json").is_file():
        raise EncoderLoadError(f"config.json not found in {d}")
    return f


def verify_weights(weights_file: Path, expected: Optional[str], allow_unverified: bool):
    """Compute the SHA-256 and compare. Returns (sha256, warnings)."""
    actual = sha256_file(weights_file)
    if expected is None:
        if not allow_unverified:
            raise EncoderIntegrityError(
                "expected_weights_sha256 is required (or set allow_unverified=True). "
                f"Computed SHA-256: {actual}")
        return actual, ["weights_sha256_not_verified_against_expected_value"]
    if actual.lower() != expected.lower():
        raise EncoderIntegrityError(
            f"weights SHA-256 mismatch: expected {expected}, actual {actual}")
    return actual, []


if __name__ == "__main__":  # python -m representation_learning.encoders.integrity <dir>
    if len(sys.argv) != 2:
        sys.exit("usage: python -m representation_learning.encoders.integrity <weights_dir>")
    wf = locate_weights(Path(sys.argv[1]))
    print(f"{sha256_file(wf)}  {wf}")
    cfg = wf.parent / "config.json"
    print(f"{sha256_file(cfg)}  {cfg}")
