"""Real-checkpoint backends (PyTorch + Hugging Face `transformers`).

Imported lazily; torch/transformers are OPTIONAL dependencies (see
requirements-phase2.txt). Only LOCAL directories are loaded (`local_files_only=True`),
and only `model.safetensors` (no pickle). Nothing here downloads anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .datatypes import BackendInfo, EncoderConfig
from .errors import EncoderIntegrityError, EncoderLoadError
from .integrity import locate_weights, sha256_file, verify_weights
from .preprocessing import check_checkpoint_preprocessing
from .datatypes import PreprocessSpec


def _import_torch_stack():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise EncoderLoadError(
            "torch and transformers are required for real checkpoints; install the pinned "
            "versions from representation_learning/requirements-phase2.txt "
            f"({e})") from e
    import torch
    import transformers
    return torch, transformers


def resolve_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise EncoderLoadError(f"device {requested!r} requested but CUDA is not available")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise EncoderLoadError(f"unsupported device {requested!r} (use auto, cpu, cuda[:N])")
    return requested


class TorchBackend:
    """Wraps a torch module; `extract(module, pixel_values_tensor) -> tensor (N,dim)`."""

    def __init__(self, config: EncoderConfig, preprocess: PreprocessSpec, *,
                 model_type: str, expected_dim: int, load_model: Callable,
                 extract: Callable, dim_from_config: Callable):
        # Configuration and file-integrity checks come FIRST and need neither torch nor
        # transformers, so an unconfigured/missing/mismatched checkpoint reports its real
        # cause even when the optional real-model stack is not installed.
        if config.precision != "float32":
            raise EncoderLoadError("only float32 is supported in Phase 2")
        wfile = locate_weights(config.weights_dir)
        wdir = wfile.parent
        sha, warns = verify_weights(wfile, config.expected_weights_sha256,
                                    config.allow_unverified)
        try:
            cfg_json = json.loads((wdir / "config.json").read_text())
        except (OSError, ValueError) as e:
            raise EncoderLoadError(f"unreadable config.json: {e}") from e
        if cfg_json.get("model_type") != model_type:
            raise EncoderIntegrityError(
                f"config.json model_type {cfg_json.get('model_type')!r} != {model_type!r}")
        cfg_dim = dim_from_config(cfg_json)
        if cfg_dim != expected_dim:
            raise EncoderIntegrityError(
                f"checkpoint dim {cfg_dim} != expected {expected_dim} for this variant")
        warns = list(warns) + check_checkpoint_preprocessing(wdir, preprocess)
        # Only now is the optional dependency stack needed.
        torch, transformers = _import_torch_stack()
        device = resolve_device(config.device, torch)
        try:
            model = load_model(transformers, str(wdir))
        except Exception as e:
            raise EncoderLoadError(f"failed to load checkpoint from {wdir}: "
                                   f"{type(e).__name__}: {e}") from e
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        self._torch, self._model, self._extract, self._device = torch, model, extract, device
        self.info = BackendInfo(
            weights_sha256=sha, device=device, precision="float32", dim=expected_dim,
            framework_versions={"torch": torch.__version__,
                                "transformers": transformers.__version__},
            extra={"config_sha256": sha256_file(wdir / "config.json")},
            warnings=tuple(warns))

    def forward(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self._device)
            y = self._extract(self._model, x)
            return y.detach().to("cpu", dtype=torch.float32).numpy()

    def peak_cuda_mb(self) -> Optional[float]:
        if self._device.startswith("cuda"):
            return self._torch.cuda.max_memory_allocated() / 2**20
        return None
