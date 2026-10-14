"""Common visual encoder interface: `encode(image) -> VisualRepresentation`.

Design points
- Backends are lazy: nothing heavy is imported or loaded until first use.
- Unit tests inject a mock backend; real backends load LOCAL weights only.
- Output records follow the Phase 1 `EmbeddingRecord` contract.
- Embeddings are float32, finite and L2-normalised. Vectors from different
  encoders (different `space_id`) are NOT comparable and this module offers no
  comparison API (Phase 1 architecture.md section 6).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

import numpy as np

from .. import __version__
from .errors import (EmbeddingValidationError, EncoderError, EncoderIntegrityError,
                     EncoderLoadError, InvalidInputError, UnsupportedEvidenceError)
from .image_io import (MIN_RECOMMENDED_SIDE_PX, crop_bbox, laplacian_variance,
                       to_rgb_uint8)
from .preprocessing import preprocess
from .datatypes import (BackendInfo, EncodeFailure, EncoderConfig, EncoderSpec,
                    VisualRepresentation)

SCHEMA_VERSION = "0.1"
NORM_TOLERANCE = 1e-5


class Backend(Protocol):
    """Runs the network. `batch` is (N,3,H,W) float32 (already preprocessed);
    returns (N,dim) raw (un-normalised) features."""

    info: BackendInfo

    def forward(self, batch: np.ndarray) -> np.ndarray: ...


def _pipeline_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            return f"{__version__}+git.{out.stdout.strip()}"
    except (OSError, subprocess.SubprocessError):
        pass
    return __version__


def _short_hash(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.hexdigest()


class VisualEncoder:
    """Base class. Subclasses provide `_make_backend(config)` (real weights) or the
    caller injects `backend=` (mocks/tests)."""

    def __init__(self, spec: EncoderSpec, config: Optional[EncoderConfig] = None,
                 backend: Optional[Backend] = None):
        self.spec = spec
        self.config = config or EncoderConfig()
        self._backend: Optional[Backend] = None
        self._space_id: Optional[str] = None
        if backend is not None:
            self._adopt_backend(backend)

    # ------------------------------------------------------------------ loading
    def _make_backend(self) -> Backend:  # pragma: no cover - overridden
        raise EncoderLoadError(f"{type(self).__name__} has no backend factory")

    def _adopt_backend(self, backend: Backend) -> None:
        info = backend.info
        if info.dim != self.spec.expected_dim:
            raise EncoderIntegrityError(
                f"backend dim {info.dim} != expected {self.spec.expected_dim} for {self.spec.name}")
        self._backend = backend
        self._space_id = None

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> BackendInfo:
        """Load weights now (otherwise done lazily on first encode)."""
        if self._backend is None:
            self._adopt_backend(self._make_backend())
        assert self._backend is not None
        return self._backend.info

    @property
    def info(self) -> BackendInfo:
        return self.load()

    # ----------------------------------------------------------------- identity
    @property
    def space_id(self) -> str:
        """Hash of encoder identity + weights hash + preprocessing + feature + norm
        (Phase 1: `space_id`). Changes whenever any of those change."""
        if self._space_id is None:
            info = self.load()
            ident = {
                "encoder": self.spec.name, "family": self.spec.family,
                "checkpoint": self.config.checkpoint_id,
                "weights_sha256": info.weights_sha256,
                "feature": self.spec.feature_name, "dim": info.dim,
                "preprocessing": self.spec.preprocess.to_dict(),
                "precision": info.precision, "normalization": "l2", "metric": self.spec.metric,
            }
            self._space_id = "space_" + _short_hash(
                json.dumps(ident, sort_keys=True).encode())[:16]
        return self._space_id

    # ------------------------------------------------------------ input handling
    @staticmethod
    def _check_evidence(evidence: Optional[Dict[str, Any]], diagnostic_only: bool = False) -> None:
        if evidence is None:
            return
        if not isinstance(evidence, dict):
            raise InvalidInputError("evidence must be a dict (Phase 1 VisualEvidenceItem)")
        det = evidence.get("detector") or {}
        if str(det.get("category", "")).lower() == "person" or \
                str(det.get("class_name", "")).lower() == "person":
            raise UnsupportedEvidenceError(
                "person evidence is out of scope (Phase 2 excludes person crops/embeddings)")
        uri = str(evidence.get("video_uri", ""))
        if uri.lower().endswith("_analyzed.mp4"):
            raise UnsupportedEvidenceError(
                "annotated video (*_analyzed.mp4) must not be encoded; use the original video")
        if evidence.get("source_kind") not in (None, "frame", "crop"):
            raise InvalidInputError(f"unknown source_kind {evidence.get('source_kind')!r}")
        if evidence.get("source_kind") == "frame" and not diagnostic_only:
            raise UnsupportedEvidenceError(
                "whole-frame evidence is not embedded for target-object representation; use a "
                "weapon/fire/smoke crop (source_kind='crop'), or pass diagnostic_only=True for "
                "pipeline diagnostics")
        if (evidence.get("sensitivity") or {}).get("retain_pixels") is True:
            raise UnsupportedEvidenceError(
                "evidence requests retain_pixels=true; embedding records never retain source pixels")

    def _prepare(self, image: Any, color_order: Optional[str],
                 evidence: Optional[Dict[str, Any]], bbox_xyxy: Optional[Sequence[float]],
                 input_is_crop: bool, diagnostic_only: bool = False):
        self._check_evidence(evidence, diagnostic_only)
        rgb = to_rgb_uint8(image, color_order)
        in_h, in_w = rgb.shape[:2]
        warnings: List[str] = []
        box = bbox_xyxy
        if box is None and evidence and evidence.get("source_kind") == "crop" \
                and not input_is_crop:
            box = evidence.get("bbox_xyxy")
            if box is None:
                raise InvalidInputError("crop evidence requires bbox_xyxy")
        if evidence and not input_is_crop and evidence.get("resolution"):
            ew, eh = evidence["resolution"]
            if (ew, eh) != (in_w, in_h):
                raise InvalidInputError(
                    f"frame is {in_w}x{in_h} but evidence.resolution is {ew}x{eh}")
        crop_box = None
        if box is not None and not input_is_crop:
            rgb, crop_box = crop_bbox(rgb, box)
            if list(map(float, box)) != [float(v) for v in crop_box]:
                warnings.append("bbox_clamped_or_rounded_to_pixel_grid")
        h, w = rgb.shape[:2]
        if min(h, w) < MIN_RECOMMENDED_SIDE_PX:
            warnings.append(f"small_input_min_side_px:{min(h, w)}")
        meta = {"frame_w": in_w, "frame_h": in_h, "crop_box": crop_box,
                "input_is_crop": bool(input_is_crop)}
        return rgb, meta, warnings

    # ------------------------------------------------------------- core compute
    def _embed(self, batch: np.ndarray) -> np.ndarray:
        self.load()
        assert self._backend is not None
        try:
            raw = self._backend.forward(batch)
        except EncoderError:
            raise
        except Exception as e:  # backend/runtime failure
            raise EmbeddingValidationError(f"backend forward failed: {type(e).__name__}: {e}") from e
        raw = np.asarray(raw)
        dim = self.spec.expected_dim
        if raw.ndim != 2 or raw.shape != (batch.shape[0], dim):
            raise EmbeddingValidationError(
                f"expected output shape {(batch.shape[0], dim)}, got {raw.shape}")
        if not np.issubdtype(raw.dtype, np.floating):
            raise EmbeddingValidationError(f"non-floating output dtype {raw.dtype}")
        raw64 = raw.astype(np.float64)
        if not np.isfinite(raw64).all():
            raise EmbeddingValidationError("model output contains NaN or Inf")
        norms = np.linalg.norm(raw64, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise EmbeddingValidationError("model output has (near-)zero norm; cannot normalise")
        out = (raw64 / norms).astype(np.float32)
        if not np.isfinite(out).all() or (np.abs(np.linalg.norm(out, axis=1) - 1.0)
                                         > NORM_TOLERANCE).any():
            raise EmbeddingValidationError("normalised output failed finite/unit-norm check")
        return out

    # ------------------------------------------------------------------ records
    def _record(self, vec: np.ndarray, rgb: np.ndarray, meta: Dict[str, Any],
                warnings: List[str], evidence: Optional[Dict[str, Any]],
                batch_size: int, diagnostic_only: bool = False) -> Dict[str, Any]:
        info = self.info
        spec = self.spec
        pix_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if evidence:
            evidence_id = evidence["evidence_id"]
            source_ref: Dict[str, Any] = {
                "evidence_id": evidence_id, "kind": "evidence_item",
                "video_sha256": evidence.get("video_sha256"),
                "frame_index": evidence.get("frame_index"),
                "bbox_xyxy": evidence.get("bbox_xyxy"),
                "source_kind": evidence.get("source_kind"),
            }
            sens = dict(evidence.get("sensitivity") or {})
            sens.setdefault("may_contain_persons", self.config.default_may_contain_persons)
            sens["retain_pixels"] = False     # never retained (rejected earlier if requested)
        else:
            evidence_id = "adhoc_" + pix_sha[:16]
            source_ref = {"evidence_id": evidence_id, "kind": "adhoc_image",
                          "pixel_sha256": pix_sha,
                          "note": "not a Phase 1 VisualEvidenceItem; no video provenance"}
            sens = {"may_contain_persons": self.config.default_may_contain_persons,
                    "retain_pixels": False}
        source_ref.update({"input_is_crop": meta["input_is_crop"],
                            "crop_box_xyxy_applied": meta["crop_box"],
                            "source_frame_size": [meta["frame_w"], meta["frame_h"]]})
        h, w = rgb.shape[:2]
        quality: Dict[str, Any] = {
            "input_width": w, "input_height": h, "min_side_px": min(h, w),
            "crop_area_px": h * w, "blur_score": round(laplacian_variance(rgb), 4),
            "blur_score_note": "variance of Laplacian on luma; uncalibrated, relative only",
        }
        if diagnostic_only:
            quality["purpose"] = "pipeline_diagnostic_only"
        det = (evidence or {}).get("detector")
        upstream: Dict[str, Any] = {}
        if det:
            quality["detector_confidence"] = det.get("confidence")
            upstream = {"detector_model_sha256": det.get("model_sha256"),
                        "detector_settings": det.get("settings"),
                        "nlp_model_version": None, "report_uri": evidence.get("report_uri")}
        embedding_id = "emb_" + _short_hash(
            self.space_id.encode(), evidence_id.encode(),
            json.dumps(meta["crop_box"]).encode(), pix_sha.encode())[:16]
        rec: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "embedding_id": embedding_id,
            "evidence_id": evidence_id,
            "modality": "visual",
            "space_id": self.space_id,
            "dim": int(vec.shape[0]),
            "dtype": "float32",
            "metric": spec.metric,
            "normalized": True,
            "norm": float(np.linalg.norm(vec.astype(np.float64))),
            "vector_ref": f"memory://{embedding_id}",
            "encoder": {
                "name": spec.name, "family": spec.family,
                "version_or_checkpoint": self.config.checkpoint_id,
                "weights_sha256": info.weights_sha256,
                "license": self.config.license or spec.license, "source_ref": spec.source_ref,
                "feature": spec.feature_name,
                **{k: v for k, v in info.extra.items() if k in ("config_sha256", "mock")},
            },
            "preprocessing": spec.preprocess.to_dict(),
            "inference": {"device": info.device, "precision": info.precision,
                          "framework_versions": dict(info.framework_versions),
                          "batch_size": batch_size},
            "source_ref": source_ref,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pipeline_version": self.config.pipeline_version or _pipeline_version(),
            "quality": quality,
            "warnings": list(warnings) + list(info.warnings)
                        + (["diagnostic_only_not_for_target_object_representation"]
                           if diagnostic_only else [])
                        + (["mock_backend_not_a_real_model"] if info.extra.get("mock") else []),
            "sensitivity": sens,
            "access_label": self.config.access_label,
        }
        if self.config.run_id:
            rec["created_by_run_id"] = self.config.run_id
        if upstream:
            rec["upstream"] = upstream
        return rec

    # --------------------------------------------------------------- public API
    def encode(self, image: Any, *, color_order: Optional[str] = None,
               evidence: Optional[Dict[str, Any]] = None,
               bbox_xyxy: Optional[Sequence[float]] = None,
               input_is_crop: bool = False,
               diagnostic_only: bool = False) -> VisualRepresentation:
        """Encode one image/frame/crop.

        image          PIL.Image, or uint8 ndarray (H,W) / (H,W,1) / (H,W,3) / (H,W,4).
        color_order    'RGB' or 'BGR' - REQUIRED for 3/4-channel ndarrays (OpenCV = 'BGR').
        evidence       optional Phase 1 VisualEvidenceItem dict (provenance).
        bbox_xyxy      optional pixel box; `image` is then the full frame.
        input_is_crop  True if `image` is already the crop (no bbox applied).
        diagnostic_only  True permits whole-frame evidence for PIPELINE DIAGNOSTICS only; the
                       record is marked `quality.purpose = pipeline_diagnostic_only`. Whole
                       frames are otherwise refused (target-object scope, Phase 2 decision).
        """
        rgb, meta, warns = self._prepare(image, color_order, evidence, bbox_xyxy, input_is_crop,
                                         diagnostic_only)
        x = preprocess(rgb, self.spec.preprocess)[None]
        vec = self._embed(x)[0]
        vec.setflags(write=False)
        return VisualRepresentation(vector=vec,
                                    record=self._record(vec, rgb, meta, warns, evidence, 1,
                                                        diagnostic_only))

    def encode_batch(self, images: Sequence[Any], *, color_order: Optional[str] = None,
                     evidences: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
                     bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
                     input_is_crop: bool = False, batch_size: Optional[int] = None,
                     diagnostic_only: bool = False, on_error: str = "collect") -> List[Union[VisualRepresentation, EncodeFailure]]:
        """Encode many inputs in chunks of `batch_size` (default `config.max_batch_size`).

        on_error='collect' (default): failed items become `EncodeFailure` at their index
        (Phase 1: best-effort per item, every skip reported). 'raise': first error propagates.
        A model-level failure in a chunk fails every item of that chunk.
        """
        if on_error not in ("collect", "raise"):
            raise ValueError("on_error must be 'collect' or 'raise'")
        n = len(images)
        evidences = evidences or [None] * n
        bboxes = bboxes or [None] * n
        if len(evidences) != n or len(bboxes) != n:
            raise InvalidInputError("evidences/bboxes length must match images")
        results: List[Optional[Union[VisualRepresentation, EncodeFailure]]] = [None] * n
        prepared = []
        for i in range(n):
            try:
                rgb, meta, warns = self._prepare(images[i], color_order, evidences[i],
                                                 bboxes[i], input_is_crop, diagnostic_only)
                prepared.append((i, rgb, meta, warns,
                                 preprocess(rgb, self.spec.preprocess)))
            except EncoderError as e:
                if on_error == "raise":
                    raise
                results[i] = EncodeFailure(i, type(e).__name__, str(e))
        step = max(1, int(batch_size or self.config.max_batch_size))
        for s in range(0, len(prepared), step):
            chunk = prepared[s:s + step]
            try:
                vecs = self._embed(np.stack([c[4] for c in chunk]))
            except EncoderError as e:
                if on_error == "raise":
                    raise
                for c in chunk:
                    results[c[0]] = EncodeFailure(c[0], type(e).__name__, str(e))
                continue
            for c, v in zip(chunk, vecs):
                v = v.copy()
                v.setflags(write=False)
                results[c[0]] = VisualRepresentation(
                    vector=v, record=self._record(v, c[1], c[2], c[3], evidences[c[0]], len(chunk),
                                              diagnostic_only))
        return results  # type: ignore[return-value]


def runtime_versions() -> Dict[str, str]:
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("torch", "transformers", "PIL"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pass
    return v
