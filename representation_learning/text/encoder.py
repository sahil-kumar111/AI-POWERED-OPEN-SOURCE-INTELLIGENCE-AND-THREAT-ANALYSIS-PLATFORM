"""Common text encoder interface: `encode(item, role=...) -> TextRepresentation` (Phase 3, G2).

Mirrors the visual `VisualEncoder` (encoders/base.py) where it can:
  * backends are lazy: nothing heavy is imported or loaded until first use; unit tests inject a mock
    backend, real backends load LOCAL weights only (see `text/torch_backend.py`);
  * output records follow the Phase 1 `EmbeddingRecord` contract (modality "text");
  * embeddings are float32, finite and L2-normalised; different encoders = different `space_id`;
    this module offers no comparison API.

Text-specific rules (docs/phase3_text_encoder.md):
  * the input is a validated `TextEvidenceItem`; evidence ids come from the upstream adapter and are
    only validated here; raw strings are refused;
  * the text is passed to the tokenizer VERBATIM after the (candidate-specific) role prefix;
  * the token limit is enforced on the full model input (prefix + text + special tokens) BEFORE
    inference, and over-limit inputs are REJECTED (`TextTooLongError`), never truncated;
  * `language` is recorded as metadata only and never changes how the text is processed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union

import numpy as np

from ..encoders.base import NORM_TOLERANCE, SCHEMA_VERSION, _pipeline_version, _short_hash
from ..encoders.datatypes import BackendInfo, EncodeFailure, EncoderConfig
from .contracts import TEXT_VARIANTS, SelectedText, TextEvidenceItem, coerce_item, text_sha256
from .errors import (EmbeddingValidationError, EncoderError, EncoderIntegrityError,
                     InvalidInputError, TextTooLongError)

ROLES = ("query", "passage")
LANGUAGE_HANDLING = ("recorded as metadata only; never used to select a model, add a prefix, "
                     "translate, filter or change the text")


class TextBackend(Protocol):
    """Runs tokenizer + network. Both methods take the FINAL model-input strings (role prefix
    already applied) and must never truncate."""

    info: BackendInfo

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        """Length of each string as the model would see it: tokens INCLUDING special tokens,
        without truncation."""

    def forward(self, texts: Sequence[str]) -> np.ndarray:
        """(N, dim) pooled (mean) features, not necessarily normalised. No truncation."""


@dataclass(frozen=True)
class TextEncoderSpec:
    """What a text-encoder candidate extracts and how its input is formed. Fixed by code and
    cross-checked against the manifest (`text/adapters.py`), not chosen by the caller."""

    name: str                                  # manifest entry name
    family: str
    feature_name: str
    expected_dim: int
    max_input_tokens: int                      # counts prefix + text + special tokens
    role_required: bool
    role_prefixes: Optional[Mapping[str, str]]  # {"query": ..., "passage": ...} or None
    license: str
    source_ref: str
    pooling: str = "mean"
    metric: str = "cosine"
    token_limit_policy: str = "reject_over_limit"

    def preprocessing_dict(self) -> Dict[str, Any]:
        return {
            "id": f"{self.name}-text-v1",
            "text_normalization": "none (text is passed verbatim after the role prefix)",
            "role_required": self.role_required,
            "role_prefixes": dict(self.role_prefixes) if self.role_prefixes else None,
            "language_handling": LANGUAGE_HANDLING,
            "tokenizer": {"source": "checkpoint tokenizer.json (local)", "add_special_tokens": True},
            "max_input_tokens": self.max_input_tokens,
            "max_input_tokens_counts": "role prefix + text + special tokens",
            "token_limit_policy": self.token_limit_policy,
            "truncation": "never",
            "pooling": self.pooling,
            "output_normalization": "l2",
            "output_layout": "vector",
            "output_dtype": "float32",
        }


@dataclass(frozen=True)
class TextRepresentation:
    """One embedding + its Phase 1 `EmbeddingRecord` (vector held in memory; `vector_ref` is only
    an in-memory placeholder). The record holds a hash and ids, not a copy of the text."""

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
class _Prepared:
    index: int
    item: TextEvidenceItem
    selected: SelectedText
    role: Optional[str]
    formatted: str
    formatted_sha256: str


def _broadcast(value: Any, n: int, name: str) -> List[Any]:
    """One value for all items, or a list/tuple with one value per item. Anything else is treated
    as a single value and rejected later by the per-item validation (never silently coerced)."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise InvalidInputError(f"{name} length {len(value)} does not match items ({n})")
        return list(value)
    return [value] * n


class TextEncoder:
    """Base class. Subclasses (`text/adapters.py`) fix the spec; real weights come from
    `_make_backend()`, tests/mocks inject `backend=`."""

    def __init__(self, spec: TextEncoderSpec, config: Optional[EncoderConfig] = None,
                 backend: Optional[TextBackend] = None, *,
                 file_sha256: Optional[Mapping[str, str]] = None):
        self.spec = spec
        self.config = config or EncoderConfig()
        self.file_sha256 = dict(file_sha256) if file_sha256 else None
        self._backend: Optional[TextBackend] = None
        self._space_id: Optional[str] = None
        if backend is not None:
            self._adopt_backend(backend)

    # ------------------------------------------------------------------ loading
    def _make_backend(self) -> TextBackend:
        from .torch_backend import TorchTextBackend      # lazy: torch/transformers only here
        return TorchTextBackend(self.config, self.spec, file_sha256=self.file_sha256)

    def _adopt_backend(self, backend: TextBackend) -> None:
        if backend.info.dim != self.spec.expected_dim:
            raise EncoderIntegrityError(
                f"backend dim {backend.info.dim} != expected {self.spec.expected_dim} "
                f"for {self.spec.name}")
        self._backend = backend
        self._space_id = None

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> BackendInfo:
        """Load the model now (otherwise done lazily on first encode)."""
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
        """Hash of encoder identity + checkpoint + weights/tokenizer hashes + preprocessing
        (prefix templates, token limit, pooling) + normalisation. Changes if any of them change.
        The caller's `role` is NOT part of it: query and passage vectors share one space."""
        if self._space_id is None:
            info = self.load()
            ident = {
                "encoder": self.spec.name, "family": self.spec.family,
                "checkpoint": self.config.checkpoint_id,
                "weights_sha256": info.weights_sha256,
                "tokenizer_sha256": info.extra.get("tokenizer_sha256"),
                "feature": self.spec.feature_name, "dim": info.dim,
                "preprocessing": self.spec.preprocessing_dict(),
                "precision": info.precision, "normalization": "l2", "metric": self.spec.metric,
            }
            self._space_id = "space_" + _short_hash(json.dumps(ident, sort_keys=True).encode())[:16]
        return self._space_id

    # ------------------------------------------------------------ input handling
    def _check_role(self, role: Any) -> Optional[str]:
        """Candidate-specific role handling is driven by the spec:
        role_required -> exactly 'query' or 'passage'; otherwise a role is optional, validated if
        given, recorded, and has NO effect on the model input."""
        if role is None:
            if self.spec.role_required:
                raise InvalidInputError(
                    f"{self.spec.name} requires role 'query' or 'passage' (its input prefix depends "
                    "on the caller's role); none was given")
            return None
        if not isinstance(role, str) or role not in ROLES:
            raise InvalidInputError(f"role must be one of {list(ROLES)} or None, got {role!r}")
        return role

    def format_input(self, text: str, role: Optional[str]) -> str:
        """The exact string handed to the tokenizer: role prefix (if the candidate has one) + text,
        nothing else - no stripping, no normalisation."""
        role = self._check_role(role)
        if self.spec.role_prefixes is None or role is None:
            return text
        return self.spec.role_prefixes[role] + text

    def _prepare(self, index: int, item: Any, role: Any, text_variant: Any) -> _Prepared:
        it = coerce_item(item)
        if text_variant not in TEXT_VARIANTS:
            raise InvalidInputError(f"text_variant must be one of {list(TEXT_VARIANTS)}, "
                                    f"got {text_variant!r}")
        role = self._check_role(role)
        sel = it.select(text_variant)
        formatted = self.format_input(sel.text, role)
        return _Prepared(index, it, sel, role, formatted, text_sha256(formatted))

    # ------------------------------------------------------------- core compute
    def _token_counts(self, texts: Sequence[str]) -> List[int]:
        self.load()
        assert self._backend is not None
        try:
            counts = self._backend.count_tokens(list(texts))
        except EncoderError:
            raise
        except Exception as e:
            raise EmbeddingValidationError(
                f"backend tokenization failed: {type(e).__name__}: {e}") from e
        if (not isinstance(counts, (list, tuple)) or len(counts) != len(texts)
                or not all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) and c >= 1
                           for c in counts)):
            raise EmbeddingValidationError(
                f"backend count_tokens must return {len(texts)} positive integers, got {counts!r}")
        return [int(c) for c in counts]

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        self.load()
        assert self._backend is not None
        try:
            raw = self._backend.forward(list(texts))
        except EncoderError:
            raise
        except Exception as e:
            raise EmbeddingValidationError(f"backend forward failed: {type(e).__name__}: {e}") from e
        raw = np.asarray(raw)
        dim = self.spec.expected_dim
        if raw.ndim != 2 or raw.shape != (len(texts), dim):
            raise EmbeddingValidationError(f"expected output shape {(len(texts), dim)}, got {raw.shape}")
        if not np.issubdtype(raw.dtype, np.floating):
            raise EmbeddingValidationError(f"non-floating output dtype {raw.dtype}")
        raw64 = raw.astype(np.float64)
        if not np.isfinite(raw64).all():
            raise EmbeddingValidationError("model output contains NaN or Inf")
        norms = np.linalg.norm(raw64, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise EmbeddingValidationError("model output has (near-)zero norm; cannot normalise")
        out = (raw64 / norms).astype(np.float32)
        if not np.isfinite(out).all() or (np.abs(np.linalg.norm(out.astype(np.float64), axis=1) - 1.0)
                                         > NORM_TOLERANCE).any():
            raise EmbeddingValidationError("normalised output failed finite/unit-norm check")
        return out

    # ------------------------------------------------------------------ records
    def _record(self, vec: np.ndarray, p: _Prepared, token_count: int,
                batch_size: int) -> Dict[str, Any]:
        info, spec, item = self.info, self.spec, p.item
        embedding_id = "emb_" + _short_hash(
            self.space_id.encode(), item.evidence_id.encode(), p.selected.variant.encode(),
            p.formatted_sha256.encode())[:16]
        warnings: List[str] = []
        if item.truncated:
            warnings.append("upstream_text_truncated")
        if p.selected.variant == "translation":
            warnings.append("translation_variant_embedded")
        preprocessing = spec.preprocessing_dict()
        preprocessing["tokenizer"] = dict(
            preprocessing["tokenizer"],
            sha256=info.extra.get("tokenizer_sha256"))
        rec: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "embedding_id": embedding_id,
            "evidence_id": item.evidence_id,
            "modality": "text",
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
                **{k: v for k, v in info.extra.items()
                   if k in ("config_sha256", "tokenizer_sha256", "mock")},
            },
            "preprocessing": preprocessing,
            "inference": {"device": info.device, "precision": info.precision,
                          "framework_versions": dict(info.framework_versions),
                          "batch_size": batch_size},
            "source_ref": {
                "evidence_id": item.evidence_id, "kind": "text_evidence_item",
                "source_kind": item.source_kind,
                "source_name": item.source.get("source_name"),
                "source_item_id": item.source.get("source_item_id"),
                "original_text_sha256": item.original_text_sha256,
                "language": item.language,
                "encoded_text_language": p.selected.language,
                "encoded_text_sha256": p.formatted_sha256,
                "encoded_text_note": "SHA-256 of the exact model input (role prefix + text)",
                "input_role": p.role,
                "role_prefix_applied": bool(spec.role_prefixes and p.role is not None),
            },
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pipeline_version": self.config.pipeline_version or _pipeline_version(),
            "quality": {
                "char_count": len(p.selected.text),
                "truncated": item.truncated,
                "encoder_truncated": False,
                "token_count": token_count,
                "max_input_tokens": spec.max_input_tokens,
                "token_count_includes": "role prefix + text + special tokens",
            },
            "warnings": warnings + list(info.warnings)
                        + (["mock_backend_not_a_real_model"] if info.extra.get("mock") else []),
            "text_variant": p.selected.variant,
            "upstream": {"detector_model_sha256": None, "detector_settings": None,
                         "nlp_model_version": None,   # never read from item.nlp_context (G2.1)
                         "report_uri": None},
            "sensitivity": dict(item.sensitivity),
            "access_label": self.config.access_label,
        }
        if self.config.run_id:
            rec["created_by_run_id"] = self.config.run_id
        return rec

    # --------------------------------------------------------------- public API
    def encode(self, item: Union[TextEvidenceItem, Mapping[str, Any]], *,
               role: Optional[str] = None, text_variant: str = "original_text") -> TextRepresentation:
        """Encode one `TextEvidenceItem` (or its dict form).

        role          'query' | 'passage'. REQUIRED when the candidate needs a prefix (E5); optional
                      and without effect on the model input otherwise (it is still recorded).
        text_variant  'original_text' (default) or 'translation' (only if the item has one). The
                      record always states which one was embedded.
        Raises InvalidInputError (contract/role/variant), TextTooLongError (over the token limit),
        EmbeddingValidationError (bad model output).
        """
        if isinstance(role, (list, tuple)) or isinstance(text_variant, (list, tuple)):
            raise InvalidInputError("encode() takes a single role/text_variant; use encode_batch "
                                    "for per-item values")
        return self.encode_batch([item], role=role, text_variant=text_variant,
                                 on_error="raise")[0]       # type: ignore[return-value]

    def encode_batch(self, items: Sequence[Union[TextEvidenceItem, Mapping[str, Any]]], *,
                     role: Union[None, str, Sequence[Optional[str]]] = None,
                     text_variant: Union[str, Sequence[str]] = "original_text",
                     batch_size: Optional[int] = None, on_error: str = "collect"
                     ) -> List[Union[TextRepresentation, EncodeFailure]]:
        """Encode many items in chunks of `batch_size` (default `config.max_batch_size`).

        `role` / `text_variant` may be one value for all items or one per item.
        on_error='collect' (default): a failed item becomes an `EncodeFailure` at its index;
        'raise': the first error propagates. A model-level failure in a chunk fails every item of
        that chunk. Over-limit items fail individually and never reach the model.
        Items sharing an `evidence_id` but carrying different `original_text_sha256` are all
        rejected (an id derived from the text hash cannot name two texts).
        """
        if on_error not in ("collect", "raise"):
            raise ValueError("on_error must be 'collect' or 'raise'")
        n = len(items)
        roles = _broadcast(role, n, "role")
        variants = _broadcast(text_variant, n, "text_variant")
        results: List[Optional[Union[TextRepresentation, EncodeFailure]]] = [None] * n

        def fail(i: int, e: EncoderError) -> None:
            if on_error == "raise":
                raise e
            results[i] = EncodeFailure(i, type(e).__name__, str(e))

        prepared: List[_Prepared] = []
        for i in range(n):
            try:
                prepared.append(self._prepare(i, items[i], roles[i], variants[i]))
            except EncoderError as e:
                fail(i, e)
        by_id: Dict[str, set] = {}
        for p in prepared:
            by_id.setdefault(p.item.evidence_id, set()).add(p.item.original_text_sha256)
        clash = {k for k, v in by_id.items() if len(v) > 1}
        ok: List[_Prepared] = []
        for p in prepared:
            if p.item.evidence_id in clash:
                fail(p.index, InvalidInputError(
                    f"evidence_id {p.item.evidence_id!r} is used for different texts in this batch "
                    "(ids are derived upstream from the text hash)"))
            else:
                ok.append(p)

        step = max(1, int(batch_size or self.config.max_batch_size))
        limit = self.spec.max_input_tokens
        for s in range(0, len(ok), step):
            chunk = ok[s:s + step]
            try:
                counts = self._token_counts([p.formatted for p in chunk])
            except EncoderError as e:
                for p in chunk:
                    fail(p.index, e)
                continue
            fit: List[tuple] = []
            for p, c in zip(chunk, counts):
                if c > limit:
                    fail(p.index, TextTooLongError(
                        f"{p.item.evidence_id}: model input has {c} tokens (limit {limit}, counted "
                        "with role prefix and special tokens); text is rejected, never truncated",
                        token_count=c, limit=limit))
                else:
                    fit.append((p, c))
            if not fit:
                continue
            try:
                vecs = self._embed([p.formatted for p, _ in fit])
            except EncoderError as e:
                for p, _ in fit:
                    fail(p.index, e)
                continue
            for (p, c), v in zip(fit, vecs):
                v = v.copy()
                v.setflags(write=False)
                results[p.index] = TextRepresentation(vector=v, record=self._record(v, p, c, len(fit)))
        return results  # type: ignore[return-value]
