"""Deterministic MOCK components for the Phase 3 text encoders. They are NOT E5/MiniLM and carry no
semantics: the "tokenizer" splits on whitespace, the "embedding" is a seeded random vector derived
from the exact input string, and the raw output is deliberately NOT normalised (so tests can prove
that the encoder normalises). Nothing here needs torch, weights or a network."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..encoders.datatypes import BackendInfo, EncoderConfig
from ..encoders.integrity import sha256_file
from ..text import manifest as T
from ..text.adapters import ADAPTERS, E5_NAME, MINILM_NAME
from ..text.contracts import text_sha256

SPECIAL_TOKENS = 2           # stands in for <s> ... </s> / [CLS] ... [SEP]


class MockTextBackend:
    """count_tokens = whitespace words + 2 special tokens. forward() records every string it is
    given (`forward_texts`) so tests can see exactly what reached the "model"."""

    def __init__(self, dim: int, seed: int = 0, *, fault: Optional[str] = None,
                 count_fault: Optional[str] = None, scale: float = 3.7, device: str = "cpu"):
        self.dim, self.seed, self.scale = dim, seed, scale
        self._fault, self._count_fault = fault, count_fault
        # fault: None | "nan" | "inf" | "zeros" | "wrong_shape" | "raise" | "int_dtype"
        # count_fault: None | "raise" | "wrong_len" | "zero" | "float"
        self.count_calls: List[List[str]] = []
        self.forward_texts: List[str] = []
        self.forward_batches: List[int] = []
        self.info = BackendInfo(
            weights_sha256=hashlib.sha256(f"mock-text-{dim}-{seed}".encode()).hexdigest(),
            device=device, precision="float32", dim=dim, framework_versions={"mock": "1"},
            extra={"mock": True,
                   "tokenizer_sha256": hashlib.sha256(b"mock-tokenizer").hexdigest()})

    @staticmethod
    def token_count(text: str) -> int:
        return len(text.split()) + SPECIAL_TOKENS

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        self.count_calls.append(list(texts))
        f = self._count_fault
        if f == "raise":
            raise RuntimeError("simulated tokenizer failure")
        counts = [self.token_count(t) for t in texts]
        if f == "wrong_len":
            return counts[:-1]
        if f == "zero":
            return [0 for _ in counts]
        if f == "float":
            return [float(c) for c in counts]       # type: ignore[misc]
        return counts

    def forward(self, texts: Sequence[str]) -> np.ndarray:
        self.forward_batches.append(len(texts))
        self.forward_texts.extend(texts)
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(f"{self.seed}|{t}".encode("utf-8", "surrogatepass"))
                                  .digest()[:8], "big")
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        out = (np.stack(rows) * self.scale).astype(np.float32)
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
        return out


def mock_text_encoder(candidate: str = "e5", seed: int = 0, fault: Optional[str] = None,
                      count_fault: Optional[str] = None, **cfg):
    """A real adapter class around a MockTextBackend. candidate: 'e5' | 'minilm'."""
    name = {"e5": E5_NAME, "minilm": MINILM_NAME}[candidate]
    cls = ADAPTERS[name]
    backend = MockTextBackend(cls.SPEC.expected_dim, seed, fault=fault, count_fault=count_fault)
    return cls(EncoderConfig(checkpoint_id="MOCK", **cfg), backend)


def make_text_item(text: str = "Example advisory text.", *, evidence_id: str = "te_0000000000000001",
                   language: str = "en", translation_text: Optional[str] = None, **over):
    """A valid Phase 1 TextEvidenceItem dict (synthetic content)."""
    item = {"schema_version": "0.1", "evidence_id": evidence_id, "source_kind": "advisory",
            "source": {"source_name": "synthetic", "source_url": None, "source_item_id": "item-1",
                       "retrieved_at": "2026-10-02T00:00:00Z", "published_at": None},
            "original_text": text, "original_text_sha256": text_sha256(text), "language": language,
            "truncated": False,
            "sensitivity": {"may_contain_personal_data": False, "retain_text": True}}
    if translation_text is not None:
        item["translation"] = {"target_language": "en", "text": translation_text,
                               "text_sha256": text_sha256(translation_text),
                               "method": "synthetic", "model_version": "none"}
    item.update(over)
    return item


# ------------------------------------------------------------------ synthetic pinned manifest
def _safetensors_bytes(dim: int = 384, rows: int = 4) -> bytes:
    nbytes = rows * dim * 4
    hdr = json.dumps({"embeddings.word_embeddings.weight":
                      {"dtype": "F32", "shape": [rows, dim], "data_offsets": [0, nbytes]}}).encode()
    return struct.pack("<Q", len(hdr)) + hdr + b"\x00" * nbytes


def make_pinned_synthetic_manifest(tmp_path: Path, name: str, *,
                                   extra_content: Optional[dict] = None) -> Path:
    """Copy of the committed text manifest, with tiny SYNTHETIC local files that satisfy its
    structural checks, then `record`-ed (revision/licence/hashes written to the copy ONLY).
    Used to exercise manifest integration offline; says nothing about real checkpoints.

    `extra_content` ({relative path: bytes|JSON-able}) replaces the content of a listed file or, for a
    path the manifest does not list, adds it as a role-'tokenizer' entry (G2.2 sidecar tests)."""
    m = json.loads(T.DEFAULT_MANIFEST.read_text())
    e = m["models"][name]
    d = tmp_path / e["local_dir"]
    seq = e["encoder"]["sentence_bert_max_seq_length"] or 512
    pooling = {"word_embedding_dimension": 384, "pooling_mode_cls_token": False,
               "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False,
               "pooling_mode_mean_sqrt_len_tokens": False}
    content = {
        "model.safetensors": _safetensors_bytes(),
        "config.json": {"model_type": "bert", "hidden_size": 384, "max_position_embeddings": 512,
                        "vocab_size": 4},
        "tokenizer.json": {"version": "1.0", "model": {"type": "Unigram", "vocab": []}},
        "tokenizer_config.json": {"tokenizer_class": "synthetic"},
        "special_tokens_map.json": {"pad_token": "<pad>"},
        "1_Pooling/config.json": pooling, "modules.json": [{"idx": 0}],
        "sentence_bert_config.json": {"max_seq_length": seq},
        "config_sentence_transformers.json": {"synthetic": True},
    }
    for rel, data in (extra_content or {}).items():
        content[rel] = data
        if rel not in e["files"]:
            e["files"][rel] = {"role": "tokenizer", "size": 1, "upstream_sha256": None, "sha256": "TO_RECORD"}
    for fname, f in e["files"].items():
        p = d / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        data = content[fname]
        p.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        f["size"] = p.stat().st_size
        if f["upstream_sha256"] is not None:
            f["upstream_sha256"] = sha256_file(p)
    mp = tmp_path / "text_models.manifest.json"
    mp.write_text(json.dumps(m, indent=2) + "\n")
    T.record_entry(mp, name, e["expected_revision"], e["expected_license_id"],
                   "Licence text copied from the model card (synthetic test)")
    return mp
