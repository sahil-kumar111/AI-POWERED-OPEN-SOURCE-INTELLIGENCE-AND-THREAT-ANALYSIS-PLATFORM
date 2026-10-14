"""The two text-encoder CANDIDATES and their explicit, candidate-specific input handling (G2).

Neither adapter is "selected" or "final": the manifest keeps `status: candidate`,
`final_encoder: false`, and nothing here changes that. Model selection is a separate decision.

Candidate-specific handling lives in the two specs below and nowhere else:

  multilingual-e5-small (E5)
    * the caller MUST say whether the text is a `query` or a `passage`; the model input is
      `"query: " + text` or `"passage: " + text` (trailing space included);
    * limit: 512 tokens for the whole model input (prefix + text + special tokens). The manifest
      leaves `max_input_tokens` null ("decided in the encoder step") and records the verified
      `max_position_embeddings = 512`; this module decides: usable limit == 512 and cross-checks it
      against the manifest (it may never exceed the verified position capacity).
  paraphrase-multilingual-MiniLM-L12-v2 (MiniLM)
    * no prefix; a role is optional, validated and recorded, but does not change the model input;
    * limit: 128 tokens for the whole model input (Sentence-Transformers `max_seq_length`, the
      manifest's decided value), special tokens included.

Both reject over-limit input (`TextTooLongError`); neither ever truncates. Both mean-pool and the
encoder base class L2-normalises the output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional

from ..encoders.datatypes import EncoderConfig
from .encoder import TextBackend, TextEncoder, TextEncoderSpec
from .errors import EncoderIntegrityError, EncoderLoadError
from .manifest import DEFAULT_MANIFEST, load_manifest, verify_entry

E5_NAME = "multilingual-e5-small"
MINILM_NAME = "paraphrase-multilingual-minilm-l12-v2"

E5_SPEC = TextEncoderSpec(
    name=E5_NAME, family="E5",
    feature_name="mean pooling over last hidden state (attention-masked), then L2",
    expected_dim=384, max_input_tokens=512, role_required=True,
    role_prefixes={"query": "query: ", "passage": "passage: "},
    license="MIT (expected; unverified until recorded in the manifest)",
    source_ref="https://huggingface.co/intfloat/multilingual-e5-small")

MINILM_SPEC = TextEncoderSpec(
    name=MINILM_NAME, family="SBERT-MiniLM",
    feature_name="mean pooling over last hidden state (attention-masked), then L2",
    expected_dim=384, max_input_tokens=128, role_required=False, role_prefixes=None,
    license="Apache-2.0 (expected; unverified until recorded in the manifest)",
    source_ref="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class E5TextEncoder(TextEncoder):
    """multilingual-e5-small candidate: role prefix required, 512-token limit."""
    SPEC = E5_SPEC

    def __init__(self, config: Optional[EncoderConfig] = None, backend: Optional[TextBackend] = None,
                 *, file_sha256: Optional[Mapping[str, str]] = None):
        super().__init__(self.SPEC, config, backend, file_sha256=file_sha256)


class MiniLMTextEncoder(TextEncoder):
    """paraphrase-multilingual-MiniLM-L12-v2 candidate: no prefix, 128-token limit."""
    SPEC = MINILM_SPEC

    def __init__(self, config: Optional[EncoderConfig] = None, backend: Optional[TextBackend] = None,
                 *, file_sha256: Optional[Mapping[str, str]] = None):
        super().__init__(self.SPEC, config, backend, file_sha256=file_sha256)


ADAPTERS = {E5_NAME: E5TextEncoder, MINILM_NAME: MiniLMTextEncoder}


def spec_problems(spec: TextEncoderSpec, entry: Mapping[str, Any]) -> List[str]:
    """Disagreements between a code spec and its manifest entry (empty list = consistent)."""
    enc = entry["encoder"]
    out: List[str] = []

    def chk(label: str, code: Any, manifest: Any) -> None:
        if code != manifest:
            out.append(f"{label}: code {code!r} != manifest {manifest!r}")

    chk("family category", "TEXT_ENCODER", entry.get("family"))
    chk("expected_dim", spec.expected_dim, enc["expected_dim"])
    chk("pooling", spec.pooling, enc["pooling"])
    chk("token_limit_policy", spec.token_limit_policy, enc["token_limit_policy"])
    tpl = enc["input_template"]
    chk("role_required", spec.role_required, tpl["role_required"])
    chk("role prefixes", dict(spec.role_prefixes) if spec.role_prefixes else None, tpl["roles"])
    if enc["max_input_tokens"] is not None:
        chk("max_input_tokens", spec.max_input_tokens, enc["max_input_tokens"])
    else:                       # decided in code: must stay within the verified position capacity
        cap = enc["max_position_embeddings"]
        if cap is None or spec.max_input_tokens > cap:
            out.append(f"max_input_tokens: code {spec.max_input_tokens} exceeds manifest "
                       f"max_position_embeddings {cap!r}")
    if enc["sentence_bert_max_seq_length"] is not None:
        chk("sentence_bert_max_seq_length", spec.max_input_tokens, enc["sentence_bert_max_seq_length"])
    chk("source_ref", spec.source_ref, entry["source_url"])
    return out


def text_encoder_from_manifest(name: str, manifest_path: Optional[Path] = None,
                               device: str = "cpu", backend: Optional[TextBackend] = None
                               ) -> TextEncoder:
    """Build a LAZY adapter from a fully pinned, verified text-manifest entry.

    Refuses (EncoderLoadError / EncoderIntegrityError) unless the entry is pinned and every local
    file verifies, and unless the code spec agrees with the manifest. Nothing is loaded here: no
    torch import, no weights read into a model, no network. Against the committed manifest (no
    pins recorded) this always refuses by design.
    """
    mp = Path(manifest_path or DEFAULT_MANIFEST)
    m = load_manifest(mp)
    d = verify_entry(m, name, mp)                    # raises unless pinned and intact
    e = m["models"][name]
    cls = ADAPTERS.get(name)
    if cls is None:
        raise EncoderLoadError(f"no text adapter implemented for manifest entry {name!r}; "
                               f"available: {sorted(ADAPTERS)}")
    probs = spec_problems(cls.SPEC, e)
    if probs:
        raise EncoderIntegrityError(f"{name}: code spec disagrees with the manifest: " + "; ".join(probs))
    cfg = EncoderConfig(weights_dir=d, expected_weights_sha256=e["files"]["model.safetensors"]["sha256"],
                        checkpoint_id=f"{e['hf_id']}@{e['revision']}", device=device,
                        license=e["license"], revision=e["revision"])
    return cls(cfg, backend, file_sha256={f: v["sha256"] for f, v in e["files"].items()})
