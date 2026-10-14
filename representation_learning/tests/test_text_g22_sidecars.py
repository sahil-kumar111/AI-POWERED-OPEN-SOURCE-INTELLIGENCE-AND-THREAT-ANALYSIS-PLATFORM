"""Phase 3 / G2.2: tokenizer-sidecar integrity of `TorchTextBackend`. All offline.

Policy under test (see text/torch_backend.py and docs/phase3_text_encoder.md section 6):
  * optional files the fast-tokenizer loader reads (`added_tokens.json`, `chat_template.jinja`,
    `additional_chat_templates/*.jinja`) must be listed in the manifest and SHA-256-pinned when present;
    present-but-unpinned is refused, before anything is loaded;
  * a `tokenizer_config.json` carrying `fast_tokenizer_files` is refused;
  * vocabulary/sentencepiece files are not required to be pinned.

IMPORTANT: that policy was derived by READING the transformers 5.18.0 source. These tests exercise
OUR checks only. Where they use the fake `torch`/`transformers` stand-ins they record arguments and
may touch files; they do NOT establish how the real Hugging Face loader behaves, and nothing here
loads a real tokenizer, model or weights."""

import json
import os
import types
from pathlib import Path

import pytest

from representation_learning.testing.text_mocks import make_pinned_synthetic_manifest
from representation_learning.text import torch_backend as TB
from representation_learning.text.adapters import (E5_NAME, E5_SPEC, MINILM_NAME, MINILM_SPEC,
                                                   text_encoder_from_manifest)
from representation_learning.text.errors import EncoderIntegrityError, EncoderLoadError

ROOT = Path(__file__).resolve().parents[1]
SIDECARS = {
    "added_tokens.json": b'{"<extra>": 5}',
    "chat_template.jinja": b"{{ messages }}",
    "additional_chat_templates/alt.jinja": b"{{ alt }}",
}
CANDS = [(E5_NAME, E5_SPEC), (MINILM_NAME, MINILM_SPEC)]
CAND_IDS = ["e5", "minilm"]


# ------------------------------------------------------------------ fake stand-ins (see docstring)
class Rec:
    def __init__(self):
        self.loads = 0
        self.during_model = None


def fake_stack(rec):
    class Tok:
        is_fast = True

    class Model:
        config = types.SimpleNamespace(hidden_size=384)

        def eval(self):
            return self

        def parameters(self):
            return []

        def to(self, device):
            return self

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kw):
            rec.loads += 1
            return Tok()

    class AutoModel:
        @staticmethod
        def from_pretrained(path, **kw):
            rec.loads += 1
            if rec.during_model:
                rec.during_model(Path(path))
            return Model()

    torch = types.SimpleNamespace(__version__="0-fake", cuda=types.SimpleNamespace(is_available=lambda: False))
    return torch, types.SimpleNamespace(__version__="0-fake", AutoTokenizer=AutoTokenizer, AutoModel=AutoModel)


@pytest.fixture
def rec(monkeypatch):
    r = Rec()
    monkeypatch.setattr(TB, "_import_torch_stack", lambda: fake_stack(r))
    return r


@pytest.fixture
def no_stack(monkeypatch):
    """Any attempt to import the model stack fails the test: the check must happen BEFORE it."""
    def boom():
        raise AssertionError("model stack imported before the sidecar check refused the directory")
    monkeypatch.setattr(TB, "_import_torch_stack", boom)


def build(tmp_path, name, extra=None):
    mp = make_pinned_synthetic_manifest(tmp_path, name, extra_content=extra)
    enc = text_encoder_from_manifest(name, mp)
    return enc.config, dict(enc.file_sha256), Path(enc.config.weights_dir)


def drop_in(d: Path, rel: str, data: bytes):
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# ------------------------------------------------------------------ present but unpinned -> refuse
@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
@pytest.mark.parametrize("rel", sorted(SIDECARS))
def test_unlisted_sidecar_is_refused_before_loading(tmp_path, no_stack, name, spec, rel):
    cfg, pins, d = build(tmp_path, name)
    drop_in(d, rel, SIDECARS[rel])
    with pytest.raises(EncoderIntegrityError, match="present but not pinned") as ei:
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    assert rel in str(ei.value)


@pytest.mark.parametrize("rel", ["additional_chat_templates/.hidden.jinja",
                                 "additional_chat_templates/UPPER.jinja",
                                 "additional_chat_templates/a.b.jinja"])
def test_every_jinja_file_in_the_template_directory_counts(tmp_path, no_stack, rel):
    cfg, pins, d = build(tmp_path, E5_NAME)
    drop_in(d, rel, b"x")
    with pytest.raises(EncoderIntegrityError, match="present but not pinned"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


def test_non_jinja_files_in_the_template_directory_are_not_loader_inputs(tmp_path, rec):
    cfg, pins, d = build(tmp_path, E5_NAME)
    drop_in(d, "additional_chat_templates/readme.txt", b"x")
    TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)                 # not a `*.jinja` -> ignored


@pytest.mark.parametrize("rel", ["added_tokens.json", "chat_template.jinja"])
def test_symlinked_sidecar_counts_as_present(tmp_path, no_stack, rel):
    cfg, pins, d = build(tmp_path, E5_NAME)
    target = d / "elsewhere.bin"
    target.write_bytes(SIDECARS[rel])
    os.symlink(target, d / rel)
    with pytest.raises(EncoderIntegrityError, match="present but not pinned"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


def test_symlinked_template_directory_is_refused(tmp_path, no_stack):
    cfg, pins, d = build(tmp_path, E5_NAME)
    real = tmp_path / "somewhere"
    real.mkdir()
    (real / "x.jinja").write_bytes(b"x")
    os.symlink(real, d / "additional_chat_templates")
    with pytest.raises(EncoderIntegrityError, match="symlink"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


@pytest.mark.parametrize("rel", sorted(SIDECARS))
def test_removing_the_pin_of_a_present_sidecar_is_refused(tmp_path, no_stack, rel):
    cfg, pins, d = build(tmp_path, E5_NAME, extra={rel: SIDECARS[rel]})
    assert rel in pins
    with pytest.raises(EncoderIntegrityError, match="present but not pinned"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256={k: v for k, v in pins.items() if k != rel})
    for blank in (None, ""):                         # a key with no usable pin is refused as well
        with pytest.raises(EncoderIntegrityError, match="not pinned|64 hex"):
            TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=dict(pins, **{rel: blank}))


@pytest.mark.parametrize("rel", sorted(SIDECARS))
def test_without_manifest_pins_any_sidecar_is_refused(tmp_path, no_stack, rel):
    cfg, _, d = build(tmp_path, E5_NAME)
    drop_in(d, rel, SIDECARS[rel])
    with pytest.raises(EncoderIntegrityError, match="present but not pinned"):
        TB.TorchTextBackend(cfg, E5_SPEC)               # only expected_weights_sha256 -> nothing to pin against


# ------------------------------------------------------------------ listed and pinned -> accepted
@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
@pytest.mark.parametrize("rel", sorted(SIDECARS))
def test_listed_and_pinned_sidecar_is_accepted_through_the_existing_manifest_schema(tmp_path, rec, name, spec, rel):
    """The G1.2 manifest schema already allows extra listed files: `record` pins them and
    `verify` (inside text_encoder_from_manifest) checks them. No manifest change was needed."""
    cfg, pins, d = build(tmp_path, name, extra={rel: SIDECARS[rel]})
    assert rel in pins and len(pins[rel]) == 64
    TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    assert rec.loads == 2


def test_all_sidecars_together_are_accepted_when_pinned(tmp_path, rec):
    cfg, pins, _ = build(tmp_path, E5_NAME, extra=dict(SIDECARS))
    TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


@pytest.mark.parametrize("rel", sorted(SIDECARS))
def test_pinned_sidecar_tampering_is_refused_before_loading(tmp_path, no_stack, rel):
    cfg, pins, d = build(tmp_path, E5_NAME, extra={rel: SIDECARS[rel]})
    data = bytearray((d / rel).read_bytes())
    data[-2] ^= 0x01                                                    # same size, different content
    (d / rel).write_bytes(bytes(data))
    with pytest.raises(EncoderIntegrityError, match="SHA-256"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


@pytest.mark.parametrize("rel", sorted(SIDECARS))
def test_pinned_sidecar_missing_or_symlinked_is_refused(tmp_path, no_stack, rel):
    cfg, pins, d = build(tmp_path, E5_NAME, extra={rel: SIDECARS[rel]})
    p = d / rel
    data = p.read_bytes()
    p.unlink()
    with pytest.raises(EncoderLoadError, match="missing or not a regular file"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)
    (d / "twin.bin").write_bytes(data)
    os.symlink(d / "twin.bin", p)                                       # identical bytes, but a symlink
    with pytest.raises(EncoderLoadError, match="missing or not a regular file"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


# ------------------------------------------------------------------ changes while loading
@pytest.mark.parametrize("rel", sorted(SIDECARS))
def test_pinned_sidecar_changed_during_load_is_detected(tmp_path, rec, rel):
    cfg, pins, _ = build(tmp_path, E5_NAME, extra={rel: SIDECARS[rel]})

    def tamper(path):
        data = bytearray((path / rel).read_bytes())
        data[-2] ^= 0x01
        (path / rel).write_bytes(bytes(data))
    rec.during_model = tamper
    with pytest.raises(EncoderIntegrityError, match="changed during load") as ei:
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)
    assert rel in str(ei.value)


@pytest.mark.parametrize("rel", sorted(SIDECARS))
def test_sidecar_appearing_during_load_is_detected(tmp_path, rec, rel):
    cfg, pins, _ = build(tmp_path, E5_NAME)
    rec.during_model = lambda path: drop_in(path, rel, SIDECARS[rel])
    with pytest.raises(EncoderIntegrityError, match="sidecar state changed during load") as ei:
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)
    assert rel in str(ei.value)


def test_fast_tokenizer_files_key_appearing_during_load_is_detected(tmp_path, rec):
    cfg, pins, _ = build(tmp_path, E5_NAME)
    cfg_path = Path(cfg.weights_dir) / "tokenizer_config.json"
    rec.during_model = lambda path: cfg_path.write_text(json.dumps(
        {"tokenizer_class": "synthetic", "fast_tokenizer_files": ["tokenizer.json"]}))
    with pytest.raises(EncoderIntegrityError, match="changed during load"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


# ------------------------------------------------------------------ versioned tokenizer redirect
@pytest.mark.parametrize("value", [["tokenizer.json", "tokenizer.5.0.0.json"], [], None, "tokenizer.json"],
                         ids=["list", "empty-list", "null", "string"])
def test_fast_tokenizer_files_key_is_refused_even_when_pinned(tmp_path, no_stack, value):
    cfg, pins, _ = build(tmp_path, E5_NAME, extra={
        "tokenizer_config.json": {"tokenizer_class": "synthetic", "fast_tokenizer_files": value}})
    with pytest.raises(EncoderIntegrityError, match="fast_tokenizer_files"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


def test_fast_tokenizer_files_key_is_refused_without_pins_too(tmp_path, no_stack):
    cfg, _, d = build(tmp_path, E5_NAME)
    (d / "tokenizer_config.json").write_text(json.dumps({"fast_tokenizer_files": ["tokenizer.json"]}))
    with pytest.raises(EncoderIntegrityError, match="fast_tokenizer_files"):
        TB.TorchTextBackend(cfg, E5_SPEC)


@pytest.mark.parametrize("raw,match", [(b"{broken", "not valid JSON"), (b"[1, 2]", "JSON object")])
def test_unparseable_tokenizer_config_is_refused_without_pins(tmp_path, no_stack, raw, match):
    cfg, _, d = build(tmp_path, E5_NAME)
    (d / "tokenizer_config.json").write_bytes(raw)
    with pytest.raises(EncoderIntegrityError, match=match):
        TB.TorchTextBackend(cfg, E5_SPEC)


def test_tokenizer_config_without_the_key_loads(tmp_path, rec):
    cfg, pins, _ = build(tmp_path, E5_NAME, extra={
        "tokenizer_config.json": {"tokenizer_class": "synthetic", "model_max_length": 512}})
    TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


# ------------------------------------------------------------------ policy: vocab files not required
@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
@pytest.mark.parametrize("rel", ["sentencepiece.bpe.model", "vocab.txt", "vocab.json", "merges.txt",
                                 "spiece.model", "tokenizer.model", "README.md"])
def test_unlisted_vocabulary_files_are_not_refused_policy(tmp_path, rec, name, spec, rel):
    """POLICY, not loader behaviour: with a pinned tokenizer.json the transformers 5.18.0 source
    resolves these files but does not open them, so they are not required to be pinned (real
    checkpoint snapshots contain e.g. sentencepiece.bpe.model, which the manifest does not list)."""
    cfg, pins, d = build(tmp_path, name)
    drop_in(d, rel, b"unlisted vocabulary-type file")
    TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    assert rec.loads == 2


def test_sidecar_name_sets_match_the_documented_policy():
    assert TB.TOKENIZER_SIDECAR_FILES == ("added_tokens.json", "chat_template.jinja")
    assert TB.CHAT_TEMPLATE_DIR == "additional_chat_templates"
    assert TB.VERSIONED_TOKENIZER_KEY == "fast_tokenizer_files"
    doc = (ROOT / "docs" / "phase3_text_encoder.md").read_text()
    for token in (*TB.TOKENIZER_SIDECAR_FILES, TB.CHAT_TEMPLATE_DIR, TB.VERSIONED_TOKENIZER_KEY):
        assert token in doc


# ------------------------------------------------------------------ nothing else regressed
def test_clean_directory_still_loads_and_use_fast_is_still_passed(tmp_path, monkeypatch):
    calls = []
    r = Rec()
    torch, tf = fake_stack(r)
    orig = tf.AutoTokenizer.from_pretrained
    tf.AutoTokenizer.from_pretrained = staticmethod(lambda path, **kw: (calls.append(kw), orig(path, **kw))[1])
    monkeypatch.setattr(TB, "_import_torch_stack", lambda: (torch, tf))
    cfg, pins, _ = build(tmp_path, E5_NAME)
    TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)
    assert calls == [{"local_files_only": True, "trust_remote_code": False, "use_fast": True}]
