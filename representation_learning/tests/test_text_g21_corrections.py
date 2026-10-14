"""Phase 3 / G2.1: regression tests for the three G2 review findings. All offline.

  1. `evidence_id` must be exactly `te_` + 16 lowercase hex characters.
  2. `upstream.nlp_model_version` is always null; `nlp_context` is never read by the encoder.
  3. The real backend uses `use_fast=True`, verifies every manifest-listed file against its pin before
     loading, and detects changes made to those files while loading.

Item 3 uses FAKE `torch` / `transformers` stand-ins injected into the backend module. No real
model, tokenizer or weights are downloaded, loaded or run; the fakes only record the arguments they
receive and may modify files to simulate a change during loading."""

import ast
import json
import os
import types
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from representation_learning.encoders.datatypes import EncodeFailure
from representation_learning.testing.text_mocks import (make_pinned_synthetic_manifest,
                                                        make_text_item, mock_text_encoder)
from representation_learning.text import torch_backend as TB
from representation_learning.text.adapters import (E5_NAME, E5_SPEC, MINILM_NAME, MINILM_SPEC,
                                                   text_encoder_from_manifest)
from representation_learning.text.contracts import TextEvidenceItem, text_evidence_problems
from representation_learning.text.errors import (EncoderIntegrityError, EncoderLoadError,
                                                 InvalidInputError)
from representation_learning.text.manifest import REQUIRED_FILES

ROOT = Path(__file__).resolve().parents[1]
CANDS = ["e5", "minilm"]


def role_for(c):
    return "passage" if c == "e5" else None


# ====================================================================== 1. evidence_id format
GOOD_IDS = ["te_0123456789abcdef", "te_0000000000000000", "te_ffffffffffffffff",
            "te_a1b2c3d4e5f60718"]
BAD_IDS = [
    ("old-8-digit-form", "te_00000001"),
    ("15-hex", "te_0123456789abcde"),
    ("17-hex", "te_0123456789abcdef0"),
    ("uppercase-hex", "te_0123456789ABCDEF"),
    ("uppercase-prefix", "TE_0123456789abcdef"),
    ("non-hex-char", "te_0123456789abcdeg"),
    ("no-prefix", "0123456789abcdef"),
    ("wrong-prefix", "ve_0123456789abcdef"),
    ("no-underscore", "te-0123456789abcdef"),
    ("trailing-newline", "te_0123456789abcdef\n"),
    ("leading-space", " te_0123456789abcdef"),
    ("trailing-space", "te_0123456789abcdef "),
    ("embedded-nul", "te_0123456789abc\x00ef"),
    ("fullwidth-digits", "te_０１２３４５６７８９abcdef"),
    ("arabic-indic-digits", "te_٠١٢٣٤٥٦٧٨٩abcdef"),
    ("double-prefix", "te_te_0123456789abcdef"),
    ("empty", ""),
    ("long", "te_" + "a" * 5000),
    ("old-free-form", "upstream-id/with:chars#1"),
    ("not-a-string-int", 1234567890123456),
    ("not-a-string-none", None),
    ("not-a-string-bytes", b"te_0123456789abcdef"),
]


@pytest.mark.parametrize("eid", GOOD_IDS)
def test_exact_format_evidence_ids_are_accepted(eid):
    d = make_text_item(evidence_id=eid)
    assert text_evidence_problems(d) == []
    assert TextEvidenceItem.from_dict(d).evidence_id == eid


@pytest.mark.parametrize("eid", [b[1] for b in BAD_IDS], ids=[b[0] for b in BAD_IDS])
def test_any_other_evidence_id_is_rejected_everywhere(eid):
    d = make_text_item(evidence_id=eid)
    probs = text_evidence_problems(d)
    assert any(p.startswith("evidence_id") for p in probs), probs
    with pytest.raises(InvalidInputError, match="evidence_id"):
        TextEvidenceItem.from_dict(d)
    with pytest.raises(InvalidInputError, match="evidence_id"):
        TextEvidenceItem(**d)                                            # direct construction too


def test_error_message_names_the_format_and_truncates_a_long_untrusted_id():
    probs = text_evidence_problems(make_text_item(evidence_id="te_" + "z" * 5000))
    msg = next(p for p in probs if p.startswith("evidence_id"))
    assert "te_" in msg and "16 lowercase hexadecimal" in msg and len(msg) < 300


@pytest.mark.parametrize("cand", CANDS)
def test_encoder_refuses_a_malformed_id_before_the_model_runs(cand):
    enc = mock_text_encoder(cand)
    with pytest.raises(InvalidInputError, match="evidence_id"):
        enc.encode(make_text_item(evidence_id="te_00000001"), role=role_for(cand))
    res = enc.encode_batch([make_text_item("a", evidence_id="te_0123456789abcdef"),
                            make_text_item("b", evidence_id="te_0123456789ABCDEF")],
                           role=role_for(cand))
    assert not isinstance(res[0], EncodeFailure) and isinstance(res[1], EncodeFailure)
    assert enc._backend.forward_texts and all("b" != t.split(": ")[-1] for t in enc._backend.forward_texts)


@pytest.mark.parametrize("cand", CANDS)
def test_id_is_validated_for_form_only_not_for_derivation(cand):
    """Two different well-formed ids for identical text are both accepted: the encoder does not
    (and does not claim to) verify how the upstream id was derived."""
    enc = mock_text_encoder(cand)
    a = enc.encode(make_text_item("same text", evidence_id="te_1111111111111111"), role=role_for(cand))
    b = enc.encode(make_text_item("same text", evidence_id="te_2222222222222222"), role=role_for(cand))
    assert a.record["evidence_id"] == "te_1111111111111111"
    assert b.record["evidence_id"] == "te_2222222222222222"


def test_strict_id_format_still_satisfies_the_phase1_schema(schemas):
    v = Draft202012Validator(schemas["text_evidence_item.schema.json"])
    for eid in GOOD_IDS:
        assert not list(v.iter_errors(make_text_item(evidence_id=eid)))


# ====================================================================== 2. nlp_model_version
NLP_VARIANTS = [
    ("string-version", {"nlp_model_version": "n9", "keywords": ["k"]}),
    ("empty-string-version", {"nlp_model_version": ""}),
    ("non-string-version", {"nlp_model_version": 7}),
    ("version-null", {"nlp_model_version": None}),
    ("other-keys-only", {"entities": ["x"], "sentiment": 0.2}),
    ("empty-context", {}),
]


@pytest.mark.parametrize("cand", CANDS)
@pytest.mark.parametrize("nlp", [v[1] for v in NLP_VARIANTS], ids=[v[0] for v in NLP_VARIANTS])
@pytest.mark.parametrize("variant", ["original_text", "translation"])
def test_nlp_model_version_is_null_regardless_of_nlp_context(cand, nlp, variant):
    enc = mock_text_encoder(cand)
    item = make_text_item("plain text", translation_text="texte", nlp_context=nlp)
    rec = enc.encode(item, role=role_for(cand), text_variant=variant).record
    assert rec["upstream"]["nlp_model_version"] is None
    assert "n9" not in json.dumps(rec)                                   # not propagated anywhere else


@pytest.mark.parametrize("cand", CANDS)
def test_nlp_model_version_is_null_without_nlp_context_and_for_prebuilt_items(cand):
    enc = mock_text_encoder(cand)
    absent = enc.encode(make_text_item("x"), role=role_for(cand)).record
    prebuilt = enc.encode(TextEvidenceItem.from_dict(make_text_item("x", nlp_context={
        "nlp_model_version": "n9"})), role=role_for(cand)).record
    assert absent["upstream"]["nlp_model_version"] is None
    assert prebuilt["upstream"]["nlp_model_version"] is None
    assert set(prebuilt["upstream"]) == {"detector_model_sha256", "detector_settings",
                                         "nlp_model_version", "report_uri"}


def test_nlp_context_is_still_validated_and_preserved_on_the_item_just_not_read_by_the_encoder():
    d = make_text_item(nlp_context={"nlp_model_version": "n9"})
    assert TextEvidenceItem.from_dict(d).to_dict()["nlp_context"] == {"nlp_model_version": "n9"}
    assert text_evidence_problems(make_text_item(nlp_context=[1]))       # shape rule unchanged


def test_record_with_null_nlp_model_version_conforms_to_the_embedding_record_schema(schemas):
    v = Draft202012Validator(schemas["embedding_record.schema.json"])
    rec = mock_text_encoder("e5").encode(
        make_text_item(nlp_context={"nlp_model_version": "n9"}), role="passage").record
    assert not list(v.iter_errors(rec))


def test_encoder_module_never_reads_nlp_context():
    tree = ast.parse((ROOT / "text" / "encoder.py").read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "nlp_context" not in attrs and "nlp_context" not in consts
    assert "nlp_model_version" in consts                                 # only as the output key


# ====================================================================== 3. torch backend
class _Recorder:
    def __init__(self):
        self.tokenizer_calls, self.model_calls = [], []
        self.during_tokenizer = None          # callable run inside AutoTokenizer.from_pretrained
        self.during_model = None              # callable run inside AutoModel.from_pretrained
        self.tokenizer_is_fast = True


def fake_stack(rec: _Recorder):
    class _Tok:
        pass

    class _Model:
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
            rec.tokenizer_calls.append((path, kw))
            if rec.during_tokenizer:
                rec.during_tokenizer(Path(path))
            t = _Tok()
            t.is_fast = rec.tokenizer_is_fast
            return t

    class AutoModel:
        @staticmethod
        def from_pretrained(path, **kw):
            rec.model_calls.append((path, kw))
            if rec.during_model:
                rec.during_model(Path(path))
            return _Model()

    torch = types.SimpleNamespace(__version__="0-fake", cuda=types.SimpleNamespace(
        is_available=lambda: False))
    transformers = types.SimpleNamespace(__version__="0-fake", AutoTokenizer=AutoTokenizer,
                                         AutoModel=AutoModel)
    return torch, transformers


@pytest.fixture
def rec_stack(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(TB, "_import_torch_stack", lambda: fake_stack(rec))
    return rec


@pytest.fixture(params=[(E5_NAME, E5_SPEC), (MINILM_NAME, MINILM_SPEC)], ids=["e5", "minilm"])
def pinned(request, tmp_path):
    name, spec = request.param
    mp = make_pinned_synthetic_manifest(tmp_path, name)
    enc = text_encoder_from_manifest(name, mp)
    return enc.config, spec, dict(enc.file_sha256), Path(enc.config.weights_dir)


def flip_mid(p: Path):
    b = bytearray(p.read_bytes())
    i = len(b) // 2
    b[i] = ord("0") if b[i] != ord("0") else ord("1")
    p.write_bytes(bytes(b))


def flip_last(p: Path):
    b = bytearray(p.read_bytes())
    b[-1] ^= 0xFF
    p.write_bytes(bytes(b))


def mutate(p: Path):
    (flip_last if p.name == "model.safetensors" else flip_mid)(p)


def test_pinned_load_succeeds_with_fakes_and_passes_the_local_only_arguments(pinned, rec_stack):
    cfg, spec, pins, d = pinned
    be = TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    assert be.info.extra["tokenizer_sha256"] == pins["tokenizer.json"]
    (tpath, tkw), = rec_stack.tokenizer_calls
    (mpath, mkw), = rec_stack.model_calls
    assert tpath == mpath == str(d)
    assert tkw == {"local_files_only": True, "trust_remote_code": False, "use_fast": True}
    assert mkw == {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True}


def test_use_fast_true_is_explicit_in_the_source():
    tree = ast.parse((ROOT / "text" / "torch_backend.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "from_pretrained" and isinstance(n.func.value, ast.Attribute)
             and n.func.value.attr == "AutoTokenizer"]
    assert len(calls) == 1
    kw = {k.arg: k.value.value for k in calls[0].keywords if isinstance(k.value, ast.Constant)}
    assert kw.get("use_fast") is True


def test_a_slow_tokenizer_result_is_refused(pinned, rec_stack):
    cfg, spec, pins, _ = pinned
    rec_stack.tokenizer_is_fast = False
    with pytest.raises(EncoderLoadError, match="fast tokenizer"):
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)


def test_every_manifest_listed_file_is_verified_before_anything_is_loaded(pinned, rec_stack):
    """Same-size content change of EACH listed file (loading files and evidence files alike) is
    refused at load time, before from_pretrained is ever called."""
    cfg, spec, pins, d = pinned
    assert set(REQUIRED_FILES) < set(pins) and "1_Pooling/config.json" in pins
    for fname in sorted(pins):
        original = (d / fname).read_bytes()
        mutate(d / fname)
        with pytest.raises(EncoderIntegrityError, match="SHA-256"):
            TB.TorchTextBackend(cfg, spec, file_sha256=pins)
        assert not rec_stack.tokenizer_calls and not rec_stack.model_calls, fname
        (d / fname).write_bytes(original)                                 # restore
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)                  # intact again -> loads
        rec_stack.tokenizer_calls.clear(), rec_stack.model_calls.clear()


def test_each_unpinned_loading_file_is_refused(pinned, rec_stack):
    cfg, spec, pins, _ = pinned
    for fname in REQUIRED_FILES:
        for broken in ({k: v for k, v in pins.items() if k != fname}, dict(pins, **{fname: None}),
                       dict(pins, **{fname: ""})):
            with pytest.raises(EncoderIntegrityError, match="no pinned SHA-256"):
                TB.TorchTextBackend(cfg, spec, file_sha256=broken)
    assert not rec_stack.tokenizer_calls


@pytest.mark.parametrize("bad", ["xyz", "0" * 63, 5], ids=["not-hex", "short", "int"])
def test_malformed_pins_and_unsafe_names_are_refused(pinned, rec_stack, bad):
    cfg, spec, pins, _ = pinned
    with pytest.raises(EncoderIntegrityError, match="not 64 hex"):
        TB.TorchTextBackend(cfg, spec, file_sha256=dict(pins, **{"modules.json": bad}))
    with pytest.raises(EncoderIntegrityError, match="unsafe pinned file name"):
        TB.TorchTextBackend(cfg, spec, file_sha256=dict(pins, **{"../escape.json": "0" * 64}))
    assert not rec_stack.tokenizer_calls


def test_missing_or_symlinked_pinned_files_are_refused(pinned, rec_stack):
    cfg, spec, pins, d = pinned
    for fname in ("tokenizer_config.json", "special_tokens_map.json", "1_Pooling/config.json"):
        p = d / fname
        data = p.read_bytes()
        p.unlink()
        with pytest.raises(EncoderLoadError, match="missing or not a regular file"):
            TB.TorchTextBackend(cfg, spec, file_sha256=pins)
        target = p.parent / "elsewhere.bin"
        target.write_bytes(data)
        os.symlink(target, p)                                            # same bytes, but a symlink
        with pytest.raises(EncoderLoadError, match="missing or not a regular file"):
            TB.TorchTextBackend(cfg, spec, file_sha256=pins)
        p.unlink()
        target.rename(p)
    assert not rec_stack.tokenizer_calls
    TB.TorchTextBackend(cfg, spec, file_sha256=pins)                     # restored -> fine


LOADING_FILES = list(REQUIRED_FILES)


@pytest.mark.parametrize("stage", ["during_tokenizer", "during_model"])
@pytest.mark.parametrize("fname", LOADING_FILES + ["1_Pooling/config.json"])
def test_a_file_changed_during_load_is_detected_and_the_load_refused(pinned, rec_stack, fname, stage):
    """Equivalent-guarantee evidence for post-verification changes: the pinned files are hashed
    again after from_pretrained; a modification made while loading (simulated by the fake) raises."""
    cfg, spec, pins, d = pinned
    setattr(rec_stack, stage, lambda path: mutate(path / fname))
    with pytest.raises(EncoderIntegrityError, match="changed during load") as ei:
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    assert fname in str(ei.value)


def test_a_file_deleted_or_replaced_by_a_symlink_during_load_is_detected(pinned, rec_stack):
    cfg, spec, pins, d = pinned
    rec_stack.during_model = lambda path: (path / "tokenizer_config.json").unlink()
    with pytest.raises(EncoderIntegrityError, match="tokenizer_config.json"):
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    # symlink swap with identical bytes
    mp = d / "special_tokens_map.json"

    def swap(path):
        data = mp.read_bytes()
        mp.unlink()
        (d / "twin.bin").write_bytes(data)
        os.symlink(d / "twin.bin", mp)
    rec_stack.during_model = swap
    (d / "tokenizer_config.json").write_text('{"tokenizer_class": "synthetic"}')   # restore same bytes
    with pytest.raises(EncoderIntegrityError, match="special_tokens_map.json"):
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)


def test_weights_config_and_tokenizer_are_rechecked_even_without_manifest_pins(pinned, rec_stack):
    cfg, spec, _, d = pinned
    rec_stack.during_model = lambda path: flip_mid(path / "config.json")
    with pytest.raises(EncoderIntegrityError, match="config.json"):
        TB.TorchTextBackend(cfg, spec)                                   # only expected_weights_sha256
