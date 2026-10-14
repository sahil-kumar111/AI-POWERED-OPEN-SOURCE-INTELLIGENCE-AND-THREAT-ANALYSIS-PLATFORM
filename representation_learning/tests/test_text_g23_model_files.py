"""Phase 3 / G2.3: single-file checkpoint contract of `TorchTextBackend`. All offline.

Policy under test (see text/torch_backend.py and docs/phase3_text_encoder.md, section 5 (G2.3)):
  * the model directory may hold exactly one weight file, the pinned `model.safetensors`;
  * adapter files (`adapter_config*`, `adapter_model*`), weight index files (`*.index.json`) and every
    other weight-format file (other `.safetensors`, `.bin`, `.pt`, ...) are refused BEFORE the model
    stack is imported, and again after loading (a file may appear while loading).

IMPORTANT: the filename list was derived by READING the transformers 5.18.0 source. These tests
exercise OUR checks only. The fake `torch`/`transformers` stand-ins record calls and may touch files;
they do NOT establish how the real Hugging Face loader behaves, and nothing here loads a real
tokenizer, model or weights."""

import os
from pathlib import Path

import pytest

from representation_learning.text import torch_backend as TB
from representation_learning.text.adapters import E5_NAME, E5_SPEC, MINILM_NAME, MINILM_SPEC
from representation_learning.text.errors import EncoderIntegrityError, EncoderLoadError

from test_text_g22_sidecars import Rec, build, drop_in, fake_stack, no_stack, rec  # noqa: F401  (fixtures)

CANDS = [(E5_NAME, E5_SPEC), (MINILM_NAME, MINILM_SPEC)]
CAND_IDS = ["e5", "minilm"]

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
                 "adapter_model.fp16.safetensors", "adapter_model-00001-of-00002.safetensors",
                 "ADAPTER_CONFIG.JSON"]
STRAY_WEIGHTS = ["pytorch_model.bin", "model.fp16.safetensors", "model-00001-of-00002.safetensors",
                 "extra.safetensors", "Model.safetensors", "MODEL.SAFETENSORS", "Weights.BIN", "pytorch_model.fp16.bin", "weights.pt",
                 "model.pth", "model.ckpt", "tf_model.h5", "flax_model.msgpack", "model.onnx",
                 "model.gguf", "state.pkl", "state.pickle"]
# `model.safetensors.index.json` itself is refused EARLIER by the approved Phase 2 `locate_weights`
# (EncoderLoadError, "sharded checkpoints"); it is tested separately below.
INDEX_FILES = ["pytorch_model.bin.index.json", "model.fp16.safetensors.index.json", "anything.index.json"]
ALL_REFUSED = ADAPTER_FILES + STRAY_WEIGHTS + INDEX_FILES


@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
@pytest.mark.parametrize("rel", ALL_REFUSED)
def test_unlisted_model_side_file_is_refused_before_loading(tmp_path, no_stack, name, spec, rel):
    cfg, pins, d = build(tmp_path, name)
    drop_in(d, rel, b"x")
    with pytest.raises(EncoderIntegrityError, match="unpinned model-side file") as ei:
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    assert rel in str(ei.value)


@pytest.mark.parametrize("rel", ["adapter_config.json", "pytorch_model.bin", "pytorch_model.bin.index.json"])
def test_refused_without_manifest_pins_too(tmp_path, no_stack, rel):
    cfg, _, d = build(tmp_path, E5_NAME)
    drop_in(d, rel, b"x")
    with pytest.raises(EncoderIntegrityError, match="unpinned model-side file"):
        TB.TorchTextBackend(cfg, E5_SPEC)


@pytest.mark.parametrize("rel", ["adapter_config.json", "pytorch_model.bin", "pytorch_model.bin.index.json"])
def test_listing_and_pinning_does_not_make_them_acceptable(tmp_path, no_stack, rel):
    """The G1 manifest already refuses stray weight files; here we also check that the backend does
    not accept such a file merely because a pin for it exists."""
    cfg, pins, d = build(tmp_path, E5_NAME)
    p = drop_in(d, rel, b"x")
    import hashlib
    pins = dict(pins, **{rel: hashlib.sha256(p.read_bytes()).hexdigest()})
    with pytest.raises(EncoderIntegrityError, match="unpinned model-side file"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


def test_symlinked_and_directory_entries_count(tmp_path, no_stack):
    cfg, pins, d = build(tmp_path, E5_NAME)
    target = tmp_path / "elsewhere.txt"
    target.write_bytes(b"x")
    os.symlink(target, d / "adapter_model.safetensors")                 # symlink with a harmless target name
    with pytest.raises(EncoderIntegrityError, match="adapter_model.safetensors"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)
    os.unlink(d / "adapter_model.safetensors")
    (d / "adapter_config.json").mkdir()                                 # a directory of that name
    with pytest.raises(EncoderIntegrityError, match="adapter_config.json"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


def test_all_problems_are_listed_together(tmp_path, no_stack):
    cfg, pins, d = build(tmp_path, E5_NAME)
    for rel in ("adapter_config.json", "pytorch_model.bin", "pytorch_model.bin.index.json"):
        drop_in(d, rel, b"x")
    with pytest.raises(EncoderIntegrityError) as ei:
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)
    for rel in ("adapter_config.json", "pytorch_model.bin", "pytorch_model.bin.index.json"):
        assert rel in str(ei.value)


@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
def test_model_safetensors_index_json_is_refused_before_loading(tmp_path, no_stack, name, spec):
    """Refused by the approved Phase 2 `locate_weights` (G2.2 behaviour, unchanged) before our check."""
    cfg, pins, d = build(tmp_path, name)
    drop_in(d, "model.safetensors.index.json", b"{}")
    with pytest.raises(EncoderLoadError, match="sharded checkpoints"):
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)


def test_dangling_symlink_index_is_caught_by_our_check(tmp_path, no_stack):
    """`locate_weights` uses `.exists()`, which is False for a dangling symlink; ours lists the directory."""
    cfg, pins, d = build(tmp_path, E5_NAME)
    os.symlink(tmp_path / "does-not-exist", d / "model.safetensors.index.json")
    with pytest.raises(EncoderIntegrityError, match="model.safetensors.index.json"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


# ------------------------------------------------------------------ appearing during load
@pytest.mark.parametrize("rel", ["adapter_config.json", "adapter_model.safetensors", "pytorch_model.bin",
                                 "extra.safetensors", "model.safetensors.index.json",
                                 "pytorch_model.bin.index.json"])
def test_model_side_file_appearing_during_load_is_detected(tmp_path, rec, rel):
    cfg, pins, _ = build(tmp_path, E5_NAME)
    rec.during_model = lambda path: drop_in(path, rel, b"x")
    with pytest.raises(EncoderIntegrityError, match="model-side file state changed during load") as ei:
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)
    assert rel in str(ei.value)
    assert rec.loads == 2                                               # the load did run; the post-check caught it


# ------------------------------------------------------------------ still accepted
@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
def test_legitimate_pinned_single_file_checkpoint_still_loads(tmp_path, rec, name, spec):
    cfg, pins, d = build(tmp_path, name)
    assert "model.safetensors" in pins
    TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    assert rec.loads == 2


@pytest.mark.parametrize("rel", ["README.md", "sentencepiece.bpe.model", "vocab.txt",
                                 "config_sentence_transformers.json", "generation_config.json",
                                 "adapter.txt", "my_adapter_config.json", "notes.safetensors.txt"])
def test_other_unlisted_files_are_not_refused_by_this_check(tmp_path, rec, rel):
    cfg, pins, d = build(tmp_path, E5_NAME)
    drop_in(d, rel, b"x")
    TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


def test_files_in_subdirectories_are_not_scanned_documented_limit(tmp_path, rec):
    """DOCUMENTED LIMIT: the loader is never given `subfolder`, so only the top level is checked."""
    cfg, pins, d = build(tmp_path, E5_NAME)
    drop_in(d, "onnx/model.onnx", b"x")
    drop_in(d, "openvino/adapter_config.json", b"x")
    TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


def test_tokenizer_sidecars_still_accepted_when_pinned(tmp_path, rec):
    cfg, pins, _ = build(tmp_path, E5_NAME, extra={"added_tokens.json": b'{"<x>": 5}',
                                                   "chat_template.jinja": b"{{ m }}",
                                                   "additional_chat_templates/a.jinja": b"{{ a }}"})
    TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


# ------------------------------------------------------------------ order and documented constants
def test_check_runs_after_pin_validation_so_pin_errors_keep_their_messages(tmp_path, no_stack):
    cfg, pins, d = build(tmp_path, E5_NAME)
    drop_in(d, "adapter_config.json", b"x")
    data = bytearray((d / "config.json").read_bytes())
    data[-2] ^= 0x01
    (d / "config.json").write_bytes(bytes(data))
    with pytest.raises(EncoderIntegrityError, match="SHA-256"):
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)


def test_constants_match_the_documented_policy():
    assert TB.WEIGHTS == "model.safetensors"
    assert TB.INDEX_SUFFIX == ".index.json"
    assert TB.ADAPTER_PREFIXES == ("adapter_config", "adapter_model")
    for suffix in (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".h5", ".onnx", ".gguf"):
        assert suffix in TB.WEIGHT_FORMAT_SUFFIXES
    doc = (Path(__file__).resolve().parents[1] / "docs" / "phase3_text_encoder.md").read_text()
    for token in ("adapter_config.json", "adapter_model.safetensors", "pytorch_model.bin",
                  "model.safetensors.index.json", "*.index.json", "single-file"):
        assert token in doc
