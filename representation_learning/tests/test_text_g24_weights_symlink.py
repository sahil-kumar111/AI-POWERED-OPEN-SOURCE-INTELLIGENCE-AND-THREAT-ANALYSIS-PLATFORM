"""Phase 3 / G2.4: a symlinked `model.safetensors` is refused before anything is loaded. All offline.

Policy under test (see text/torch_backend.py and docs/phase3_text_encoder.md, section 5, G2.4):
  * the weight file at `<weights_dir>/model.safetensors` must be a regular file, not a symlink
    (including a dangling one), whether or not manifest hashes are supplied;
  * the check runs before the model stack is imported or any model is loaded.

IMPORTANT: these tests exercise OUR check only, with the offline fake `torch`/`transformers`
stand-ins from test_text_g22_sidecars.py. They say nothing about how the real Hugging Face loader
treats symlinks, and nothing here loads a real tokenizer, model or weights."""

import os

import pytest

from representation_learning.text import torch_backend as TB
from representation_learning.text.adapters import E5_NAME, E5_SPEC, MINILM_NAME, MINILM_SPEC
from representation_learning.text.errors import EncoderIntegrityError, EncoderLoadError

from test_text_g22_sidecars import build, no_stack, rec  # noqa: F401  (fixtures and helpers)

CANDS = [(E5_NAME, E5_SPEC), (MINILM_NAME, MINILM_SPEC)]
CAND_IDS = ["e5", "minilm"]
W = "model.safetensors"


def symlink_weights(d, tmp_path, *, dangling=False):
    """Replace the regular weight file by a symlink. Non-dangling: identical bytes elsewhere (so a
    hash comparison alone would pass). Dangling: points at a path that does not exist."""
    real = d / W
    if dangling:
        real.unlink()
        os.symlink(tmp_path / "does-not-exist.safetensors", real)
        return
    twin = tmp_path / "elsewhere.dat"
    twin.write_bytes(real.read_bytes())
    real.unlink()
    os.symlink(twin, real)


@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
def test_symlinked_weights_refused_before_loading_with_pins(tmp_path, no_stack, name, spec):
    cfg, pins, d = build(tmp_path, name)
    symlink_weights(d, tmp_path)
    with pytest.raises((EncoderLoadError, EncoderIntegrityError), match="is a symlink") as ei:
        TB.TorchTextBackend(cfg, spec, file_sha256=pins)
    assert W in str(ei.value)


@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
def test_symlinked_weights_refused_before_loading_without_manifest_hashes(tmp_path, no_stack, name, spec):
    """No `file_sha256`: the only pin is `expected_weights_sha256`, which the identical-bytes target
    satisfies. Before G2.4 this directory was accepted."""
    cfg, _, d = build(tmp_path, name)
    symlink_weights(d, tmp_path)
    assert cfg.expected_weights_sha256 is not None        # the hash of the target would match
    with pytest.raises((EncoderLoadError, EncoderIntegrityError), match="is a symlink") as ei:
        TB.TorchTextBackend(cfg, spec)
    assert W in str(ei.value)


def test_symlinked_weights_refused_with_allow_unverified(tmp_path, no_stack):
    """Even with no expected hash at all (allow_unverified), a symlink is not accepted."""
    import dataclasses
    cfg, _, d = build(tmp_path, E5_NAME)
    symlink_weights(d, tmp_path)
    cfg = dataclasses.replace(cfg, expected_weights_sha256=None, allow_unverified=True)
    with pytest.raises((EncoderLoadError, EncoderIntegrityError), match="is a symlink"):
        TB.TorchTextBackend(cfg, E5_SPEC)


@pytest.mark.parametrize("pinned", [True, False], ids=["with-pins", "without-pins"])
@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
def test_dangling_symlink_is_refused_explicitly_before_loading(tmp_path, no_stack, name, spec, pinned):
    cfg, pins, d = build(tmp_path, name)
    symlink_weights(d, tmp_path, dangling=True)
    assert os.path.islink(d / W) and not (d / W).exists()
    with pytest.raises((EncoderLoadError, EncoderIntegrityError), match="is a symlink") as ei:
        TB.TorchTextBackend(cfg, spec, file_sha256=pins if pinned else None)
    assert W in str(ei.value)                             # an explicit refusal, not "weights not found"


def test_symlink_swapped_in_during_load_is_still_detected(tmp_path, rec):
    """Existing post-load hashing (unchanged in G2.4) also covers the weight file."""
    cfg, pins, _ = build(tmp_path, E5_NAME)
    rec.during_model = lambda path: symlink_weights(path, tmp_path)
    with pytest.raises(EncoderIntegrityError, match="changed during load") as ei:
        TB.TorchTextBackend(cfg, E5_SPEC, file_sha256=pins)
    assert W in str(ei.value)


@pytest.mark.parametrize("pinned", [True, False], ids=["with-pins", "without-pins"])
@pytest.mark.parametrize("name,spec", CANDS, ids=CAND_IDS)
def test_regular_weights_still_load(tmp_path, rec, name, spec, pinned):
    cfg, pins, d = build(tmp_path, name)
    assert not os.path.islink(d / W) and (d / W).is_file()
    TB.TorchTextBackend(cfg, spec, file_sha256=pins if pinned else None)
    assert rec.loads == 2


def test_symlinked_weights_directory_itself_is_not_part_of_this_check(tmp_path, rec):
    """Scope note: only the weight FILE is checked; a symlinked weights directory (a deployment choice)
    is unchanged G2.3 behaviour and still loads."""
    cfg, pins, d = build(tmp_path, E5_NAME)
    link = tmp_path / "linked_dir"
    os.symlink(d, link)
    import dataclasses
    TB.TorchTextBackend(dataclasses.replace(cfg, weights_dir=link), E5_SPEC, file_sha256=pins)
    assert rec.loads == 2


def test_documentation_describes_the_check():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "phase3_text_encoder.md").read_text()
    for token in ("G2.4", "_refuse_symlinked_weights", "dangling", "is a symlink"):
        assert token in doc
