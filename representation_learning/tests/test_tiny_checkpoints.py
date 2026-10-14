"""Exercise the REAL loader/inference code (safetensors + transformers + torch) using
tiny RANDOMLY-INITIALISED local checkpoints. These are not DINOv2/CLIP weights, so
nothing here says anything about embedding quality. No network, no downloads."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from representation_learning.encoders import (ClipEncoder, Dinov2Encoder, EncoderConfig,  # noqa: E402
                                              EncoderIntegrityError, EncoderLoadError)
from representation_learning.encoders.integrity import sha256_file  # noqa: E402
from representation_learning.testing.tiny_checkpoints import (CLIP_PREPROC_JSON,  # noqa: E402
                                                             make_tiny_clip, make_tiny_dinov2)

pytestmark = pytest.mark.tiny_random


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    return make_tiny_dinov2(root / "dino"), make_tiny_clip(root / "clip")


def _cfg(d, **kw):
    return EncoderConfig(weights_dir=d, checkpoint_id="tiny-random-NOT-PRETRAINED",
                         expected_weights_sha256=sha256_file(d / "model.safetensors"),
                         device="cpu", **kw)


def _encoders(ckpts):
    return [Dinov2Encoder("small", _cfg(ckpts[0])), ClipEncoder("vit-b-32", _cfg(ckpts[1]))]


def test_real_code_path_outputs(ckpts, rgb_img):
    for enc, dim in zip(_encoders(ckpts), (384, 512)):
        r = enc.encode(rgb_img, color_order="BGR")
        assert r.vector.shape == (dim,) and r.vector.dtype == np.float32
        assert np.isfinite(r.vector).all() and abs(np.linalg.norm(r.vector) - 1) < 1e-5
        assert r.record["inference"]["device"] == "cpu"
        assert r.record["encoder"]["version_or_checkpoint"].startswith("tiny-random")
        assert r.record["warnings"] == []
        assert len(r.record["encoder"]["config_sha256"]) == 64


def test_schema_valid_and_space_ids(ckpts, rgb_img, schemas):
    from jsonschema import Draft202012Validator
    v = Draft202012Validator(schemas["embedding_record.schema.json"])
    spaces = set()
    for enc in _encoders(ckpts):
        rec = enc.encode(rgb_img, color_order="RGB").record
        assert [e.message for e in v.iter_errors(rec)] == []
        spaces.add(rec["space_id"])
    assert len(spaces) == 2


def test_repeatability_and_batch_consistency(ckpts, rgb_img):
    for enc in _encoders(ckpts):
        imgs = [np.roll(rgb_img, k * 11, axis=0) for k in range(4)]
        a = enc.encode(imgs[0], color_order="RGB").vector
        b = enc.encode(imgs[0].copy(), color_order="RGB").vector
        assert np.array_equal(a, b)
        batch = enc.encode_batch(imgs, color_order="RGB")
        singles = [enc.encode(i, color_order="RGB").vector for i in imgs]
        for s, bt in zip(singles, batch):
            assert np.allclose(s, bt.vector, atol=1e-5)


def test_weights_hash_mismatch_and_missing_hash(ckpts):
    d = ckpts[0]
    with pytest.raises(EncoderIntegrityError, match="mismatch"):
        Dinov2Encoder("small", EncoderConfig(weights_dir=d, expected_weights_sha256="0" * 64,
                                              device="cpu")).load()
    with pytest.raises(EncoderIntegrityError, match="required"):
        Dinov2Encoder("small", EncoderConfig(weights_dir=d, device="cpu")).load()
    info = Dinov2Encoder("small", EncoderConfig(weights_dir=d, device="cpu",
                                                allow_unverified=True)).load()
    assert any("not_verified" in w for w in info.warnings)


def test_dimension_variant_mismatch(ckpts):
    with pytest.raises(EncoderIntegrityError, match="dim"):
        Dinov2Encoder("base", _cfg(ckpts[0])).load()       # optional variant 'base' (768) vs a 384-d dir
    with pytest.raises(EncoderIntegrityError, match="model_type"):
        ClipEncoder("vit-b-32", _cfg(ckpts[0])).load()     # DINOv2 dir given to CLIP


def test_preprocessing_mismatch_in_checkpoint(tmp_path):
    d = make_tiny_clip(tmp_path / "c", preproc=dict(CLIP_PREPROC_JSON, image_mean=[0.5, 0.5, 0.5]))
    with pytest.raises(EncoderIntegrityError, match="preprocess"):
        ClipEncoder("vit-b-32", _cfg(d)).load()


def test_missing_preprocessor_config_warns(tmp_path):
    d = make_tiny_clip(tmp_path / "c")
    (d / "preprocessor_config.json").unlink()
    info = ClipEncoder("vit-b-32", _cfg(d)).load()
    assert any("preprocessor_config.json_not_found" in w for w in info.warnings)


def test_corrupt_weights_raise_load_error(tmp_path):
    d = make_tiny_dinov2(tmp_path / "d")
    (d / "model.safetensors").write_bytes(b"not a safetensors file")
    with pytest.raises(EncoderLoadError, match="failed to load"):
        Dinov2Encoder("small", _cfg(d)).load()


def test_unreadable_config(tmp_path):
    d = make_tiny_dinov2(tmp_path / "d")
    cfg = _cfg(d)
    (d / "config.json").write_text("{broken")
    with pytest.raises(EncoderLoadError, match="config.json"):
        Dinov2Encoder("small", cfg).load()


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
def test_cuda_requested_without_cuda_fails_clearly(ckpts):
    cfg = EncoderConfig(weights_dir=ckpts[0], allow_unverified=True, device="cuda")
    with pytest.raises(EncoderLoadError, match="CUDA"):
        Dinov2Encoder("small", cfg).load()


def test_auto_device_resolves(ckpts):
    cfg = EncoderConfig(weights_dir=ckpts[0], allow_unverified=True, device="auto")
    assert Dinov2Encoder("small", cfg).load().device in ("cpu", "cuda")


def test_bad_device_string(ckpts):
    cfg = EncoderConfig(weights_dir=ckpts[0], allow_unverified=True, device="tpu")
    with pytest.raises(EncoderLoadError, match="unsupported device"):
        Dinov2Encoder("small", cfg).load()


def test_non_float32_precision_rejected(ckpts):
    cfg = EncoderConfig(weights_dir=ckpts[0], allow_unverified=True, device="cpu", precision="float16")
    with pytest.raises(EncoderLoadError, match="float32"):
        Dinov2Encoder("small", cfg).load()
