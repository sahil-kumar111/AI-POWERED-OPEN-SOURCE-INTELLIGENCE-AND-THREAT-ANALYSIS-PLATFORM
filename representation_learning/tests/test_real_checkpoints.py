"""REAL pretrained checkpoint tests (status: PENDING until weights are pinned locally).

They use `representation_learning/models.manifest.json`. They SKIP while the manifest
entries are unpinned or the local files are missing - a skip is NOT a pass. They never
download anything. Run exactly as documented in docs/weights_and_models.md.
"""

import os

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from representation_learning.encoders import manifest as M
from representation_learning.encoders.errors import EncoderError

pytestmark = pytest.mark.real_checkpoint
pytest.importorskip("torch")
pytest.importorskip("transformers")

MANIFEST = os.environ.get("RL_MANIFEST", str(M.DEFAULT_MANIFEST))
EXPECTED = {"dinov2-small": (384, "dinov2-small"), "clip-vit-b-32": (512, "clip-vit-b-32")}


@pytest.fixture(params=sorted(EXPECTED))
def enc(request):
    name = request.param
    m = M.load_manifest(MANIFEST)
    probs = M.pin_problems(m["models"][name])
    if probs:
        pytest.skip(f"{name}: manifest not pinned ({'; '.join(probs)})")
    try:
        return M.encoder_from_manifest(name, MANIFEST, device="cpu")
    except EncoderError as e:
        pytest.skip(f"{name}: local weights unavailable/unverified: {e}")


def test_real_load_dimension_and_hash(enc):
    info = enc.load()
    assert info.dim == enc.spec.expected_dim
    assert info.device == "cpu" and info.warnings == ()   # hash verified, preprocessing cross-checked


def test_real_inference_valid_vectors(enc):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    a = enc.encode(img, color_order="RGB")
    b = enc.encode(img.copy(), color_order="RGB")
    assert a.vector.shape == (enc.spec.expected_dim,) and a.vector.dtype == np.float32
    assert np.isfinite(a.vector).all() and abs(np.linalg.norm(a.vector) - 1) < 1e-5
    assert np.allclose(a.vector, b.vector, atol=1e-6)
    assert a.record["warnings"] == []


def test_real_record_matches_phase1_schema_and_pins(enc, schemas):
    img = np.random.default_rng(1).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    rec = enc.encode(img, color_order="RGB").record
    assert [e.message for e in Draft202012Validator(
        schemas["embedding_record.schema.json"]).iter_errors(rec)] == []
    m = M.load_manifest(MANIFEST)["models"]
    key = enc.spec.name
    entry = m[key]
    assert rec["encoder"]["version_or_checkpoint"] == f"{entry['hf_id']}@{entry['revision']}"
    assert rec["encoder"]["weights_sha256"] == entry["files"]["model.safetensors"]


def test_real_batch_vs_single_agreement(enc):
    rng = np.random.default_rng(2)
    imgs = [rng.integers(0, 256, (150, 210, 3), dtype=np.uint8) for _ in range(3)]
    singles = [enc.encode(i, color_order="RGB").vector for i in imgs]
    batch = enc.encode_batch(imgs, color_order="RGB")
    for s, b in zip(singles, batch):
        assert np.allclose(s, b.vector, atol=1e-4)
