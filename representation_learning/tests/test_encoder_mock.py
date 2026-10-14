"""Unit tests with MOCK backends only: no weights, no torch, no network.
They verify the interface, validation, metadata and error handling - NOT the
quality of DINOv2/CLIP embeddings."""

import json
import subprocess
import sys

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from representation_learning.encoders import (ClipEncoder, Dinov2Encoder, EmbeddingValidationError,
                                              EncodeFailure, EncoderConfig, EncoderIntegrityError,
                                              EncoderLoadError, InvalidInputError,
                                              UnsupportedEvidenceError)
from representation_learning.testing.mocks import MockBackend, mock_encoder

FAMILIES = [("dinov2", 384), ("clip", 512)]


@pytest.fixture(params=FAMILIES, ids=["dinov2", "clip"])
def enc_dim(request):
    fam, dim = request.param
    return mock_encoder(fam), dim


def test_valid_input_shape_dtype_norm_finite(enc_dim, rgb_img):
    enc, dim = enc_dim
    r = enc.encode(rgb_img, color_order="BGR")
    assert r.vector.shape == (dim,) and r.vector.dtype == np.float32
    assert np.isfinite(r.vector).all()
    assert abs(float(np.linalg.norm(r.vector.astype(np.float64))) - 1.0) < 1e-5
    assert not r.vector.flags.writeable
    assert r.dim == dim


def test_pil_grayscale_rgba_inputs(enc_dim, pil_img):
    enc, dim = enc_dim
    for im in (pil_img.convert("L"), pil_img.convert("RGBA"), pil_img):
        assert enc.encode(im).vector.shape == (dim,)
    assert enc.encode(np.full((50, 60), 9, np.uint8)).vector.shape == (dim,)
    assert enc.encode(np.zeros((50, 60, 4), np.uint8), color_order="RGB").vector.shape == (dim,)


@pytest.mark.parametrize("bad", [None, np.zeros((0, 4, 3), np.uint8), "x.jpg",
                                 np.zeros((4, 4, 3), np.float32)])
def test_invalid_image_rejected(enc_dim, bad):
    with pytest.raises(InvalidInputError):
        enc_dim[0].encode(bad, color_order="RGB")


def test_determinism_repeated_and_copy(enc_dim, rgb_img):
    enc, _ = enc_dim
    a = enc.encode(rgb_img, color_order="RGB").vector
    b = enc.encode(rgb_img.copy(), color_order="RGB").vector
    assert np.array_equal(a, b)


def test_different_images_differ(enc_dim, rgb_img):
    enc, _ = enc_dim
    other = 255 - rgb_img
    assert not np.array_equal(enc.encode(rgb_img, color_order="RGB").vector,
                              enc.encode(other, color_order="RGB").vector)


def test_batch_matches_single_and_chunks(rgb_img):
    enc = mock_encoder("clip", max_batch_size=2)
    imgs = [np.roll(rgb_img, k * 7, axis=1) for k in range(5)]
    singles = [enc.encode(i, color_order="RGB").vector for i in imgs]
    batch = enc.encode_batch(imgs, color_order="RGB")
    assert enc._backend.batch_sizes[-3:] == [2, 2, 1]
    for s, b in zip(singles, batch):
        assert np.allclose(s, b.vector, atol=1e-6)
    assert [b.record["inference"]["batch_size"] for b in batch] == [2, 2, 2, 2, 1]


def test_batch_collects_failures_and_raise_mode(rgb_img):
    enc = mock_encoder("dinov2")
    out = enc.encode_batch([rgb_img, None, rgb_img], color_order="RGB")
    assert isinstance(out[1], EncodeFailure) and out[1].error_type == "InvalidInputError"
    assert out[0].dim == 384 and out[2].dim == 384
    with pytest.raises(InvalidInputError):
        enc.encode_batch([rgb_img, None], color_order="RGB", on_error="raise")


# ------------------------------------------------------------------ metadata
def test_record_validates_against_phase1_schema(enc_dim, rgb_img, schemas):
    enc, _ = enc_dim
    rec = enc.encode(rgb_img, color_order="RGB").record
    v = Draft202012Validator(schemas["embedding_record.schema.json"])
    assert [e.message for e in v.iter_errors(rec)] == []
    json.dumps(rec)  # JSON-serialisable


def test_record_metadata_content(rgb_img):
    enc = mock_encoder("dinov2", seed=3)
    r = enc.encode(rgb_img, color_order="RGB")
    rec = r.record
    assert rec["modality"] == "visual" and rec["dtype"] == "float32"
    assert rec["metric"] == "cosine" and rec["normalized"] is True
    assert rec["encoder"]["name"] == "dinov2-small" and rec["encoder"]["family"] == "DINOv2"
    assert "CLS" in rec["encoder"]["feature"]
    assert len(rec["encoder"]["weights_sha256"]) == 64
    assert rec["preprocessing"]["id"].startswith("dinov2")
    assert rec["quality"]["input_width"] == 160 and rec["quality"]["input_height"] == 120
    assert rec["source_ref"]["kind"] == "adhoc_image"
    assert rec["sensitivity"]["retain_pixels"] is False
    assert "mock_backend_not_a_real_model" in rec["warnings"]
    assert rec["space_id"] == enc.space_id and rec["vector_ref"].startswith("memory://")


def test_space_id_depends_on_encoder_and_weights(rgb_img):
    a = mock_encoder("dinov2", seed=0).space_id
    assert a == mock_encoder("dinov2", seed=0).space_id
    assert a != mock_encoder("dinov2", seed=1).space_id      # different weights hash
    assert a != mock_encoder("clip", seed=0).space_id         # different encoder


def test_clip_and_dino_spaces_are_distinct_and_not_comparable(rgb_img):
    d, c = mock_encoder("dinov2"), mock_encoder("clip")
    rd, rc = d.encode(rgb_img, color_order="RGB"), c.encode(rgb_img, color_order="RGB")
    assert rd.space_id != rc.space_id and rd.dim != rc.dim
    assert not hasattr(d, "compare") and not hasattr(d, "similarity")


def test_evidence_frame_and_crop_provenance(rgb_img, evidence_factory, schemas):
    enc = mock_encoder("clip")
    det = {"category": "weapon", "class_name": "rifle", "confidence": 0.58, "track_id": None,
           "detector_name": "WeaponDetector", "model_sha256": "cd" * 32,
           "settings": {"imgsz": 320, "conf": 0.3, "iou": 0.5}}
    ev = evidence_factory(source_kind="crop", bbox_xyxy=[10, 20, 90, 100], detector=det)
    assert [e.message for e in Draft202012Validator(
        schemas["visual_evidence_item.schema.json"]).iter_errors(ev)] == []
    r = enc.encode(rgb_img, color_order="RGB", evidence=ev)
    rec = r.record
    assert rec["evidence_id"] == "ve_test000001"
    assert rec["source_ref"]["crop_box_xyxy_applied"] == [10, 20, 90, 100]
    assert rec["quality"]["crop_area_px"] == 80 * 80 and rec["quality"]["detector_confidence"] == 0.58
    assert rec["upstream"]["detector_model_sha256"] == "cd" * 32
    assert [e.message for e in Draft202012Validator(
        schemas["embedding_record.schema.json"]).iter_errors(rec)] == []


def test_input_is_crop_skips_bbox(rgb_img, evidence_factory):
    enc = mock_encoder("dinov2")
    ev = evidence_factory(source_kind="crop", bbox_xyxy=[0, 0, 10, 10])
    r = enc.encode(rgb_img[:50, :60], color_order="RGB", evidence=ev, input_is_crop=True)
    assert r.record["quality"]["input_width"] == 60


def test_frame_resolution_mismatch_rejected(rgb_img, evidence_factory):
    ev = evidence_factory(source_kind="crop", bbox_xyxy=[0, 0, 40, 40])
    with pytest.raises(InvalidInputError, match="resolution"):
        mock_encoder("dinov2").encode(rgb_img[:50], color_order="RGB", evidence=ev)


def test_person_evidence_rejected(rgb_img, evidence_factory):
    det = {"category": "person", "class_name": "person", "confidence": 0.9,
           "detector_name": "PersonDetector", "settings": {"imgsz": 640, "conf": 0.3, "iou": 0.5}}
    ev = evidence_factory(source_kind="crop", bbox_xyxy=[0, 0, 50, 50], detector=det)
    with pytest.raises(UnsupportedEvidenceError):
        mock_encoder("dinov2").encode(rgb_img, color_order="RGB", evidence=ev)


def test_annotated_video_rejected(rgb_img, evidence_factory):
    ev = evidence_factory(video_uri="output/clip_analyzed.mp4")
    with pytest.raises(UnsupportedEvidenceError, match="annotated"):
        mock_encoder("clip").encode(rgb_img, color_order="RGB", evidence=ev)


def test_small_crop_warns(rgb_img):
    r = mock_encoder("dinov2").encode(rgb_img[:10, :10], color_order="RGB")
    assert any(w.startswith("small_input_min_side_px") for w in r.record["warnings"])


# ------------------------------------------------------- model-output failures
@pytest.mark.parametrize("fault", ["nan", "inf", "zeros", "wrong_shape", "raise", "int_dtype"])
def test_bad_model_output_rejected(rgb_img, fault):
    enc = mock_encoder("dinov2", fault=fault)
    with pytest.raises(EmbeddingValidationError):
        enc.encode(rgb_img, color_order="RGB")


def test_bad_model_output_batch_collected(rgb_img):
    out = mock_encoder("clip", fault="nan").encode_batch([rgb_img, rgb_img], color_order="RGB")
    assert all(isinstance(o, EncodeFailure) and o.error_type == "EmbeddingValidationError" for o in out)


def test_backend_dim_mismatch_rejected():
    with pytest.raises(EncoderIntegrityError, match="dim"):
        Dinov2Encoder("small", backend=MockBackend(768))


# ------------------------------------------------------------ loading / lazy
def test_lazy_loading_and_no_heavy_imports():
    code = ("import sys; import representation_learning.encoders as e;"
            "enc = e.Dinov2Encoder(); "
            "assert not enc.loaded; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(__import__('pathlib').Path(__file__).resolve().parents[2]))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


@pytest.mark.parametrize("cls", [Dinov2Encoder, ClipEncoder])
def test_load_failure_unconfigured_and_missing(cls, tmp_path, rgb_img):
    with pytest.raises(EncoderLoadError, match="not configured"):
        cls().encode(rgb_img, color_order="RGB")
    enc = cls(config=EncoderConfig(weights_dir=tmp_path / "missing"))
    with pytest.raises(EncoderLoadError, match="not found"):
        enc.load()
    assert not enc.loaded


@pytest.mark.parametrize("cls", [Dinov2Encoder, ClipEncoder])
def test_config_errors_do_not_touch_torch_or_transformers(cls, tmp_path, rgb_img, monkeypatch):
    """In-process guard: if the optional stack were imported on these paths the test fails."""
    from representation_learning.encoders import torch_backends

    def _boom():
        raise AssertionError("optional torch/transformers import attempted on a config-error path")
    monkeypatch.setattr(torch_backends, "_import_torch_stack", _boom)
    for config, msg in [(EncoderConfig(), "not configured"),
                        (EncoderConfig(weights_dir=tmp_path / "missing"), "not found"),
                        (EncoderConfig(weights_dir=tmp_path, precision="float16"), "float32")]:
        with pytest.raises(EncoderLoadError, match=msg):
            cls(config=config).load()


_BLOCKED_STACK = """
import sys, json
class _Block:
    BLOCK = ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub")
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCK:
            raise ImportError(f"{name} is blocked for this test")
sys.meta_path.insert(0, _Block())
import numpy as np
from pathlib import Path
import tempfile
from representation_learning.encoders import (Dinov2Encoder, ClipEncoder, EncoderConfig,
                                              EncoderLoadError, EncoderIntegrityError)
out = {}
img = np.zeros((64, 64, 3), np.uint8)
for cls in (Dinov2Encoder, ClipEncoder):
    n = cls.__name__
    try: cls().encode(img, color_order="RGB")
    except EncoderLoadError as e: out[n + ":unconfigured"] = str(e)
    try: cls(config=EncoderConfig(weights_dir=Path("/nonexistent_dir_xyz"))).load()
    except EncoderLoadError as e: out[n + ":missing"] = str(e)
    d = Path(tempfile.mkdtemp())
    (d / "model.safetensors").write_bytes(b"x"); (d / "config.json").write_text("{}")
    try: cls(config=EncoderConfig(weights_dir=d)).load()
    except EncoderIntegrityError as e: out[n + ":no_hash"] = str(e)
    try: cls(config=EncoderConfig(weights_dir=d, allow_unverified=True)).load()
    except EncoderIntegrityError as e: out[n + ":wrong_model_type"] = str(e)
    # fully valid-looking checkpoint metadata, but no optional stack installed:
    ok = Path(tempfile.mkdtemp())
    (ok / "model.safetensors").write_bytes(b"x")
    mt, dim = ("dinov2", 384) if cls is Dinov2Encoder else ("clip", 512)
    (ok / "config.json").write_text(json.dumps({"model_type": mt, "hidden_size": dim, "projection_dim": dim}))
    try: cls(config=EncoderConfig(weights_dir=ok, allow_unverified=True)).load()
    except EncoderLoadError as e: out[n + ":needs_stack"] = str(e)
assert "torch" not in sys.modules and "transformers" not in sys.modules
print(json.dumps(out))
"""


def test_unconfigured_encoder_needs_no_torch_or_transformers_even_if_blocked():
    """Subprocess with torch/transformers imports BLOCKED (also true when they are installed):
    config/integrity errors surface first; the optional stack is demanded only after them."""
    import pathlib
    root = str(pathlib.Path(__file__).resolve().parents[2])
    r = subprocess.run([sys.executable, "-c", _BLOCKED_STACK], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for n in ("Dinov2Encoder", "ClipEncoder"):
        assert "not configured" in out[n + ":unconfigured"]
        assert "not found" in out[n + ":missing"]
        assert "required" in out[n + ":no_hash"]                      # expected_weights_sha256 required
        assert "model_type" in out[n + ":wrong_model_type"]
        assert "torch and transformers are required" in out[n + ":needs_stack"]


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        Dinov2Encoder("huge")
    with pytest.raises(ValueError):
        ClipEncoder("rn50")


def test_unit_tests_do_not_import_torch_via_mocks():
    # mocks must work in an environment without torch (guards against accidental hard deps)
    code = ("import sys, numpy as np;"
            "from representation_learning.testing.mocks import mock_encoder;"
            "r = mock_encoder('clip').encode(np.zeros((64,64,3),np.uint8), color_order='RGB');"
            "assert r.dim == 512 and 'torch' not in sys.modules; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(__import__('pathlib').Path(__file__).resolve().parents[2]))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_network_is_blocked_in_tests():
    import socket
    with pytest.raises(AssertionError, match="network"):
        socket.getaddrinfo("example.com", 80)


# ----------------------------------------------------- input-scope / privacy policy
def test_whole_frame_evidence_refused_by_default(rgb_img, evidence_factory):
    enc = mock_encoder("clip")
    with pytest.raises(UnsupportedEvidenceError, match="whole-frame"):
        enc.encode(rgb_img, color_order="RGB", evidence=evidence_factory())
    out = enc.encode_batch([rgb_img], color_order="RGB", evidences=[evidence_factory()])
    assert isinstance(out[0], EncodeFailure) and out[0].error_type == "UnsupportedEvidenceError"


def test_whole_frame_allowed_only_as_marked_diagnostic(rgb_img, evidence_factory, schemas):
    enc = mock_encoder("dinov2")
    r = enc.encode(rgb_img, color_order="RGB", evidence=evidence_factory(), diagnostic_only=True)
    assert r.record["quality"]["purpose"] == "pipeline_diagnostic_only"
    assert "diagnostic_only_not_for_target_object_representation" in r.record["warnings"]
    assert [e.message for e in Draft202012Validator(
        schemas["embedding_record.schema.json"]).iter_errors(r.record)] == []
    b = enc.encode_batch([rgb_img], color_order="RGB", evidences=[evidence_factory()],
                         diagnostic_only=True)
    assert b[0].record["quality"]["purpose"] == "pipeline_diagnostic_only"


def test_crop_records_are_not_marked_diagnostic(rgb_img, evidence_factory):
    ev = evidence_factory(source_kind="crop", bbox_xyxy=[0, 0, 80, 80])
    rec = mock_encoder("clip").encode(rgb_img, color_order="RGB", evidence=ev).record
    assert "purpose" not in rec["quality"]


def test_retain_pixels_request_rejected(rgb_img, evidence_factory):
    ev = evidence_factory(source_kind="crop", bbox_xyxy=[0, 0, 80, 80],
                          sensitivity={"may_contain_persons": True, "retain_pixels": True})
    with pytest.raises(UnsupportedEvidenceError, match="retain_pixels"):
        mock_encoder("dinov2").encode(rgb_img, color_order="RGB", evidence=ev)


def _walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _walk(v)
    else:
        yield o


def test_record_holds_no_pixel_data(rgb_img, evidence_factory):
    ev = evidence_factory(source_kind="crop", bbox_xyxy=[0, 0, 80, 80])
    for e in (ev, None):
        rec = mock_encoder("clip").encode(rgb_img, color_order="RGB", evidence=e).record
        leaves = list(_walk(rec))
        assert not any(isinstance(x, (np.ndarray, bytes, bytearray)) for x in leaves)
        assert len(json.dumps(rec)) < 4000          # a pixel dump would be orders larger
        assert rec["sensitivity"]["retain_pixels"] is False


def test_pinned_license_and_revision_flow_into_record(rgb_img):
    enc = mock_encoder("clip", license="MIT", revision="a" * 40)
    assert enc.encode(rgb_img, color_order="RGB").record["encoder"]["license"] == "MIT"
