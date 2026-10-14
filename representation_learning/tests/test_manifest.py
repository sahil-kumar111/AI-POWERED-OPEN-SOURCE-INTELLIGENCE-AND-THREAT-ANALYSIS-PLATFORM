"""Manifest tests: fake files for pin/verify logic; tiny random checkpoints for the
manifest -> encoder path. No network, no pretrained weights."""

import json
from pathlib import Path

import pytest

from representation_learning.encoders import manifest as M
from representation_learning.encoders.errors import EncoderIntegrityError, EncoderLoadError

REV = "0123456789abcdef0123456789abcdef01234567"


def _setup(tmp_path, name="dinov2-small", real=False):
    tpl = json.loads(M.DEFAULT_MANIFEST.read_text())
    mdir = tmp_path / "models"
    mp = tmp_path / "models.manifest.json"
    mp.write_text(json.dumps(tpl))
    d = mdir / tpl["models"][name]["local_dir"].split("/")[-1]
    d.mkdir(parents=True)
    if not real:
        for f in M.REQUIRED_FILES:
            (d / f).write_bytes(f"fake-{f}".encode())
    return mp, d


def test_committed_manifest_is_a_template_with_no_pins():
    m = M.load_manifest()
    assert set(m["models"]) == {"dinov2-small", "clip-vit-b-32"}
    for e in m["models"].values():
        assert M.pin_problems(e)                       # nothing pinned yet
        assert e["revision"] == M.PLACEHOLDER and e["files"]["model.safetensors"] == M.PLACEHOLDER
        assert e["source_url"].startswith("https://huggingface.co/")
        assert "unverified" in e["license_expected_per_model_card"].lower()
    assert m["models"]["dinov2-small"]["expected_dim"] == 384
    assert m["models"]["clip-vit-b-32"]["expected_dim"] == 512


def test_unpinned_entry_refuses_everything(tmp_path):
    mp, _ = _setup(tmp_path)
    with pytest.raises(EncoderLoadError, match="not pinned"):
        M.verify_entry(M.load_manifest(mp), "dinov2-small", mp)
    with pytest.raises(EncoderLoadError, match="not pinned"):
        M.encoder_from_manifest("dinov2-small", mp)


def test_record_then_verify_roundtrip_and_tamper_detection(tmp_path):
    mp, d = _setup(tmp_path)
    e = M.record_entry(mp, "dinov2-small", REV, "Apache-2.0 (copied from card)")
    assert e["revision"] == REV and len(e["files"]["model.safetensors"]) == 64
    assert M.pin_problems(M.load_manifest(mp)["models"]["dinov2-small"]) == []
    assert M.verify_entry(M.load_manifest(mp), "dinov2-small", mp) == d.resolve()
    (d / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(EncoderIntegrityError, match="mismatch"):
        M.verify_entry(M.load_manifest(mp), "dinov2-small", mp)
    (d / "config.json").unlink()
    with pytest.raises(EncoderIntegrityError, match="missing"):
        M.verify_entry(M.load_manifest(mp), "dinov2-small", mp)


@pytest.mark.parametrize("rev,lic", [("abc", "MIT"), (REV, ""), (REV, M.PLACEHOLDER)])
def test_record_rejects_bad_revision_or_license(tmp_path, rev, lic):
    mp, _ = _setup(tmp_path)
    with pytest.raises(EncoderIntegrityError):
        M.record_entry(mp, "dinov2-small", rev, lic)


def test_record_requires_local_files_and_known_name(tmp_path):
    mp, d = _setup(tmp_path)
    (d / "preprocessor_config.json").unlink()
    with pytest.raises(EncoderLoadError, match="not found"):
        M.record_entry(mp, "dinov2-small", REV, "MIT")
    with pytest.raises(EncoderLoadError, match="unknown manifest entry"):
        M.record_entry(mp, "nope", REV, "MIT")


def test_cli_status_and_verify_exit_codes(tmp_path, capsys):
    mp, _ = _setup(tmp_path)
    assert M._main(["--manifest", str(mp), "status"]) == 0
    assert "NOT PINNED" in capsys.readouterr().out
    assert M._main(["--manifest", str(mp), "verify"]) == 2
    assert M._main(["--manifest", str(mp), "record", "dinov2-small", "--revision", REV,
                    "--license", "Apache-2.0"]) == 0
    assert M._main(["--manifest", str(mp), "verify", "dinov2-small"]) == 0


@pytest.mark.tiny_random
@pytest.mark.parametrize("name,maker,dim", [("dinov2-small", "make_tiny_dinov2", 384),
                                            ("clip-vit-b-32", "make_tiny_clip", 512)])
def test_manifest_to_encoder_with_tiny_random_checkpoint(tmp_path, name, maker, dim, rgb_img):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from representation_learning.testing import tiny_checkpoints as tc
    mp, d = _setup(tmp_path, name, real=True)
    getattr(tc, maker)(d)
    M.record_entry(mp, name, REV, "TEST-LICENSE (tiny random)")
    enc = M.encoder_from_manifest(name, mp, device="cpu")
    r = enc.encode(rgb_img, color_order="RGB")
    assert r.dim == dim
    assert r.record["encoder"]["version_or_checkpoint"].endswith("@" + REV)
    assert r.record["encoder"]["license"] == "TEST-LICENSE (tiny random)"
    assert r.record["warnings"] == []          # hash verified against the recorded value
