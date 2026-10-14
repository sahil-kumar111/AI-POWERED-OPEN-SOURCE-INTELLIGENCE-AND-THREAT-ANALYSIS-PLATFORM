import hashlib

import pytest

from representation_learning.encoders.errors import EncoderIntegrityError, EncoderLoadError
from representation_learning.encoders.integrity import locate_weights, sha256_file, verify_weights


def _mk(tmp_path, weights=b"abc"):
    (tmp_path / "model.safetensors").write_bytes(weights)
    (tmp_path / "config.json").write_text("{}")
    return tmp_path


def test_sha256_matches_hashlib(tmp_path):
    d = _mk(tmp_path)
    assert sha256_file(d / "model.safetensors") == hashlib.sha256(b"abc").hexdigest()


def test_locate_errors(tmp_path):
    with pytest.raises(EncoderLoadError):
        locate_weights(None)
    with pytest.raises(EncoderLoadError, match="not found"):
        locate_weights(tmp_path / "nope")
    with pytest.raises(EncoderLoadError, match="model.safetensors"):
        locate_weights(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(EncoderLoadError, match="config.json"):
        locate_weights(tmp_path)


def test_sharded_rejected(tmp_path):
    d = _mk(tmp_path)
    (d / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(EncoderLoadError, match="sharded"):
        locate_weights(d)


def test_verify_paths(tmp_path):
    f = locate_weights(_mk(tmp_path))
    good = hashlib.sha256(b"abc").hexdigest()
    assert verify_weights(f, good.upper(), False) == (good, [])
    with pytest.raises(EncoderIntegrityError, match="mismatch"):
        verify_weights(f, "0" * 64, False)
    with pytest.raises(EncoderIntegrityError, match="required"):
        verify_weights(f, None, False)
    sha, warns = verify_weights(f, None, True)
    assert sha == good and warns
