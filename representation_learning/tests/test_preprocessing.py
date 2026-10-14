import json

import numpy as np
import pytest

from representation_learning.encoders.errors import EncoderIntegrityError
from representation_learning.encoders.preprocessing import (
    CLIP_PREPROCESS, DINOV2_PREPROCESS, check_checkpoint_preprocessing, preprocess)
from representation_learning.testing.tiny_checkpoints import (CLIP_PREPROC_JSON,
                                                             DINOV2_PREPROC_JSON)


@pytest.mark.parametrize("spec", [DINOV2_PREPROCESS, CLIP_PREPROCESS])
@pytest.mark.parametrize("shape", [(100, 300, 3), (300, 100, 3), (224, 224, 3), (7, 9, 3)])
def test_output_shape_dtype_finite(spec, shape):
    x = preprocess(np.full(shape, 128, np.uint8), spec)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32 and np.isfinite(x).all()


@pytest.mark.parametrize("spec", [DINOV2_PREPROCESS, CLIP_PREPROCESS])
def test_solid_color_normalisation_values(spec):
    x = preprocess(np.full((64, 64, 3), 255, np.uint8), spec)
    expect = (1.0 - np.array(spec.mean)) / np.array(spec.std)
    assert np.allclose(x[:, 100, 100], expect, atol=1e-5)


def test_preprocessing_is_deterministic(rgb_img):
    a = preprocess(rgb_img, DINOV2_PREPROCESS)
    b = preprocess(rgb_img.copy(), DINOV2_PREPROCESS)
    assert np.array_equal(a, b)


def test_specs_differ_and_are_identified():
    assert DINOV2_PREPROCESS.id != CLIP_PREPROCESS.id
    assert DINOV2_PREPROCESS.resize_shortest_edge == 256 and CLIP_PREPROCESS.resize_shortest_edge == 224
    assert DINOV2_PREPROCESS.to_dict()["color_order"] == "RGB"


def _write(tmp_path, cfg):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_checkpoint_crosscheck_ok(tmp_path):
    assert check_checkpoint_preprocessing(_write(tmp_path, DINOV2_PREPROC_JSON), DINOV2_PREPROCESS) == []
    assert check_checkpoint_preprocessing(_write(tmp_path, CLIP_PREPROC_JSON), CLIP_PREPROCESS) == []


def test_checkpoint_crosscheck_missing_file_warns(tmp_path):
    w = check_checkpoint_preprocessing(tmp_path, CLIP_PREPROCESS)
    assert w and "not_found" in w[0]


@pytest.mark.parametrize("patch", [{"image_mean": [0.5, 0.5, 0.5]}, {"crop_size": 256},
                                   {"size": {"shortest_edge": 300}}, {"do_center_crop": False},
                                   {"resample": 2}])
def test_checkpoint_crosscheck_mismatch(tmp_path, patch):
    cfg = dict(CLIP_PREPROC_JSON, **patch)
    with pytest.raises(EncoderIntegrityError):
        check_checkpoint_preprocessing(_write(tmp_path, cfg), CLIP_PREPROCESS)
