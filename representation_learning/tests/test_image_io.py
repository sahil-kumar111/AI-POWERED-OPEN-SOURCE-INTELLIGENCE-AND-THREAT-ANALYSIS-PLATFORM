import numpy as np
import pytest
from PIL import Image

from representation_learning.encoders.errors import InvalidInputError
from representation_learning.encoders.image_io import crop_bbox, to_rgb_uint8


def test_rgb_ndarray_passthrough(rgb_img):
    out = to_rgb_uint8(rgb_img, "RGB")
    assert out.shape == (120, 160, 3) and out.dtype == np.uint8
    assert np.array_equal(out, rgb_img)


def test_bgr_is_swapped_to_rgb():
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 0] = 200  # blue channel in BGR
    out = to_rgb_uint8(bgr, "BGR")
    assert out[0, 0].tolist() == [0, 0, 200]


def test_color_order_required_for_3ch_ndarray(rgb_img):
    with pytest.raises(InvalidInputError, match="color_order"):
        to_rgb_uint8(rgb_img)


def test_grayscale_2d_and_1ch():
    g = np.full((8, 9), 77, np.uint8)
    assert to_rgb_uint8(g).shape == (8, 9, 3)
    assert (to_rgb_uint8(g[..., None]) == 77).all()


def test_rgba_composited_on_white():
    rgba = np.zeros((2, 2, 4), np.uint8)
    rgba[..., :3] = 0          # black
    rgba[0, 0, 3] = 255        # opaque -> stays black
    rgba[0, 1, 3] = 0          # transparent -> white
    out = to_rgb_uint8(rgba, "RGB")
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == [255, 255, 255]


def test_pil_modes(pil_img):
    for mode in ("L", "RGBA", "P", "1"):
        out = to_rgb_uint8(pil_img.convert(mode))
        assert out.shape == (120, 160, 3) and out.dtype == np.uint8


def test_pil_unsupported_mode():
    with pytest.raises(InvalidInputError, match="mode"):
        to_rgb_uint8(Image.new("CMYK", (4, 4)))


@pytest.mark.parametrize("bad", [None, "path.jpg", [1, 2, 3], np.zeros((0, 5, 3), np.uint8),
                                 np.zeros((5, 5, 3), np.float32), np.zeros((5, 5, 2), np.uint8),
                                 np.zeros((5,), np.uint8), np.zeros((2, 2, 2, 3), np.uint8)])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(InvalidInputError):
        to_rgb_uint8(bad, "RGB")


def test_crop_bbox_clamps_and_returns_box(rgb_img):
    crop, box = crop_bbox(rgb_img, [-5.2, 10.4, 50.1, 9999])
    assert box == [0, 10, 51, 120]
    assert crop.shape == (110, 51, 3)


@pytest.mark.parametrize("box", [[10, 10, 10, 50], [200, 0, 300, 50], [5, 5, 4, 4],
                                 [0, 0, np.nan, 5], [0, 0, 5]])
def test_crop_bbox_rejects_bad_boxes(rgb_img, box):
    with pytest.raises(InvalidInputError):
        crop_bbox(rgb_img, box)
