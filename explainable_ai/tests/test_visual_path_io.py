from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from explainable_ai.visual_explainer import _imread_unicode, _imwrite_unicode


@unittest.skipIf(cv2 is None or np is None, "requires opencv-python and numpy")
class UnicodeImagePathTests(unittest.TestCase):
    def test_png_round_trip_through_unicode_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "AEGIS_文档_тест"
            target = folder / "frame_0.png"
            original = np.zeros((12, 16, 3), dtype=np.uint8)
            original[:, :, 1] = 180
            self.assertTrue(_imwrite_unicode(target, original, cv2))
            loaded = _imread_unicode(target, cv2)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, original.shape)
            self.assertTrue(np.array_equal(loaded, original))


if __name__ == "__main__":
    unittest.main()
