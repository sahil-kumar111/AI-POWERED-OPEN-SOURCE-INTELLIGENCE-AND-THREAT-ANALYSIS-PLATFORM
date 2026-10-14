import json
import tempfile
import unittest
from pathlib import Path

from osint_export import save_osint_results


class OSINTExportTests(unittest.TestCase):
    def test_export_preserves_original_rows_and_unicode(self):
        rows = [{"title": "மாதிரி incident", "publishedAt": "2026-10-13T12:00:00Z", "incident_latitude": 13.0, "custom": {"keep": True}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = save_osint_results(rows, Path(tmp) / "nested" / "osint_results.json")
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), rows)
            self.assertIn("மாதிரி", path.read_text(encoding="utf-8"))

    def test_empty_result_is_saved_as_empty_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_osint_results([], Path(tmp) / "output" / "osint_results.json")
            self.assertEqual(path.read_text(encoding="utf-8"), "[]\n")

    def test_non_list_results_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                save_osint_results({"results": []}, Path(tmp) / "osint_results.json")


if __name__ == "__main__":
    unittest.main()
