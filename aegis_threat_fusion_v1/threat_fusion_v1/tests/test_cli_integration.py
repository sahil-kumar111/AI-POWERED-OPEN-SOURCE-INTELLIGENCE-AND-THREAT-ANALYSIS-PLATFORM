import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from threat_fusion_v1.cli import _peak_crowd_measurement

T = "2026-10-13T12:00:00Z"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"


class CLIIntegrationTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "threat_fusion_v1", *map(str, args)],
            cwd=PACKAGE_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_end_to_end_saved_video_and_osint_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "threat_score.json"
            result = self.run_cli(
                "--video-report", FIXTURES / "video_report_actual_shape.json",
                "--osint-json", FIXTURES / "osint_results.json",
                "--zone", "gate_2", "--event-time", T,
                "--latitude", "13.0827", "--longitude", "80.2707",
                "--window-seconds", "3600", "--osint-radius-meters", "1000",
                "--output", out,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "aegis.threat_fusion.v1")
            self.assertAlmostEqual(data["threat_score"], .40 * .7069 + .15)
            self.assertEqual(data["run_metadata"]["validation_only"], False)
            self.assertEqual(data["evidence_count"], 2)
            self.assertTrue(any(item["event_type"] == "osint_threat" for item in data["contributions"]))
            self.assertTrue(any(item.get("raw_ref") for item in data["contributions"]))

    def test_empty_osint_json_is_valid_and_does_not_add_osint_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.json"
            empty.write_text("[]\n", encoding="utf-8")
            out = Path(tmp) / "threat_score.json"
            result = self.run_cli(
                "--video-report", FIXTURES / "video_report_actual_shape.json",
                "--osint-json", empty, "--zone", "gate_2", "--event-time", T,
                "--latitude", "13.0827", "--longitude", "80.2707", "--output", out,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(data["threat_score"], .40 * .7069)
            self.assertEqual(data["evidence_count"], 1)

    def test_missing_incident_time_is_explicitly_excluded_unless_proxy_opted_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            osint = Path(tmp) / "published_only.json"
            osint.write_text(json.dumps([{
                "title": "Reported attack threat near site", "category": "Security",
                "publishedAt": T, "url": "https://example.test/published-only",
                "incident_latitude": 13.0828, "incident_longitude": 80.2708,
            }]), encoding="utf-8")
            out = Path(tmp) / "score.json"
            base = ["--video-report", FIXTURES / "video_report_actual_shape.json",
                    "--osint-json", osint, "--zone", "gate_2", "--event-time", T,
                    "--latitude", "13.0827", "--longitude", "80.2707", "--output", out]
            result = self.run_cli(*base)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(any(row["event_type"] == "osint_threat" for row in data["contributions"]))
            self.assertTrue(any("publishedAt is article publication time" in row["reason"] for row in data["excluded_events"]))
            result = self.run_cli(*base[:-2], "--allow-publication-time-proxy", *base[-2:])
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(data["run_metadata"]["explicit_publication_time_proxy_enabled"])
            # With the explicit proxy, the article is eligible only because this is a test fixture timestamp.
            self.assertTrue(any(row["event_type"] == "osint_threat" for row in data["contributions"]))

    def test_out_of_time_window_osint_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "score.json"
            result = self.run_cli(
                "--video-report", FIXTURES / "video_report_actual_shape.json",
                "--osint-json", FIXTURES / "osint_results.json",
                "--zone", "gate_2", "--event-time", T,
                "--latitude", "13.0827", "--longitude", "80.2707",
                "--window-seconds", "5", "--output", out,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(any(row["event_type"] == "osint_threat" for row in data["contributions"]))
            self.assertTrue(any("outside ±5s time window" in row["reason"] for row in data["excluded_events"]))

    def test_peak_concurrent_count_needs_timestamp_to_be_used(self):
        report = {"detections": [{"category": "person", "peak_concurrent_persons": 42}]}
        self.assertEqual(_peak_crowd_measurement(report), (42, None))
        report["detections"][0]["peak_timestamp_sec"] = 3.5
        self.assertEqual(_peak_crowd_measurement(report), (42, 3.5))

    def test_validation_only_output_is_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "score.json"
            result = self.run_cli(
                "--video-report", FIXTURES / "video_report_actual_shape.json",
                "--zone", "validation_zone", "--event-time", "2000-01-01T00:00:00Z",
                "--validation-only", "--output", out,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(data["run_metadata"]["validation_only"])
            self.assertIn("do not use this output", data["run_metadata"]["warning"])


if __name__ == "__main__":
    unittest.main()
