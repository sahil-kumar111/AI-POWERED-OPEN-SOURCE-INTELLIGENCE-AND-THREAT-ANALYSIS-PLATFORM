import json
import unittest
from datetime import datetime
from pathlib import Path

from threat_fusion_v1.adapters import video_report_to_events, osint_rows_to_events, crowd_count_to_event
from threat_fusion_v1.engine import ThreatFusionEngine

T = "2026-10-13T12:00:00Z"
FIXTURES = Path(__file__).parent / "fixtures"


class AdapterTests(unittest.TestCase):
    def test_video_report_aggregates_repeated_frame_detections(self):
        report = {
            "input_video": "input/clip.mp4",
            "detection_counts": {"weapon": 115, "fire": 15, "smoke": 0},
            "detection_frames": {"weapon": 83, "fire": 15, "smoke": 0},
            "detection_ratios": {"weapon": .92, "fire": .17, "smoke": 0},
            "average_confidence": {"weapon": .47, "fire": .52, "smoke": 0},
            "max_confidence": {"weapon": .89, "fire": .91, "smoke": 0},
        }
        events = video_report_to_events(report, zone="gate_2", capture_time=T, latitude=13.0827, longitude=80.2707)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["confidence"], .89)
        self.assertEqual(events[0]["details"]["detection_count_across_report"], 115)
        result = ThreatFusionEngine().fuse(events, zone="gate_2", reference_time=T)
        self.assertAlmostEqual(result["threat_score"], .4 * .89 + .25 * .91)

    def test_actual_threat_video_pipeline_schema_uses_max_and_relative_timestamps(self):
        report = json.loads((FIXTURES / "video_report_actual_shape.json").read_text(encoding="utf-8"))
        events = video_report_to_events(
            report, zone="gate_2", capture_time=T, latitude=13.0827, longitude=80.2707,
            report_ref="output/YOUTUBE_YouTubeCCTV196_gun_2_report.json",
        )
        self.assertEqual([event["type"] for event in events], ["weapon_detection", "fire"])
        self.assertEqual(events[0]["confidence"], .7069)
        self.assertEqual(events[1]["confidence"], .5725)
        self.assertEqual(events[0]["subtype"], "rifle")
        self.assertEqual(events[0]["timestamp"], T)
        self.assertEqual(events[1]["details"]["timestamp_basis"], "capture_time_plus_first_threat_event_timestamp_sec")
        self.assertEqual(events[0]["details"]["detection_count_across_report"], 115)
        result = ThreatFusionEngine(time_window_seconds=3600).fuse(events, zone="gate_2", reference_time=T)
        # Only weapon confidence > .60 qualifies; fire confidence remains unchanged and is excluded.
        self.assertAlmostEqual(result["threat_score"], .40 * .7069)
        self.assertTrue(any("strictly > 0.60" in row["reason"] for row in result["excluded_events"]))
        self.assertIn("source_report", events[0]["details"])
        self.assertEqual(result["contributions"][0]["raw_ref"], "output/YOUTUBE_YouTubeCCTV196_gun_2_report.json")

    def test_current_video_report_schema_is_supported(self):
        report = {
            "input_video": r"input\crowd.mp4", "fps": 30.0,
            "detections": [
                {"category": "weapon", "class_name": "knife", "detected": True,
                 "avg_confidence": 0.5107, "frames_detected": 2, "first_seen_sec": 5.7, "last_seen_sec": 17.13},
                {"category": "smoke", "detected": True, "avg_confidence": 0.5453, "first_seen_sec": 5.13},
                {"category": "person", "detected": True, "count": 50, "peak_concurrent_persons": 15},
            ],
        }
        events = video_report_to_events(report, zone="gate_2", capture_time=T)
        self.assertEqual([e["type"] for e in events], ["weapon_detection", "smoke"])
        self.assertEqual(events[0]["confidence"], .5107)
        self.assertEqual(events[0]["timestamp"], "2026-10-13T12:00:05.700000Z")
        result = ThreatFusionEngine(time_window_seconds=60).fuse(events, zone="gate_2", reference_time=T)
        self.assertEqual(result["threat_score"], 0.0)
        self.assertEqual(len(result["excluded_events"]), 2)
        self.assertTrue(all("strictly > 0.60" in row["reason"] for row in result["excluded_events"]))

    def test_duplicate_reports_same_clip_do_not_double_score(self):
        report = {"input_video": "input/clip.mp4", "detection_counts": {"weapon": 2}, "max_confidence": {"weapon": .9}}
        first = video_report_to_events(report, zone="gate_2", capture_time=T)
        second = video_report_to_events(report, zone="gate_2", capture_time=T)
        result = ThreatFusionEngine().fuse(first + second, zone="gate_2", reference_time=T)
        self.assertAlmostEqual(result["threat_score"], .36)
        self.assertEqual(result["evidence_count"], 1)
        self.assertTrue(any("duplicate" in row["reason"] for row in result["excluded_events"]))

    def test_osint_article_contract(self):
        rows = [{"title": "Police investigate reported attack threat near site", "description": "weapon seized", "source": "Local news",
                 "publishedAt": T, "url": "https://example.test/item", "category": "Weapons / Explosives", "severity": "High",
                 "incident_latitude": 13.0828, "incident_longitude": 80.2708, "distance_km": .02, "geographic_classification": "Nearby"}]
        events = osint_rows_to_events(rows, zone="gate_2", allow_publication_time_proxy=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "osint_threat")
        self.assertIn("binary upstream", events[0]["details"]["confidence_basis"])
        self.assertIn("recency_proxy_not_incident_time", events[0]["details"]["timestamp_basis"])

    def test_published_at_not_used_as_incident_time_by_default(self):
        row = {"title": "Reported attack threat", "category": "Violence / Security", "incident_latitude": 13.0,
               "incident_longitude": 80.0, "publishedAt": T, "url": "https://example.test/pub-only"}
        events = osint_rows_to_events([row], zone="gate_2")
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]["timestamp"])
        self.assertIn("publishedAt is article publication time", events[0]["details"]["timestamp_exclusion_reason"])
        result = ThreatFusionEngine().fuse(events, zone="gate_2", reference_time=T)
        self.assertTrue(any("publishedAt is article publication time" in row["reason"] for row in result["excluded_events"]))

    def test_no_osint_location_or_timestamp_is_not_fabricated(self):
        rows = [
            {"title": "A weapon threat", "category": "Weapons", "incident_latitude": None, "incident_longitude": None, "incident_timestamp": T, "url": "https://example.test/no-coords"},
            {"title": "A weapon threat", "category": "Weapons", "incident_latitude": 500, "incident_longitude": 900, "incident_timestamp": "not-a-time", "url": "https://example.test/bad"},
        ]
        events = osint_rows_to_events(rows, zone="gate_2")
        self.assertEqual(len(events), 2)
        result = ThreatFusionEngine().fuse(events, zone="gate_2", reference_time=T)
        reasons = " ".join(x["reason"] for x in result["excluded_events"])
        self.assertIn("missing or invalid OSINT incident coordinates", reasons)
        self.assertIn("invalid ISO-8601 timestamp", reasons)

    def test_no_video_camera_anchor_explains_exclusion(self):
        rows = [{"title": "An attack threat", "category": "Security", "incident_latitude": 13.0,
                 "incident_longitude": 80.0, "incident_timestamp": T, "url": "https://example.test/no-camera"}]
        events = osint_rows_to_events(rows, zone="gate_2")
        result = ThreatFusionEngine().fuse(events, zone="gate_2", reference_time=T)
        self.assertTrue(any("actual camera coordinates are missing" in row["reason"] for row in result["excluded_events"]))

    def test_out_of_radius_osint_is_excluded(self):
        video = video_report_to_events({"input_video": "clip.mp4", "detection_counts": {"weapon": 1}, "max_confidence": {"weapon": .9}},
                                       zone="gate_2", capture_time=T, latitude=13.0827, longitude=80.2707)
        osint = osint_rows_to_events([{"title": "Reported attack threat", "category": "Security", "incident_timestamp": T,
                                      "incident_latitude": 14.0, "incident_longitude": 80.0}], zone="gate_2")
        result = ThreatFusionEngine(osint_radius_meters=1000).fuse(video + osint, zone="gate_2", reference_time=T)
        self.assertTrue(any("outside OSINT radius" in row["reason"] for row in result["excluded_events"]))

    def test_empty_osint_array_is_valid(self):
        self.assertEqual(osint_rows_to_events([], zone="gate_2"), [])

    def test_crowd_adapter_uses_single_measurement(self):
        event = crowd_count_to_event(340, 300, zone="gate_2", timestamp=T, latitude=13.0827, longitude=80.2707)
        result = ThreatFusionEngine().fuse([event], zone="gate_2", reference_time=T)
        self.assertAlmostEqual(result["threat_score"], .2)
        self.assertIn("single measurement", event["details"]["count_basis"])

    def test_timezone_offsets_are_normalized(self):
        event = crowd_count_to_event(10, 5, zone="gate_2", timestamp="2026-10-13T17:30:00+05:30")
        self.assertEqual(event["timestamp"], T)

    def test_missing_osint_id_uses_deterministic_rule(self):
        row = {"title": "Attack threat", "category": "Security", "url": "https://example.test/stable", "publishedAt": T}
        first = osint_rows_to_events([row], zone="gate_2")[0]["event_id"]
        second = osint_rows_to_events([row], zone="gate_2")[0]["event_id"]
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("osint:sha256:"))


if __name__ == "__main__":
    unittest.main()
