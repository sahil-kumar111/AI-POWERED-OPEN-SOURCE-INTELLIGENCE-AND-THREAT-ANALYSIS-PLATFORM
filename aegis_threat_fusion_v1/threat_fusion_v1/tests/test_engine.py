import unittest
from threat_fusion_v1.engine import ThreatFusionEngine

T = "2026-10-13T12:00:00Z"


def event(event_id, kind, confidence=0.9, details=None, *, zone="gate_2", modality="video", lat=13.0827, lon=80.2707, timestamp=T, source_id="cam_1"):
    return {
        "event_id": event_id, "timestamp": timestamp,
        "source": {"type": "camera" if modality == "video" else "osint", "id": source_id, "zone": zone},
        "type": kind, "confidence": confidence, "location": {"lat": lat, "lon": lon},
        "details": details or {}, "modality": modality,
    }


class ThreatFusionTests(unittest.TestCase):
    def setUp(self):
        self.engine = ThreatFusionEngine(time_window_seconds=60, osint_radius_meters=1000)

    def test_weapon_weighted_score_and_level(self):
        result = self.engine.fuse([event("w1", "weapon_detection", 0.9)], zone="gate_2", reference_time=T)
        self.assertAlmostEqual(result["threat_score"], 0.36)
        self.assertEqual(result["threat_level"], "MEDIUM")
        self.assertEqual(result["event_ids"], ["w1"])

    def test_all_four_modalities_can_reach_critical_under_weights(self):
        rows = [
            event("w", "weapon_detection", 0.9, {"track_id": 1}),
            event("c", "crowd_alert", 1.0, {"count": 340, "capacity": 300}),
            event("f", "fire", 0.9),
            event("o", "osint_threat", 1.0, {"title": "Reported attack threat near gate", "confidence_basis": "test"}, modality="text", source_id="feed_1", lat=13.0828, lon=80.2708),
        ]
        result = self.engine.fuse(rows, zone="gate_2", reference_time=T)
        self.assertAlmostEqual(result["threat_score"], 0.36 + 0.2 + 0.225 + 0.15)
        self.assertEqual(result["threat_level"], "CRITICAL")
        self.assertEqual(set(result["event_ids"]), {"w", "c", "f", "o"})

    def test_three_signal_case_does_not_fabricate_critical(self):
        rows = [event("w", "weapon_detection", 0.93), event("c", "crowd_alert", details={"count": 340, "capacity": 300}),
                event("o", "osint_threat", 1.0, {"title": "threat attack"}, modality="text", source_id="news", lat=13.0828, lon=80.2708)]
        result = self.engine.fuse(rows, zone="gate_2", reference_time=T)
        self.assertAlmostEqual(result["threat_score"], 0.372 + 0.2 + 0.15)
        self.assertEqual(result["threat_level"], "HIGH")

    def test_threshold_is_strictly_greater_than_point_six(self):
        result = self.engine.fuse([event("w", "weapon_detection", 0.60)], zone="gate_2", reference_time=T)
        self.assertEqual(result["threat_score"], 0)
        self.assertTrue(any("strictly > 0.60" in row["reason"] for row in result["excluded_events"]))

    def test_boundary_levels(self):
        engine = ThreatFusionEngine()
        self.assertEqual(engine.fuse([event("x", "weapon_detection", .75)], zone="gate_2")["threat_level"], "MEDIUM")
        # 0.4 * 1 isn't eligible above? This asserts boundary helper by combinations.
        rows = [event("w", "weapon_detection", 1.0), event("c", "crowd_alert", details={"count": 101, "capacity": 100})]
        self.assertEqual(engine.fuse(rows, zone="gate_2")["threat_level"], "HIGH")  # exact 0.60

    def test_track_duplicates_keep_highest_confidence(self):
        rows = [event("w1", "weapon_detection", .7, {"track_id": "track-x"}), event("w2", "weapon_detection", .9, {"track_id": "track-x"})]
        result = self.engine.fuse(rows, zone="gate_2")
        self.assertEqual(result["event_ids"], ["w2"])
        self.assertTrue(any("lower-confidence" in item["reason"] for item in result["excluded_events"]))

    def test_osint_outside_radius_excluded(self):
        rows = [event("w", "weapon_detection", .9), event("o", "osint_threat", 1.0, {"title": "attack threat"}, modality="text", lat=14.0, lon=80.0, source_id="news")]
        result = self.engine.fuse(rows, zone="gate_2")
        self.assertNotIn("o", result["event_ids"])
        self.assertTrue(any("outside OSINT radius" in row["reason"] for row in result["excluded_events"]))

    def test_osint_requires_keyword_and_video_location_anchor(self):
        no_keyword = event("o", "osint_threat", 1.0, {"title": "A routine weather update"}, modality="text", source_id="news")
        result = self.engine.fuse([event("w", "weapon_detection", .9), no_keyword], zone="gate_2")
        self.assertTrue(any("keyword match" in row["reason"] for row in result["excluded_events"]))
        no_anchor = event("o2", "osint_threat", 1.0, {"title": "attack threat"}, modality="text", source_id="news")
        result = self.engine.fuse([no_anchor], zone="gate_2")
        self.assertTrue(any("video event location anchor" in row["reason"] for row in result["excluded_events"]))

    def test_zone_and_time_window(self):
        rows = [event("wrong-zone", "weapon_detection", zone="gate_3"), event("old", "fire", timestamp="2026-10-13T11:00:00Z")]
        result = self.engine.fuse(rows, zone="gate_2", reference_time=T)
        self.assertEqual(result["evidence_count"], 0)
        self.assertEqual(len(result["excluded_events"]), 2)

    def test_invalid_confidence_does_not_crash(self):
        result = self.engine.fuse([event("w", "weapon_detection", 1.2)], zone="gate_2")
        self.assertEqual(result["threat_score"], 0)
        self.assertTrue(result["excluded_events"])

    def test_score_capped_at_one(self):
        rows = [event("w", "weapon_detection", 1.0), event("c", "crowd_alert", details={"count": 999, "capacity": 1}),
                event("f", "fire", 1.0), event("s", "smoke", 1.0)]
        result = self.engine.fuse(rows, zone="gate_2")
        self.assertEqual(result["threat_score"], 1.0)
        self.assertEqual(result["threat_level"], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
