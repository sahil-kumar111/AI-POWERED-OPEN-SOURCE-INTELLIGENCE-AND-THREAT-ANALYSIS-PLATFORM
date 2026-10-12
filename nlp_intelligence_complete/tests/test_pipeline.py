import unittest
from nlp_intelligence import process_record, process_batch


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "input_video": "input/test.mp4",
            "fps": 10,
            "detection_counts": {"weapon": 4, "fire": 0, "smoke": 0, "person": 3},
            "detection_ratios": {"weapon": 0.5, "fire": 0, "smoke": 0, "person": 0.5},
            "average_confidence": {"weapon": 0.7, "fire": 0, "smoke": 0, "person": 0.8},
            "threat_events": [
                {"frame": 10, "threats": ["weapon"]},
                {"frame": 11, "threats": ["weapon"]},
                {"frame": 12, "threats": ["fire", "weapon"]},
            ],
        }

    def test_id_is_preserved(self):
        self.assertEqual(process_record(self.report, record_id="clip-1", use_ner=False)["record_id"], "clip-1")

    def test_weapon_is_extracted(self):
        result = process_record(self.report, use_ner=False)
        self.assertTrue(any(item["category"] == "weapon" for item in result["threats"]))

    def test_adjacent_frames_are_grouped(self):
        result = process_record(self.report, use_ner=False)
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["events"][0]["start_frame"], 10)
        self.assertEqual(result["events"][0]["end_frame"], 11)

    def test_sentiment_not_fabricated_without_text(self):
        result = process_record(self.report, use_ner=False)
        self.assertEqual(result["sentiment"]["label"], "not_applicable")

    def test_description_threat_candidate(self):
        result = process_record(
            self.report,
            incident_description="A possible explosion was reported near the station.",
            use_ner=False
        )
        self.assertTrue(any(item["category"] == "explosion" for item in result["threats"]))

    def test_bad_batch_record_returns_error(self):
        result = process_batch([self.report, {"detection_counts": "invalid"}], use_ner=False)
        self.assertEqual(result[1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
