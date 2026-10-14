from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from explainable_ai.video_explainer import explain_video_report
from explainable_ai.nlp_explainer import explain_nlp_output
from explainable_ai.fusion_explainer import explain_fusion_score
from explainable_ai.osint_explainer import explain_osint_result
from explainable_ai.report import build_xai_report, write_xai_report


class VideoExplainerTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "input_video": "test.mp4", "fps": 30, "frames_processed": 10,
            "detection_counts": {"weapon": 3, "fire": 0},
            "detection_frames": {"weapon": 2, "fire": 0},
            "detection_ratios": {"weapon": 0.2, "fire": 0.0},
            "average_confidence": {"weapon": 0.7, "fire": 0.0},
            "max_confidence": {"weapon": 0.9, "fire": 0.0},
            "detections": [
                {"category": "weapon", "detected": True, "detection_frames": 2, "detection_ratio": 0.2, "average_confidence": 0.7, "max_confidence": 0.9},
                {"category": "fire", "detected": False, "detection_frames": 0, "detection_ratio": 0.0, "average_confidence": 0.0, "max_confidence": 0.0},
            ],
            "threat_events": [{"frame": 2, "timestamp_sec": 0.067, "threats": ["weapon"], "detections": [{"category": "weapon", "class_name": "rifle", "confidence": 0.8, "bbox": [1, 2, 3, 4]}]}],
        }

    def test_detected_category_preserves_counts(self):
        out = explain_video_report(self.report)
        row = next(x for x in out["categories"] if x["category"] == "weapon")
        self.assertEqual(row["detection_count"], 3)
        self.assertEqual(row["frames_with_detection"], 2)
        self.assertAlmostEqual(row["mean_confidence"], 0.7)

    def test_negative_detection_not_claimed_as_absence(self):
        out = explain_video_report(self.report)
        row = next(x for x in out["categories"] if x["category"] == "fire")
        self.assertIn("not proof of absence", row["explanation"])

    def test_bbox_trace(self):
        out = explain_video_report(self.report)
        self.assertEqual(out["event_examples"][0]["detections"][0]["bounding_box_xyxy"], [1, 2, 3, 4])

    def test_rejects_wrong_schema(self):
        with self.assertRaises(ValueError): explain_video_report([1, 2])


class NLPExplainerTests(unittest.TestCase):
    def test_traces_rule_evidence_and_negation_flags(self):
        combined = {
            "video": {"filename": "clip.mp4"},
            "overall_assessment": {"status": "potential_threat_evidence_present", "decision_basis": "source-level evidence reconciliation", "detector_only_categories": ["weapon"], "description_only_categories": ["smoke"]},
            "gemini_video_understanding": {"text_derived_threat_evidence": {"smoke": [{"matched_term": "smoke", "evidence": "Smoke or vapor is visible.", "negated_or_denied": False, "uncertain_language": False}]}},
            "nlp_analysis": {"threat_reconciliation": {"smoke": {"status": "description_only", "detector_support": False, "gemini_text_support": True, "detector_evidence": {}, "explanation": "Description supports smoke."}}, "entities": [{"text": "daytime", "type": "TIME", "source": "gemini_description", "verified": False}], "keywords": ["smoke"], "limitations": ["No human verification."]},
        }
        out = explain_nlp_output(combined)
        smoke = next(x for x in out["categories"] if x["category"] == "smoke")
        self.assertEqual(smoke["reconciliation_status"], "description_only")
        self.assertEqual(smoke["matched_text_evidence"][0]["matched_term"], "smoke")
        self.assertTrue(any("does not perform SHAP" in item for item in out["limitations"]))

    def test_empty_nlp_schema_still_explains(self):
        out = explain_nlp_output({"nlp_analysis": {}, "overall_assessment": {}})
        self.assertEqual(out["categories"], [])


class FusionExplainerTests(unittest.TestCase):
    def setUp(self):
        self.score = {"zone": "validation_only", "window": {"seconds": 3600}, "threat_score": 0.28,
                      "threat_level": "LOW", "evidence_count": 1, "event_ids": ["evt"],
                      "contributions": [{"event_id": "evt", "event_type": "weapon_detection", "weight": 0.4, "confidence": 0.7, "sub_score": 0.28, "reason": "weapon detected"}],
                      "reasons": ["weapon detected"], "excluded_events": [{"event_id": "evt2", "reason": "outside window"}],
                      "run_metadata": {"validation_only": True, "warning": "validation only"}}

    def test_component_arithmetic(self):
        out = explain_fusion_score(self.score)
        self.assertTrue(out["contributions"][0]["arithmetic_matches"])
        self.assertTrue(out["contribution_sum_matches_score"])

    def test_exclusions_exposed(self):
        out = explain_fusion_score(self.score)
        self.assertEqual(out["exclusions"][0]["reason"], "outside window")

    def test_validation_warning_preserved(self):
        out = explain_fusion_score(self.score)
        self.assertTrue(out["validation_only"])
        self.assertEqual(out["warning"], "validation only")

    def test_level_threshold_explained_and_checked(self):
        out = explain_fusion_score(self.score)
        self.assertEqual(out["threat_level_rule"], "score < 0.30")
        self.assertTrue(out["threshold_consistency_check"])

    def test_demo_fixture_overrides_false_validation_flag(self):
        score = {
            "zone": "gate_2", "threat_score": 0.947, "threat_level": "CRITICAL",
            "event_ids": ["demo-crowd-001", "demo-osint-001"],
            "contributions": [
                {"event_id": "demo-crowd-001", "event_type": "crowd_alert", "weight": 0.2, "confidence": 1.0, "sub_score": 0.2, "reason": "crowd over capacity"},
                {"event_id": "demo-osint-001", "event_type": "osint_threat", "weight": 0.15, "confidence": 1.0, "sub_score": 0.15, "reason": "nearby (synthetic fixture)"},
            ],
            "reasons": ["synthetic fixture"], "excluded_events": [],
            "run_metadata": {"validation_only": False},
        }
        out = explain_fusion_score(score)
        self.assertTrue(out["validation_only"])
        self.assertEqual(out["validation_basis"], "synthetic_marker_detected")
        self.assertIn("software-validation output only", out["warning"])

    def test_non_demo_score_not_marked_validation_only_without_evidence(self):
        score = {
            "zone": "gate_2", "threat_score": 0.4, "threat_level": "MEDIUM",
            "event_ids": ["incident-123"], "contributions": [], "reasons": [],
            "excluded_events": [], "run_metadata": {"validation_only": False},
        }
        out = explain_fusion_score(score)
        self.assertFalse(out["validation_only"])
        self.assertIsNone(out["validation_basis"])


class OSINTExplainerTests(unittest.TestCase):
    def test_uses_actual_fields_without_hidden_rule_claims(self):
        out = explain_osint_result({"title": "Local report", "category": "violence", "severity": "High", "geographic_classification": "Nearby", "distance_km": 5.2, "publishedAt": "2026-10-12"})
        self.assertEqual(out["evidence"][0]["field"], "category")
        self.assertTrue(any("exact rule is not exposed" in row["meaning"] for row in out["evidence"] if row["field"] == "severity"))
        self.assertTrue(any("not necessarily the incident timestamp" in row["meaning"] for row in out["evidence"] if row["field"] == "publishedAt"))

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError): explain_osint_result([1])


class ReportTests(unittest.TestCase):
    def test_builds_selected_explanation_sections_only(self):
        out = build_xai_report(video_report={"detection_counts": {}, "threat_events": []})
        self.assertEqual(list(out["explanations"]), ["video"])

    def test_writes_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "xai.json"
            out = write_xai_report(path, threat_score={"threat_score": 0, "threat_level": "LOW", "contributions": [], "excluded_events": [], "reasons": []})
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "aegis.xai.v1")
            self.assertEqual(saved["explanations"]["fusion"]["threat_score"], 0)

    def test_visual_section_can_be_attached(self):
        out = build_xai_report(visual_explanation={"method": "EigenCAM"})
        self.assertEqual(out["explanations"]["visual"]["method"], "EigenCAM")

    def test_requires_input(self):
        with self.assertRaises(ValueError): build_xai_report()


if __name__ == "__main__":
    unittest.main()
