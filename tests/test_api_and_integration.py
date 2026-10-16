"""Integration and Unit Test Suite for AEGIS Threat Intelligence Platform.

Tests:
1. Backend FastAPI health & model endpoints
2. Video & Location input validation (missing location, invalid formats, file limits)
3. Threat fusion & scoring calculations (ATI index, weights, level thresholds, proximity multiplier)
4. ByteTrack unique person vs detection count verification
5. 7-Question executive report generation & schema verification
6. Artifacts generation & GeoJSON layer integrity
"""

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app
from backend.job_manager import job_manager
from backend.pipeline import (
    LocationGISAgent,
    ThreatFusionAgent,
    VideoIntelligenceAgent,
    run_complete_pipeline,
)

client = TestClient(app)


# =============================================================================
# 1. API HEALTH & MODEL TESTS
# =============================================================================
def test_api_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AEGIS Threat Intelligence Platform"
    assert "status" in data
    assert "models" in data
    assert "detectors_available" in data


def test_api_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "video_detectors" in data
    assert "weapon" in data["video_detectors"]
    assert "fire_smoke" in data["video_detectors"]
    assert "person" in data["video_detectors"]
    assert data["representation_learning"]["embedding_dimension"] == 384


# =============================================================================
# 2. INPUT VALIDATION TESTS
# =============================================================================
def test_analyze_empty_location_rejected():
    response = client.post("/api/analyze", data={"location": "   ", "venue_capacity": 15})
    assert response.status_code == 400
    assert "Location cannot be empty" in response.json()["detail"]


def test_analyze_unsupported_format_rejected():
    response = client.post(
        "/api/analyze",
        data={"location": "New Delhi", "venue_capacity": 15},
        files={"video": ("suspicious_script.sh", b"#!/bin/bash\necho test", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]


# =============================================================================
# 3. THREAT FUSION & SCORING TESTS
# =============================================================================
def test_threat_scoring_weights_and_thresholds():
    fusion_agent = ThreatFusionAgent()

    # Case A: Low risk baseline (no threats)
    mock_report_low = {
        "detection_counts": {"weapon": 0, "fire": 0, "smoke": 0, "person": 5},
        "max_confidence": {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.85},
        "person_statistics": {"crowd_over_capacity": False, "peak_concurrent_persons": 3, "venue_capacity": 15},
    }
    result_low = fusion_agent.fuse(mock_report_low, [], {"sentiment": {"compound": 0.0}})
    assert result_low["threat_level"] == "LOW"
    assert result_low["aegis_threat_index_ati"] < 30.0
    assert result_low["dominant_threat_category"] == "none"

    # Case B: High risk with weapon detection
    mock_report_high = {
        "detection_counts": {"weapon": 45, "fire": 0, "smoke": 0, "person": 3},
        "max_confidence": {"weapon": 0.85, "fire": 0.0, "smoke": 0.0, "person": 0.80},
        "person_statistics": {"crowd_over_capacity": False, "peak_concurrent_persons": 2, "venue_capacity": 15},
    }
    result_high = fusion_agent.fuse(mock_report_high, [], {"sentiment": {"compound": -0.40}}, infra_multiplier=1.15)
    assert result_high["threat_level"] in ("MEDIUM", "HIGH", "CRITICAL")
    assert result_high["dominant_threat_category"] == "weapon"
    assert result_high["threat_classification"] == "ARMED_ASSAULT_INCIDENT"
    assert result_high["threat_detected"] is True


def test_person_count_alone_does_not_cause_critical_threat():
    """Do not allow person count alone to determine threat severity."""
    fusion_agent = ThreatFusionAgent()
    mock_report_crowd = {
        "detection_counts": {"weapon": 0, "fire": 0, "smoke": 0, "person": 100},
        "max_confidence": {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.90},
        "person_statistics": {"crowd_over_capacity": True, "peak_concurrent_persons": 30, "venue_capacity": 10},
    }
    result_crowd = fusion_agent.fuse(mock_report_crowd, [], {"sentiment": {"compound": 0.0}})
    # Crowd alert weight is 0.20, so alone it cannot reach HIGH (>= 60) or CRITICAL (>= 80)
    assert result_crowd["threat_level"] == "LOW"
    assert result_crowd["aegis_threat_index_ati"] <= 25.0


# =============================================================================
# 4. LOCATION & GIS TESTS
# =============================================================================
def test_location_geocoding_and_infrastructure():
    gis_agent = LocationGISAgent()
    geo = gis_agent.geocode("Connaught Place, New Delhi")
    assert "latitude" in geo and "longitude" in geo
    assert abs(geo["latitude"] - 28.63) < 0.1

    infra, mult = gis_agent.identify_critical_infrastructure(geo["latitude"], geo["longitude"])
    assert len(infra) > 0
    assert mult >= 1.0


def test_no_fabricated_infrastructure_for_remote_location():
    gis_agent = LocationGISAgent()
    # Coordinates in rural ocean/desert area far from known cities
    infra, mult = gis_agent.identify_critical_infrastructure(0.0, 0.0)
    assert infra == []
    assert mult == 1.0


# =============================================================================
# 5. END-TO-END PIPELINE & REPORT SCHEMA TEST
# =============================================================================
def test_end_to_end_pipeline_execution(tmp_path):
    sample_video = PROJECT_ROOT / "input" / "YOUTUBE_YouTubeCCTV196_gun_2.mp4"
    if not sample_video.is_file():
        pytest.skip("Sample video file not available.")

    result = run_complete_pipeline(
        location="Connaught Place, New Delhi",
        video_path=sample_video,
        venue_capacity=15,
        output_dir=tmp_path,
        frame_skip=5,  # Faster test execution
    )

    assert result["status"] == "success"
    assert "event_overview" in result
    assert "person_and_crowd_analysis" in result
    assert "threat_detection" in result
    assert "heatmaps_and_visualizations" in result
    assert "final_intelligence_report" in result

    # Check 7 mandatory questions
    seven_q = result["final_intelligence_report"]["seven_questions"]
    assert "q1_what_is_happening" in seven_q
    assert "q2_how_many_people" in seven_q
    assert "q3_which_threats_detected" in seven_q
    assert "q4_dominant_threat" in seven_q
    assert "q5_overall_threat_severity_and_why" in seven_q
    assert "q6_evidence_supporting_conclusion" in seven_q
    assert "q7_limitations_and_additional_info" in seven_q

    # Check ByteTrack person tracking distinction
    p_analysis = result["person_and_crowd_analysis"]
    assert "unique_people_count" in p_analysis
    assert "total_person_detections" in p_analysis
    assert "detections_vs_unique_explanation" in p_analysis

    # Check saved artifacts
    assert (tmp_path / "threat_score.json").is_file()
    assert (tmp_path / "daily_intelligence_report.md").is_file()
    assert (tmp_path / "daily_intelligence_report.json").is_file()
    assert (tmp_path / "gis_layer.geojson").is_file()
