from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from representation_learning.text.location_source_adapter import (
    adapt_location_results,
    location_result_to_text_evidence,
)
from representation_learning.text.contracts import text_sha256
from representation_learning.text.errors import InvalidInputError


RETRIEVED = "2026-10-12T15:00:00Z"


def sample(**overrides):
    result = {
        "title": "Fire reported near central station",
        "description": "Authorities reported smoke near the station.",
        "source": "Example News",
        "publishedAt": "2026-10-12T12:30:00Z",
        "url": "https://example.test/article/1",
        "category": "Fire / Industrial",
        "severity": "Medium",
        "incident_location": "Chennai",
        "incident_latitude": 13.0827,
        "incident_longitude": 80.2707,
        "distance_km": 3.2,
        "geographic_classification": "Nearby",
    }
    result.update(overrides)
    return result


def test_maps_valid_location_result_and_preserves_provenance():
    item = location_result_to_text_evidence(sample(), retrieved_at=RETRIEVED)
    d = item.to_dict()
    assert d["source_kind"] == "article"
    assert d["source"]["source_name"] == "Example News"
    assert d["source"]["source_url"] == "https://example.test/article/1"
    assert d["source"]["source_item_id"].startswith("https://example.test/article/1#incident=")
    assert d["source"]["retrieved_at"] == RETRIEVED
    assert d["source"]["published_at"] == "2026-10-12T12:30:00Z"
    assert d["original_text"] == (
        "Fire reported near central station\nAuthorities reported smoke near the station."
    )
    assert d["original_text_sha256"] == text_sha256(d["original_text"])
    assert d["language"] == "und"
    assert d["truncated"] is False
    assert d["location"] == {
        "place_label": "Chennai", "lat": 13.0827, "lon": 80.2707
    }
    assert d["nlp_context"]["threat_category"] == "Fire / Industrial"
    assert "Fire / Industrial" not in d["original_text"]


def test_evidence_id_is_stable_and_tracks_source_text_hash():
    a = location_result_to_text_evidence(sample(), retrieved_at=RETRIEVED)
    b = location_result_to_text_evidence(sample(), retrieved_at=RETRIEVED)
    changed = location_result_to_text_evidence(
        sample(description="Updated details"), retrieved_at=RETRIEVED
    )
    assert a.evidence_id == b.evidence_id
    assert a.evidence_id != changed.evidence_id
    assert a.evidence_id.startswith("te_")
    assert len(a.evidence_id) == 19


def test_same_url_different_incident_context_is_not_deduplicated():
    rows = [
        sample(category="Fire / Industrial", incident_location="Chennai"),
        sample(category="Security", incident_location="Mumbai",
               incident_latitude=19.0760, incident_longitude=72.8777),
    ]
    items = adapt_location_results(rows, retrieved_at=RETRIEVED)
    assert len(items) == 2
    assert items[0].evidence_id != items[1].evidence_id


def test_url_missing_uses_deterministic_source_identity():
    row = sample(url=None)
    a = location_result_to_text_evidence(row, retrieved_at=RETRIEVED)
    b = location_result_to_text_evidence(row, retrieved_at=RETRIEVED)
    assert a.evidence_id == b.evidence_id
    assert a.to_dict()["source"]["source_url"] is None
    assert a.to_dict()["source"]["source_item_id"].startswith("newsitem:")


def test_caller_must_supply_retrieval_timestamp():
    with pytest.raises(TypeError):
        location_result_to_text_evidence(sample())
    with pytest.raises(InvalidInputError, match="retrieved_at"):
        location_result_to_text_evidence(sample(), retrieved_at="")


def test_bad_timestamps_rejected_by_contract():
    with pytest.raises(InvalidInputError, match="UTC offset"):
        location_result_to_text_evidence(
            sample(publishedAt="2026-10-12T12:30:00"), retrieved_at=RETRIEVED
        )
    with pytest.raises(InvalidInputError, match="UTC offset"):
        location_result_to_text_evidence(
            sample(), retrieved_at="2026-10-12T15:00:00"
        )


def test_missing_text_or_publisher_is_rejected():
    with pytest.raises(InvalidInputError, match="no article title or description"):
        location_result_to_text_evidence(
            sample(title=" ", description=None), retrieved_at=RETRIEVED
        )
    with pytest.raises(InvalidInputError, match="source_name"):
        location_result_to_text_evidence(
            sample(source=None), retrieved_at=RETRIEVED
        )


def test_oversized_text_rejected_not_truncated():
    with pytest.raises(InvalidInputError, match="never truncated"):
        location_result_to_text_evidence(
            sample(description="x" * 20001, title=""), retrieved_at=RETRIEVED
        )


def test_explicit_language_is_preserved():
    item = location_result_to_text_evidence(
        sample(), retrieved_at=RETRIEVED, language="ta"
    )
    assert item.language == "ta"


def test_invalid_results_container_rejected():
    with pytest.raises(InvalidInputError, match="list or tuple"):
        adapt_location_results("not rows", retrieved_at=RETRIEVED)


from representation_learning.text.nlp_source_adapter import nlp_record_to_text_evidence


def nlp_sample(description="Smoke was reported near the entrance."):
    return {
        "record_id": "record-17",
        "source": {
            "type": "video_report",
            "source_report": "output/clip_report.json",
            "input_video": "input/clip.mp4",
            "output_video": "output/clip_analyzed.mp4",
        },
        "analysis_text": (
            "User-provided incident description: " + description +
            "\nVideo-file and detector-count boilerplate."
        ),
        "entities": [{"text": "entrance", "label": "LOCATION"}],
        "threats": [{"category": "smoke"}],
        "events": [],
        "language": "en",
    }


def test_nlp_adapter_embeds_only_original_description():
    description = "Smoke was reported near the entrance."
    item = nlp_record_to_text_evidence(
        nlp_sample(description), original_description=description,
        retrieved_at=RETRIEVED,
    )
    d = item.to_dict()
    assert d["original_text"] == description
    assert "User-provided incident description" not in d["original_text"]
    assert "detector-count" not in d["original_text"]
    assert d["language"] == "und"
    assert d["source"]["source_item_id"] == "record-17"
    assert d["nlp_context"]["threats"] == [{"category": "smoke"}]


def test_nlp_adapter_refuses_unverified_description():
    with pytest.raises(InvalidInputError, match="could not be verified"):
        nlp_record_to_text_evidence(
            nlp_sample(), original_description="A different description",
            retrieved_at=RETRIEVED,
        )


def test_nlp_adapter_requires_description_and_timestamp():
    with pytest.raises(InvalidInputError, match="original_description"):
        nlp_record_to_text_evidence(
            nlp_sample(), original_description=" ", retrieved_at=RETRIEVED
        )
    with pytest.raises(InvalidInputError, match="retrieved_at"):
        nlp_record_to_text_evidence(
            nlp_sample(), original_description="Smoke was reported near the entrance.",
            retrieved_at=""
        )


def test_nlp_adapter_preserves_explicit_language_only_when_caller_supplies_it():
    item = nlp_record_to_text_evidence(
        nlp_sample(), original_description="Smoke was reported near the entrance.",
        retrieved_at=RETRIEVED, language="en",
    )
    assert item.language == "en"
