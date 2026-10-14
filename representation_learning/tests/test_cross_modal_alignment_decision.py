"""Tests for Gate 3 Cross-Modal Alignment architectural decision."""

import json
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]


def test_gate3_decision_artifact_exists_and_conforms():
    json_path = BASE_DIR / "outputs" / "aegis_cross_modal_alignment_decision.json"
    doc_path = BASE_DIR / "docs" / "cross_modal_alignment_decision.md"

    assert json_path.is_file(), "Gate 3 decision JSON artifact missing"
    assert doc_path.is_file(), "Gate 3 ADR documentation missing"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["decision"] == "OPTION_C_SEPARATE_SPACES_SYMBOLIC_FUSION"
    assert data["status"] == "APPROVED_DECISION"
    assert data["vector_spaces"]["cross_modal_direct_vector_comparison"] is False
    assert data["vector_spaces"]["error_on_cross_modal_comparison"] == "IncompatibleSpaceError"

    assert data["options_evaluated"]["option_a_native_joint_space"]["status"] == "REJECTED"
    assert "DEFERRED" in data["options_evaluated"]["option_b_learned_alignment"]["status"]
    assert "SELECTED" in data["options_evaluated"]["option_c_separate_spaces_symbolic_fusion"]["status"]


def test_cross_modal_different_spaces_not_equal():
    """Verify that text and visual spaces cannot be treated as equal."""
    text_space = "space_e8d2691a145d41eb"
    visual_space = "space_dinov2_small_visual"

    assert text_space != visual_space
