"""Integration tests for Gate 1 semantic quality evaluation and metric computation."""

import json
from pathlib import Path
import pytest

from representation_learning.evaluation.eval_semantic_quality import (
    SimpleBM25,
    compute_metrics,
    compute_ndcg_at_k
)

BASE_DIR = Path(__file__).resolve().parents[1]


def test_ndcg_computation_perfect_order():
    judgments = {
        "doc1": {"relevance_grade": 2, "binary_relevance": 1},
        "doc2": {"relevance_grade": 1, "binary_relevance": 1},
        "doc3": {"relevance_grade": 0, "binary_relevance": 0},
    }
    ranked = ["doc1", "doc2", "doc3"]
    score = compute_ndcg_at_k(ranked, judgments, 3)
    assert pytest.approx(score, 1e-4) == 1.0


def test_ndcg_computation_reversed_order():
    judgments = {
        "doc1": {"relevance_grade": 2, "binary_relevance": 1},
        "doc2": {"relevance_grade": 1, "binary_relevance": 1},
        "doc3": {"relevance_grade": 0, "binary_relevance": 0},
    }
    ranked = ["doc3", "doc2", "doc1"]
    score = compute_ndcg_at_k(ranked, judgments, 3)
    assert 0.0 < score < 1.0


def test_compute_metrics_relevance():
    judgments = {
        "doc1": {"relevance_grade": 2, "binary_relevance": 1},
        "doc2": {"relevance_grade": 1, "binary_relevance": 1},
        "doc3": {"relevance_grade": 0, "binary_relevance": 0},
    }
    ranked = ["doc1", "doc3", "doc2"]
    m = compute_metrics(ranked, judgments)
    assert m["mrr"] == 1.0
    assert m["precision@1"] == 1.0
    assert m["precision@3"] == 2.0 / 3.0
    assert m["recall@1"] == 0.5
    assert m["recall@3"] == 1.0


def test_bm25_lexical_scoring():
    corpus = {
        "d1": "Fire detected in building corridor",
        "d2": "Smoke plume rising near factory",
        "d3": "No weapon detected in footage"
    }
    bm25 = SimpleBM25(corpus)
    scores = bm25.score("Active fire")
    assert scores["d1"] > scores["d2"]
    assert scores["d1"] > scores["d3"]


def test_eval_set_and_reports_exist_and_conform():
    eval_set_path = BASE_DIR / "outputs" / "aegis_semantic_eval_set.json"
    quality_report_path = BASE_DIR / "outputs" / "aegis_semantic_quality_report.json"
    comp_report_path = BASE_DIR / "outputs" / "aegis_text_encoder_comparison.json"

    assert eval_set_path.is_file(), "Evaluation dataset missing"
    assert quality_report_path.is_file(), "Gate 1 quality report missing"
    assert comp_report_path.is_file(), "Gate 2 comparison report missing"

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    assert eval_set["status"] == "PROVISIONAL_PENDING_HUMAN_APPROVAL"
    assert len(eval_set["queries"]) == 6
    assert eval_set["candidate_pool"]["total_records"] == 28

    with open(quality_report_path, "r", encoding="utf-8") as f:
        q_rep = json.load(f)
    assert "e5_small_v2_clean" in q_rep["models_evaluated"]
    assert "lexical_bm25_baseline" in q_rep["models_evaluated"]
    assert q_rep["models_evaluated"]["e5_small_v2_clean"]["ndcg@3"] > 0.70

    with open(comp_report_path, "r", encoding="utf-8") as f:
        c_rep = json.load(f)
    assert c_rep["final_encoder_set"] is False
    assert c_rep["manifest_state_unchanged"] is True
    assert "multilingual-e5-small" in c_rep["candidates"]
    assert "paraphrase-multilingual-MiniLM-L12-v2" in c_rep["candidates"]
