"""Explainability helpers for existing AEGIS outputs.

The package explains saved predictions and source evidence; it does not retrain or
change any detector, NLP pipeline, or threat-fusion score.
"""
from .video_explainer import explain_video_report
from .nlp_explainer import explain_nlp_output
from .fusion_explainer import explain_fusion_score
from .osint_explainer import explain_osint_result
from .report import build_xai_report

__all__ = [
    "explain_video_report",
    "explain_nlp_output",
    "explain_fusion_score",
    "explain_osint_result",
    "build_xai_report",
]
