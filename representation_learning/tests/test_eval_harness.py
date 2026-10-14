"""Harness tests: mock encoders only (no weights, no network)."""

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from representation_learning.encoders.errors import InvalidInputError, UnsupportedEvidenceError
from representation_learning.evaluation import inputs as inp
from representation_learning.evaluation.measure import measure_encoder
from representation_learning.evaluation.run_eval import NOT_EVALUATED, _parser
from representation_learning.testing.mocks import mock_encoder

REPO = Path(__file__).resolve().parents[2]
VIDEO = REPO / "input" / "YOUTUBE_YouTubeCCTV196_gun_2.mp4"
REPORT = REPO / "output" / "YOUTUBE_YouTubeCCTV196_gun_2_report.json"


def test_synthetic_inputs_deterministic_and_varied():
    a, b = inp.synthetic_inputs(12, 0), inp.synthetic_inputs(12, 0)
    assert [x.name for x in a] == [x.name for x in b]
    for x, y in zip(a, b):
        xa, ya = np.asarray(x.image), np.asarray(y.image)
        assert np.array_equal(xa, ya)
    assert any(x.name.endswith("gray") for x in a) and any(x.name.endswith("rgba") for x in a)
    assert all(x.kind == "synthetic" for x in a)


@pytest.mark.parametrize("family", ["dinov2", "clip"])
def test_measure_with_mock_encoder(family):
    enc = mock_encoder(family)
    res = measure_encoder(enc, inp.synthetic_inputs(8), repeats=2, batch_sizes=(1, 3))
    assert res["load"]["ok"] and res["load"]["mock"] is True
    assert res["extraction"]["n_ok"] == 8 and res["extraction"]["n_failed"] == 0
    v = res["validity"]
    assert v["all_finite"] and v["dim_matches_spec"] and v["max_abs_norm_deviation"] < 1e-5
    assert res["repeatability"]["bitwise_identical"] is True
    assert set(res["latency_per_image_by_batch_size"]) == {"1", "3"}
    assert "NOT_A_QUALITY_METRIC" in "".join(res)
    json.dumps(res)


def test_measure_records_failures_not_crashes():
    items = inp.synthetic_inputs(3) + [inp.EvalInput("bad", None, None,
                                                     extra={"load_error": "boom"})]
    res = measure_encoder(mock_encoder("dinov2"), items, repeats=1, batch_sizes=(1,))
    assert res["extraction"]["n_failed"] == 1 and res["extraction"]["failures"][0]["input"] == "bad"


def test_measure_load_failure_is_reported():
    from representation_learning.encoders import Dinov2Encoder
    res = measure_encoder(Dinov2Encoder(), inp.synthetic_inputs(2))   # default = small, no weights
    assert res["load"]["ok"] is False and res["load"]["error_type"] == "EncoderLoadError"


def test_statement_disclaims_quality_and_ranking():
    s = NOT_EVALUATED.lower()
    assert "not" in s and "no encoder ranking" in s and "labelled" in s


def test_parser_defaults_cpu_and_no_network_flags():
    a = _parser().parse_args([])
    assert a.device == "cpu" and a.source == "synthetic" and not a.tiny_random


@pytest.mark.skipif(not (VIDEO.exists() and REPORT.exists()), reason="committed video/report absent")
def test_video_report_inputs_are_phase1_valid_and_exclude_person(schemas):
    items = inp.video_report_inputs(VIDEO, REPORT, max_items=6)
    assert items and all(it.kind == "crop" for it in items)      # frames off by default
    diag = inp.video_report_inputs(VIDEO, REPORT, max_items=3, include_frames=True)
    assert any(it.kind == "frame" for it in diag)
    v = Draft202012Validator(schemas["visual_evidence_item.schema.json"])
    assert items
    for it in items:
        assert [e.message for e in v.iter_errors(it.evidence)] == []
        assert it.color_order == "BGR"
        assert it.evidence["video_uri"].endswith(".mp4") and "_analyzed" not in it.evidence["video_uri"]
        if it.kind == "crop":
            assert it.evidence["detector"]["category"] in ("weapon", "fire", "smoke")
    # and they are accepted by the encoder (mock) end-to-end
    enc = mock_encoder("clip")
    for it in items:
        r = enc.encode(it.image, color_order=it.color_order, evidence=it.evidence)
        assert r.record["evidence_id"] == it.evidence["evidence_id"]


@pytest.mark.skipif(not (VIDEO.exists() and REPORT.exists()), reason="committed video/report absent")
def test_video_inputs_reject_annotated_and_mismatched(tmp_path):
    ann = tmp_path / "x_analyzed.mp4"
    ann.write_bytes(b"")
    with pytest.raises(UnsupportedEvidenceError):
        inp.video_report_inputs(ann, REPORT)
    other = tmp_path / "other.mp4"
    other.write_bytes(b"")
    with pytest.raises(InvalidInputError, match="report is for"):
        inp.video_report_inputs(other, REPORT)


def test_directory_inputs(tmp_path):
    from PIL import Image
    Image.new("RGB", (40, 30), (1, 2, 3)).save(tmp_path / "a.png")
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    items = inp.directory_inputs(tmp_path)
    assert [i.name for i in items] == ["a.png", "broken.jpg"]
    assert items[1].image is None and "load_error" in items[1].extra
    with pytest.raises(InvalidInputError):
        inp.directory_inputs(tmp_path / "missing")


@pytest.mark.skipif(not (VIDEO.exists() and REPORT.exists()), reason="committed video/report absent")
def test_diagnostic_frames_are_separate_and_excluded_from_target_stats():
    from representation_learning.evaluation.measure import measure_diagnostic_frames
    items = inp.video_report_inputs(VIDEO, REPORT, max_items=3, include_frames=True)
    frames = [i for i in items if i.kind == "frame"]
    crops = [i for i in items if i.kind != "frame"]
    enc = mock_encoder("clip")
    d = measure_diagnostic_frames(enc, frames)
    assert d["n_ok"] == len(frames) and "NOT target-object" in d["purpose"]
    res = measure_encoder(enc, crops, repeats=1, batch_sizes=(1,))
    assert res["extraction"]["n_inputs"] == len(crops)
    # frames are refused if they reach the normal path
    assert measure_encoder(enc, frames, repeats=1, batch_sizes=(1,))["extraction"]["n_failed"] == len(frames)
