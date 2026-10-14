"""The initial CPU baseline is DINOv2-small + CLIP ViT-B/32 (resource-based, not a quality
ranking). These tests keep code defaults, CLI defaults, the manifest and the docs aligned.
DINOv2-base stays a selectable, optional variant."""

import json
import re
from pathlib import Path

from representation_learning.encoders import ClipEncoder, Dinov2Encoder
from representation_learning.encoders.manifest import DEFAULT_MANIFEST
from representation_learning.evaluation.run_eval import _parser

ROOT = Path(__file__).resolve().parents[1]


def test_code_defaults():
    assert Dinov2Encoder().spec.name == "dinov2-small" and Dinov2Encoder().spec.expected_dim == 384
    assert ClipEncoder().spec.name == "clip-vit-b-32" and ClipEncoder().spec.expected_dim == 512
    assert Dinov2Encoder("base").spec.expected_dim == 768         # optional variant still works


def test_cli_defaults():
    a = _parser().parse_args([])
    assert a.dinov2_variant == "small" and a.clip_variant == "vit-b-32"


def test_manifest_slots_match_defaults():
    m = json.loads(DEFAULT_MANIFEST.read_text())["models"]
    assert set(m) == {"dinov2-small", "clip-vit-b-32"}              # no base slot by default
    for name, enc in (("dinov2-small", Dinov2Encoder()), ("clip-vit-b-32", ClipEncoder())):
        assert m[name]["expected_dim"] == enc.spec.expected_dim
        assert m[name]["hf_id"] in enc.spec.source_ref
        assert m[name]["variant"] == (enc.variant)
        assert enc.spec.name == name


def test_docs_use_the_same_variants_and_paths():
    m = json.loads(DEFAULT_MANIFEST.read_text())["models"]
    wm = (ROOT / "docs" / "weights_and_models.md").read_text()
    for e in m.values():
        assert "representation_learning/" + e["local_dir"] in wm
        assert e["hf_id"] in wm
    p2 = (ROOT / "docs" / "phase2_encoders.md").read_text()
    row = [l for l in p2.splitlines() if l.startswith("| Default checkpoint")][0]
    assert "dinov2-small" in row and "dinov2-base" not in row
    assert "**384**" in [l for l in p2.splitlines() if l.startswith("| Output dimension")][0]


def test_no_doc_names_base_as_default():
    pat = re.compile(r"(default[^\n]{0,80}dinov2[- ]base|dinov2[- ]base[^\n]{0,40}\(?default)", re.I)
    for f in list((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]:
        assert not pat.search(f.read_text()), f
