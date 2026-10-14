"""Guard: the approved Phase 2 files must stay byte-identical (Phase 3 work is additive only).

`phase2_baseline.sha256` lists the 44 files of the approved closeout archive
(SHA-256 of representation_learning_phase2_closeout.zip = 2ecdbef3...5abc). Paths are relative to
the directory that CONTAINS the `representation_learning` package.

One deliberate exception: `models.manifest.json` is *expected* to change when the owner later runs
the Phase 2 `record` command for the visual weights. It is therefore checked structurally (same
models, sources, paths, dims) unless it still equals the baseline hash. If any other approved file
must change, re-baseline deliberately and report it; do not edit this test to hide a change."""

import json
from pathlib import Path

from representation_learning.encoders.integrity import sha256_file

PKG = Path(__file__).resolve().parents[1]
BASELINE = PKG / "tests" / "phase2_baseline.sha256"
MUTABLE = "representation_learning/models.manifest.json"
TEMPLATE = {
    "dinov2-small": dict(family="DINOv2", variant="small", hf_id="facebook/dinov2-small",
                         source_url="https://huggingface.co/facebook/dinov2-small",
                         local_dir="models/dinov2-small", expected_dim=384),
    "clip-vit-b-32": dict(family="CLIP", variant="vit-b-32", hf_id="openai/clip-vit-base-patch32",
                          source_url="https://huggingface.co/openai/clip-vit-base-patch32",
                          local_dir="models/clip-vit-base-patch32", expected_dim=512),
}


def _baseline():
    rows = {}
    for line in BASELINE.read_text().splitlines():
        h, _, p = line.partition("  ")
        rows[p.strip()] = h.strip()
    return rows


def test_baseline_lists_the_44_approved_files():
    rows = _baseline()
    assert len(rows) == 44 and MUTABLE in rows
    assert "representation_learning/encoders/manifest.py" in rows
    assert not any("text" in p.split("/")[1:2] for p in rows)       # no Phase 3 file in the baseline


def test_approved_phase2_files_are_byte_identical():
    changed = []
    for rel, want in _baseline().items():
        if rel == MUTABLE:
            continue
        fp = PKG.parent / rel
        if not fp.is_file() or sha256_file(fp) != want:
            changed.append(rel)
    assert changed == [], f"approved Phase 2 files changed or missing: {changed}"


def test_visual_manifest_unchanged_or_only_recorded_fields_changed():
    fp = PKG.parent / MUTABLE
    if sha256_file(fp) == _baseline()[MUTABLE]:
        return
    m = json.loads(fp.read_text())["models"]
    assert set(m) == set(TEMPLATE)
    for name, want in TEMPLATE.items():
        assert {k: m[name][k] for k in want} == want
