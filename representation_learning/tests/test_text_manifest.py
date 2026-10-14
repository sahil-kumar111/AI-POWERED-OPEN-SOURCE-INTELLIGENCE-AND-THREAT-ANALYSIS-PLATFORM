"""Phase 3 / G1: text-model manifest tests.

All offline. Local "checkpoints" are tiny SYNTHETIC files (valid safetensors container, JSON
configs); no model weights are downloaded or loaded, and nothing here says anything about real
checkpoints. The committed manifest is only ever read; mutated copies live in tmp_path."""

import copy
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from representation_learning.encoders import manifest as visual_manifest
from representation_learning.encoders.errors import EncoderIntegrityError, EncoderLoadError
from representation_learning.encoders.integrity import sha256_file
from representation_learning.text import manifest as T

ROOT = Path(__file__).resolve().parents[1]
E5, MINI = "multilingual-e5-small", "paraphrase-multilingual-minilm-l12-v2"
POOL, SEQ = "1_Pooling/config.json", "sentence_bert_config.json"      # semantic-evidence files

# Verified G0 values, typed independently of the manifest (trip-wire against drift/tampering).
G0 = {
    E5: dict(
        hf_id="intfloat/multilingual-e5-small", rev="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        lic="mit",
        files={"model.safetensors": (470641600, "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"),
               "tokenizer.json": (17082730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
               "config.json": (655, None), "tokenizer_config.json": (443, None),
               "special_tokens_map.json": (167, None), "1_Pooling/config.json": (200, None),
               "modules.json": (387, None), "sentence_bert_config.json": (57, None)}),
    MINI: dict(
        hf_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        rev="e8f8c211226b894fcb81acc59f3b34ba3efd5f42", lic="apache-2.0",
        files={"model.safetensors": (470641600, "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b"),
               "tokenizer.json": (9081518, "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"),
               "config.json": (645, None), "tokenizer_config.json": (526, None),
               "special_tokens_map.json": (239, None), "1_Pooling/config.json": (190, None),
               "modules.json": (229, None), "sentence_bert_config.json": (53, None),
               "config_sentence_transformers.json": (122, None)}),
}


def committed():
    return json.loads(T.DEFAULT_MANIFEST.read_text())


# ------------------------------------------------------------------ synthetic local checkpoint
def _safetensors_bytes(dim=384, rows=4, name="embeddings.word_embeddings.weight"):
    nbytes = rows * dim * 4
    hdr = json.dumps({name: {"dtype": "F32", "shape": [rows, dim], "data_offsets": [0, nbytes]}}).encode()
    return struct.pack("<Q", len(hdr)) + hdr + b"\x00" * nbytes


def _write(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())


def make_env(tmp_path, name):
    """Copy of the committed manifest whose sizes/upstream hashes match synthetic local files."""
    m = committed()
    e = m["models"][name]
    d = tmp_path / e["local_dir"]
    seq = e["encoder"]["sentence_bert_max_seq_length"] or 512
    pooling = {"word_embedding_dimension": 384, "pooling_mode_cls_token": False,
               "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False,
               "pooling_mode_mean_sqrt_len_tokens": False}
    content = {
        "model.safetensors": _safetensors_bytes(),
        "config.json": {"model_type": "bert", "hidden_size": 384, "max_position_embeddings": 512,
                        "vocab_size": 4},
        "tokenizer.json": {"version": "1.0", "model": {"type": "Unigram", "vocab": []}},
        "tokenizer_config.json": {"tokenizer_class": "synthetic"},
        "special_tokens_map.json": {"pad_token": "<pad>"},
        "1_Pooling/config.json": pooling,
        "modules.json": [{"idx": 0}],
        "sentence_bert_config.json": {"max_seq_length": seq},
        "config_sentence_transformers.json": {"synthetic": True},
    }
    for fname, f in e["files"].items():
        _write(d / fname, content[fname])
        f["size"] = (d / fname).stat().st_size
        if f["upstream_sha256"] is not None:
            f["upstream_sha256"] = sha256_file(d / fname)
    mp = tmp_path / "text_models.manifest.json"
    mp.write_text(json.dumps(m, indent=2) + "\n")
    return mp, d, e


def rec(mp, name, **over):
    e = T.load_manifest(mp)["models"][name]
    kw = dict(revision=e["expected_revision"], license_id=e["expected_license_id"],
              license_text="Licence text copied from the model card (synthetic test)")
    kw.update(over)
    return T.record_entry(mp, name, kw["revision"], kw["license_id"], kw["license_text"])


# ------------------------------------------------------------------ committed manifest
def test_committed_manifest_is_valid_candidates_only():
    m = committed()
    assert T.manifest_problems(m) == []
    assert m["kind"] == "text" and set(m["models"]) == {E5, MINI}
    for e in m["models"].values():
        assert e["status"] == "candidate" and e["final_encoder"] is False


@pytest.mark.parametrize("name", [E5, MINI])
def test_committed_manifest_matches_verified_g0_metadata(name):
    e, g = committed()["models"][name], G0[name]
    assert e["hf_id"] == g["hf_id"] and e["source_url"] == "https://huggingface.co/" + g["hf_id"]
    assert e["expected_revision"] == g["rev"] and e["expected_license_id"] == g["lic"]
    assert {f: (v["size"], v["upstream_sha256"]) for f, v in e["files"].items()} == g["files"]


def test_no_invented_hashes_only_lfs_files_have_upstream_sha256():
    for e in committed()["models"].values():
        with_hash = {f for f, v in e["files"].items() if v["upstream_sha256"] is not None}
        assert with_hash == {"model.safetensors", "tokenizer.json"}


@pytest.mark.parametrize("name", [E5, MINI])
def test_committed_manifest_is_unpinned_nothing_recorded(name):
    e = committed()["models"][name]
    assert e["revision"] == e["license_id"] == e["license"] == "TO_RECORD"
    assert e["observed_config"] is None and e["recorded_on"] is None
    assert all(f["sha256"] == "TO_RECORD" for f in e["files"].values())
    assert T.pin_problems(e)


def test_design_decisions_are_encoded():
    m = committed()["models"]
    e5, mini = m[E5]["encoder"], m[MINI]["encoder"]
    assert mini["max_input_tokens"] == 128 and mini["sentence_bert_max_seq_length"] == 128
    assert mini["input_template"] == {"role_required": False, "roles": None}
    assert e5["input_template"] == {"role_required": True,
                                    "roles": {"query": "query: ", "passage": "passage: "}}
    assert e5["max_position_embeddings"] == 512 and e5["max_input_tokens"] is None
    assert T.design_open_items(m[E5]) == ["max_input_tokens"] and T.design_open_items(m[MINI]) == []
    for enc in (e5, mini):
        assert enc["expected_dim"] == 384 and enc["pooling"] == "mean"
        assert enc["output_normalization"] == "l2" and enc["token_limit_policy"] == "reject_over_limit"
    assert e5["upstream_normalize_module"] is True


def test_visual_manifest_is_separate_and_unaffected():
    vm = visual_manifest.load_manifest()
    assert set(vm["models"]) == {"dinov2-small", "clip-vit-b-32"}
    assert not set(vm["models"]) & set(committed()["models"])
    with pytest.raises(EncoderLoadError, match="unknown manifest entry"):
        visual_manifest.encoder_from_manifest(E5)
    with pytest.raises(EncoderLoadError, match="kind"):
        T.validate_manifest(vm)                      # a visual manifest is not a text manifest


# ------------------------------------------------------------------ invalid manifests
def _e(m, name=E5):
    return m["models"][name]


BAD = [
    ("short-revision", lambda m: _e(m).update(expected_revision="614241f6"), "40-character"),
    ("uppercase-revision", lambda m: _e(m).update(expected_revision=G0[E5]["rev"].upper()), "40-character"),
    ("branch-name-revision", lambda m: _e(m).update(expected_revision="main"), "40-character"),
    ("39-hex-revision", lambda m: _e(m).update(expected_revision=G0[E5]["rev"][:-1]), "40-character"),
    ("recorded-revision-bad", lambda m: _e(m).update(revision="abc"), "revision must be"),
    ("final-true", lambda m: _e(m).update(final_encoder=True), "final_encoder"),
    ("final-string", lambda m: _e(m).update(final_encoder="false"), "final_encoder"),
    ("final-missing", lambda m: _e(m).pop("final_encoder"), "final_encoder"),
    ("status-selected", lambda m: _e(m).update(status="selected"), "status"),
    ("unknown-entry-key", lambda m: _e(m).update(winner=True), "unknown keys"),
    ("language-allow-list", lambda m: _e(m).update(language_allow_list=["en"]), "allow-list"),
    ("language-allow-list-in-encoder", lambda m: _e(m)["encoder"].update(supported_languages=["en"]), "allow-list"),
    ("required-file-removed", lambda m: _e(m)["files"].pop("tokenizer.json"), "required file"),
    ("weights-removed", lambda m: _e(m)["files"].pop("model.safetensors"), "required file"),
    ("pooling-evidence-removed-e5", lambda m: _e(m)["files"].pop(POOL), "1_Pooling/config.json"),
    ("pooling-evidence-removed-minilm", lambda m: _e(m, MINI)["files"].pop(POOL), "1_Pooling/config.json"),
    ("seq-len-evidence-removed-minilm", lambda m: _e(m, MINI)["files"].pop(SEQ), "sentence_bert_config.json"),
    ("path-traversal", lambda m: _e(m)["files"].update({"../evil.json": copy.deepcopy(_e(m)["files"]["config.json"])}), "unsafe"),
    ("absolute-path", lambda m: _e(m)["files"].update({"/etc/passwd": copy.deepcopy(_e(m)["files"]["config.json"])}), "unsafe"),
    ("pickle-weights-listed", lambda m: _e(m)["files"].update({"pytorch_model.bin": copy.deepcopy(_e(m)["files"]["config.json"])}), "not allowed"),
    ("second-safetensors-listed", lambda m: _e(m)["files"].update({"other.safetensors": copy.deepcopy(_e(m)["files"]["config.json"])}), "not allowed"),
    ("bad-upstream-hash", lambda m: _e(m)["files"]["model.safetensors"].update(upstream_sha256="zz" * 32), "upstream_sha256"),
    ("weights-no-upstream", lambda m: _e(m)["files"]["model.safetensors"].update(upstream_sha256=None), "provenance"),
    ("bad-recorded-hash", lambda m: _e(m)["files"]["config.json"].update(sha256="1234"), "sha256 must be"),
    ("zero-size", lambda m: _e(m)["files"]["config.json"].update(size=0), "size"),
    ("bool-size", lambda m: _e(m)["files"]["config.json"].update(size=True), "size"),
    ("file-extra-key", lambda m: _e(m)["files"]["config.json"].update(note="x"), "exactly keys"),
    ("weights-role-on-config", lambda m: _e(m)["files"]["config.json"].update(role="weights"), "role"),
    ("local-dir-escape", lambda m: _e(m).update(local_dir="models/../../x"), "local_dir"),
    ("local-dir-absolute", lambda m: _e(m).update(local_dir="/models/x"), "local_dir"),
    ("local-dir-not-models", lambda m: _e(m).update(local_dir="elsewhere/x"), "local_dir"),
    ("duplicate-local-dir", lambda m: _e(m, MINI).update(local_dir=_e(m)["local_dir"]), "duplicates"),
    ("source-url-mismatch", lambda m: _e(m).update(source_url="https://huggingface.co/other/model"), "source_url"),
    ("pooling-not-mean", lambda m: _e(m)["encoder"].update(pooling="cls"), "pooling"),
    ("truncation-allowed", lambda m: _e(m)["encoder"].update(token_limit_policy="truncate"), "truncation"),
    ("not-l2", lambda m: _e(m)["encoder"].update(output_normalization="none"), "l2"),
    ("limit-above-positions", lambda m: _e(m)["encoder"].update(max_input_tokens=1024), "exceeds"),
    ("e5-role-prefix-missing", lambda m: _e(m)["encoder"]["input_template"]["roles"].pop("passage"), "prefixes"),
    ("minilm-roles-set", lambda m: _e(m, MINI)["encoder"]["input_template"].update(roles={"query": "q"}), "roles"),
    ("dim-not-int", lambda m: _e(m)["encoder"].update(expected_dim="384"), "expected_dim"),
    ("wrong-kind", lambda m: m.update(kind="visual"), "kind"),
    ("wrong-version", lambda m: m.update(manifest_version="9"), "manifest_version"),
    ("extra-top-key", lambda m: m.update(winner=E5), "top-level"),
    ("recorded-on-bad", lambda m: _e(m).update(recorded_on="yesterday"), "recorded_on"),
]


@pytest.mark.parametrize("mutate,expect", [(b[1], b[2]) for b in BAD], ids=[b[0] for b in BAD])
def test_invalid_manifest_is_rejected(mutate, expect):
    m = committed()
    mutate(m)
    probs = T.manifest_problems(m)
    assert probs and any(expect in p for p in probs), probs
    with pytest.raises(EncoderLoadError, match="invalid text manifest"):
        T.validate_manifest(m)


def test_load_manifest_rejects_unreadable_and_invalid_files(tmp_path):
    with pytest.raises(EncoderLoadError, match="cannot read"):
        T.load_manifest(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{broken")
    with pytest.raises(EncoderLoadError, match="cannot read"):
        T.load_manifest(tmp_path / "bad.json")
    m = committed()
    m["models"][E5]["final_encoder"] = True
    (tmp_path / "t.json").write_text(json.dumps(m))
    with pytest.raises(EncoderLoadError, match="final_encoder"):
        T.load_manifest(tmp_path / "t.json")


# ------------------------------------------------------------------ G1.2: semantic-evidence files must be listed
def _without(name, rel, **encoder_over):
    m = committed()
    m["models"][name]["encoder"].update(encoder_over)
    m["models"][name]["files"].pop(rel, None)
    return m


def _not_listed(probs, rel):
    return any(rel in p and "not listed" in p for p in probs)


@pytest.mark.parametrize("name", [E5, MINI])
def test_pooling_evidence_file_is_required_for_every_entry(name):
    assert T.manifest_problems(committed()) == []
    assert _not_listed(T.manifest_problems(_without(name, POOL)), POOL)


def test_sequence_limit_evidence_file_is_required_only_when_the_limit_is_set():
    # MiniLM: limit set (128) -> the file read by the semantic check must be listed
    assert committed()["models"][MINI]["encoder"]["sentence_bert_max_seq_length"] == 128
    assert _not_listed(T.manifest_problems(_without(MINI, SEQ)), SEQ)
    # E5: limit null -> the file is not read, so its absence is structurally fine ...
    assert committed()["models"][E5]["encoder"]["sentence_bert_max_seq_length"] is None
    assert T.manifest_problems(_without(E5, SEQ)) == []
    # ... and the condition follows the field, not the model name:
    assert _not_listed(T.manifest_problems(_without(E5, SEQ, sentence_bert_max_seq_length=512)), SEQ)
    assert T.manifest_problems(_without(MINI, SEQ, sentence_bert_max_seq_length=None)) == []
    only_set = committed()
    only_set["models"][E5]["encoder"]["sentence_bert_max_seq_length"] = 512      # file is listed -> fine
    assert T.manifest_problems(only_set) == []


@pytest.mark.parametrize("name,rel", [(E5, POOL), (MINI, POOL), (MINI, SEQ)])
def test_missing_evidence_entry_is_rejected_before_record_and_verify(tmp_path, name, rel):
    mp, d, _ = make_env(tmp_path, name)
    m = json.loads(mp.read_text())
    m["models"][name]["files"].pop(rel)
    mp.write_text(json.dumps(m, indent=2) + "\n")
    before = mp.read_bytes()
    with pytest.raises(EncoderLoadError, match="invalid text manifest.*not listed"):
        T.record_entry(mp, name, G0[name]["rev"], G0[name]["lic"], "Licence text (synthetic test)")
    assert mp.read_bytes() == before                                   # nothing was written
    with pytest.raises(EncoderLoadError, match="invalid text manifest.*not listed"):
        T.load_manifest(mp)
    assert T._main(["--manifest", str(mp), "status"]) == 2
    # verification of an already-recorded manifest whose evidence entry is later removed:
    mp2, _, _ = make_env(tmp_path / "second", name)
    rec(mp2, name)
    m2 = json.loads(mp2.read_text())
    m2["models"][name]["files"].pop(rel)
    mp2.write_text(json.dumps(m2, indent=2) + "\n")
    with pytest.raises(EncoderLoadError, match="invalid text manifest.*not listed"):
        T.verify_entry(m2, name, mp2)                                  # rejected structurally,
    assert T._main(["--manifest", str(mp2), "verify", name]) == 2      # not as a hash mismatch


@pytest.mark.parametrize("name,rel", [(E5, POOL), (E5, SEQ), (MINI, POOL), (MINI, SEQ)])
def test_listed_evidence_files_get_the_full_file_checks(tmp_path, name, rel):
    """Missing / resized / symlinked evidence files are refused by record; once recorded, a
    same-size content change or a deletion is caught by verify (SHA-256 / existence)."""
    def fresh(tag):
        mp, d, _ = make_env(tmp_path / tag, name)
        return mp, d

    mp, d = fresh("missing")
    (d / rel).unlink()
    with pytest.raises(EncoderLoadError, match="missing"):
        rec(mp, name)
    mp, d = fresh("size")
    (d / rel).write_bytes(b"{}")
    with pytest.raises(EncoderIntegrityError, match="size mismatch"):
        rec(mp, name)
    mp, d = fresh("link")
    _symlink(d / rel)
    with pytest.raises(EncoderLoadError, match="regular file"):
        rec(mp, name)
    mp, d = fresh("tamper")
    rec(mp, name)
    b = bytearray((d / rel).read_bytes())
    b[len(b) // 2] = ord("0") if b[len(b) // 2] != ord("0") else ord("1")
    (d / rel).write_bytes(bytes(b))
    with pytest.raises(EncoderIntegrityError, match="SHA-256 mismatch"):
        T.verify_entry(T.load_manifest(mp), name, mp)
    mp, d = fresh("deleted")
    rec(mp, name)
    (d / rel).unlink()
    with pytest.raises(EncoderIntegrityError, match="missing"):
        T.verify_entry(T.load_manifest(mp), name, mp)


# ------------------------------------------------------------------ record + verify
@pytest.mark.parametrize("name", [E5, MINI])
def test_record_then_verify_roundtrip(tmp_path, name):
    mp, d, e0 = make_env(tmp_path, name)
    other = MINI if name == E5 else E5
    other_before = json.dumps(T.load_manifest(mp)["models"][other], sort_keys=True)
    visual_before = visual_manifest.DEFAULT_MANIFEST.read_bytes()
    e = rec(mp, name)
    assert e["revision"] == G0[name]["rev"] and e["license_id"] == G0[name]["lic"]
    assert all(len(f["sha256"]) == 64 for f in e["files"].values())
    assert e["observed_config"]["hidden_size"] == 384
    assert e["observed_config"]["safetensors_word_embedding_shape"] == [4, 384]
    assert e["status"] == "candidate" and e["final_encoder"] is False
    m = T.load_manifest(mp)
    assert T.pin_problems(m["models"][name]) == []
    assert T.verify_entry(m, name, mp) == d.resolve()
    assert json.dumps(m["models"][other], sort_keys=True) == other_before     # other entry untouched
    assert visual_manifest.DEFAULT_MANIFEST.read_bytes() == visual_before     # Phase 2 manifest untouched


def test_record_is_idempotent_and_refuses_to_replace_a_pin(tmp_path):
    mp, d, _ = make_env(tmp_path, E5)
    first = rec(mp, E5)
    again = rec(mp, E5)
    assert again["recorded_on"] == first["recorded_on"] and again["files"] == first["files"]
    _write(d / "special_tokens_map.json", {"pad_token": "<pat>"})    # same size, different content
    with pytest.raises(EncoderIntegrityError, match="already recorded"):
        rec(mp, E5)


REC_FAIL = [
    ("wrong-revision", dict(revision="0123456789abcdef0123456789abcdef01234567"), "expected_revision"),
    ("short-revision", dict(revision="614241f6"), "40-character"),
    ("uppercase-revision", dict(revision=G0[E5]["rev"].upper()), "40-character"),
    ("wrong-license-id", dict(license_id="gpl-3.0"), "licence id"),
    ("empty-license-text", dict(license_text="  "), "licence text"),
    ("placeholder-license-text", dict(license_text="TO_RECORD"), "licence text"),
]


@pytest.mark.parametrize("over,expect", [(r[1], r[2]) for r in REC_FAIL], ids=[r[0] for r in REC_FAIL])
def test_record_rejects_bad_arguments_and_leaves_manifest_untouched(tmp_path, over, expect):
    mp, _, _ = make_env(tmp_path, E5)
    before = mp.read_bytes()
    with pytest.raises(EncoderIntegrityError, match=expect):
        rec(mp, E5, **over)
    assert mp.read_bytes() == before


def _mut_file(rel, fn):
    def go(d):
        p = d / rel
        fn(p)
    return go


LOCAL_FAIL = [
    ("missing-weights", lambda d: (d / "model.safetensors").unlink(), EncoderLoadError, "missing"),
    ("missing-tokenizer", lambda d: (d / "tokenizer.json").unlink(), EncoderLoadError, "missing"),
    ("missing-pooling-evidence", lambda d: (d / "1_Pooling/config.json").unlink(), EncoderLoadError, "missing"),
    ("size-mismatch", lambda d: (d / "config.json").write_bytes(b"{}"), EncoderIntegrityError, "size mismatch"),
    ("weights-hash-vs-upstream", lambda d: _flip_byte(d / "model.safetensors"), EncoderIntegrityError, "upstream_sha256"),
    ("tokenizer-hash-vs-upstream", lambda d: _flip_byte(d / "tokenizer.json"), EncoderIntegrityError, "upstream_sha256"),
    ("stray-pytorch-bin", lambda d: (d / "pytorch_model.bin").write_bytes(b"x"), EncoderIntegrityError, "extra weight"),
    ("stray-onnx-subdir", lambda d: _write(d / "onnx/model.onnx", b"x"), EncoderIntegrityError, "extra weight"),
    ("stray-second-safetensors", lambda d: (d / "extra.safetensors").write_bytes(b"x"), EncoderIntegrityError, "extra weight"),
    ("sharded-index", lambda d: (d / "model.safetensors.index.json").write_text("{}"), EncoderIntegrityError, "extra weight"),
    ("symlinked-config", lambda d: _symlink(d / "config.json"), EncoderLoadError, "regular file"),
]


def _flip_byte(p):
    b = bytearray(p.read_bytes())
    b[-1] ^= 0xFF
    p.write_bytes(bytes(b))


def _symlink(p):
    target = p.parent / "target.json"
    target.write_bytes(p.read_bytes())
    p.unlink()
    os.symlink(target, p)


@pytest.mark.parametrize("mutate,exc,expect", [(r[1], r[2], r[3]) for r in LOCAL_FAIL],
                         ids=[r[0] for r in LOCAL_FAIL])
def test_record_rejects_bad_local_files(tmp_path, mutate, exc, expect):
    mp, d, _ = make_env(tmp_path, E5)
    mutate(d)
    before = mp.read_bytes()
    with pytest.raises(exc, match=expect):
        rec(mp, E5)
    assert mp.read_bytes() == before


def test_record_rejects_missing_local_directory(tmp_path):
    mp, d, _ = make_env(tmp_path, E5)
    import shutil
    shutil.rmtree(d)
    with pytest.raises(EncoderLoadError, match="not found"):
        rec(mp, E5)


def _rewrite_with_resize(mp, d, rel, data):
    """Replace a synthetic file and make the manifest size/upstream match it, so only the
    semantic check (not size/upstream) can object."""
    _write(d / rel, data)
    m = json.loads(mp.read_text())
    for e in m["models"].values():
        if rel in e["files"] and (d.name == Path(e["local_dir"]).name):
            e["files"][rel]["size"] = (d / rel).stat().st_size
            if e["files"][rel]["upstream_sha256"] is not None:
                e["files"][rel]["upstream_sha256"] = sha256_file(d / rel)
    mp.write_text(json.dumps(m, indent=2) + "\n")


SEM_FAIL = [
    ("hidden-size", "config.json", {"model_type": "bert", "hidden_size": 768, "max_position_embeddings": 512}, "hidden_size"),
    ("hidden-size-missing", "config.json", {"model_type": "bert"}, "hidden_size"),
    ("max-positions", "config.json", {"model_type": "bert", "hidden_size": 384, "max_position_embeddings": 128}, "max_position_embeddings"),
    ("config-not-object", "config.json", [1, 2], "JSON object"),
    ("pooling-cls", "1_Pooling/config.json", {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False}, "mean pooling"),
    ("pooling-extra-mode", "1_Pooling/config.json", {"pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": True}, "other pooling"),
    ("pooling-dim", "1_Pooling/config.json", {"pooling_mode_mean_tokens": True, "word_embedding_dimension": 768}, "dimension"),
    ("tokenizer-no-model", "tokenizer.json", {"version": "1.0"}, "'model'"),
    ("tokenizer-config-list", "tokenizer_config.json", [1], "JSON object"),
    ("safetensors-too-small", "model.safetensors", b"abc", "too small"),
    ("safetensors-bad-header-len", "model.safetensors", struct.pack("<Q", 10**12) + b"{}", "header length"),
    ("safetensors-not-json", "model.safetensors", struct.pack("<Q", 4) + b"nope", "not valid JSON"),
    ("safetensors-offset-mismatch", "model.safetensors", _safetensors_bytes() + b"\x00" * 8, "offsets"),
    ("safetensors-embedding-width", "model.safetensors", _safetensors_bytes(dim=256), "width"),
    ("safetensors-no-tensors", "model.safetensors", struct.pack("<Q", 2) + b"{}", "no tensors"),
]


@pytest.mark.parametrize("rel,data,expect", [(r[1], r[2], r[3]) for r in SEM_FAIL], ids=[r[0] for r in SEM_FAIL])
def test_record_rejects_semantic_mismatch_with_design_facts(tmp_path, rel, data, expect):
    mp, d, _ = make_env(tmp_path, E5)
    _rewrite_with_resize(mp, d, rel, data)
    before = mp.read_bytes()
    with pytest.raises(EncoderIntegrityError, match=expect):
        rec(mp, E5)
    assert mp.read_bytes() == before


def test_minilm_128_token_evidence_is_enforced(tmp_path):
    mp, d, _ = make_env(tmp_path, MINI)
    _rewrite_with_resize(mp, d, "sentence_bert_config.json", {"max_seq_length": 512})
    with pytest.raises(EncoderIntegrityError, match="max_seq_length"):
        rec(mp, MINI)


def test_real_manifest_refuses_synthetic_files_via_upstream_provenance(tmp_path):
    """The committed manifest carries the REAL upstream sizes/hashes, so synthetic files must fail."""
    import shutil
    mp, d, _ = make_env(tmp_path, E5)
    real = tmp_path / "real" / "text_models.manifest.json"
    real.parent.mkdir()
    shutil.copy(T.DEFAULT_MANIFEST, real)
    shutil.copytree(d, real.parent / "models" / "multilingual-e5-small")
    with pytest.raises(EncoderIntegrityError, match="size mismatch|upstream_sha256"):
        rec(real, E5)


# ------------------------------------------------------------------ verify: tampering
@pytest.fixture
def recorded(tmp_path):
    mp, d, _ = make_env(tmp_path, E5)
    rec(mp, E5)
    return mp, d


def test_verify_detects_weight_tamper(recorded):
    mp, d = recorded
    _flip_byte(d / "model.safetensors")
    with pytest.raises(EncoderIntegrityError, match="SHA-256 mismatch"):
        T.verify_entry(T.load_manifest(mp), E5, mp)


@pytest.mark.parametrize("rel", ["tokenizer.json", "config.json", "1_Pooling/config.json"])
def test_verify_detects_file_tamper_same_size(recorded, rel):
    mp, d = recorded
    b = bytearray((d / rel).read_bytes())
    b[len(b) // 2] = ord("0") if b[len(b) // 2] != ord("0") else ord("1")
    (d / rel).write_bytes(bytes(b))
    with pytest.raises(EncoderIntegrityError, match="SHA-256 mismatch"):
        T.verify_entry(T.load_manifest(mp), E5, mp)


def test_verify_detects_missing_resized_and_stray_files(recorded):
    mp, d = recorded
    (d / "special_tokens_map.json").write_bytes(b"{}")
    with pytest.raises(EncoderIntegrityError, match="size mismatch"):
        T.verify_entry(T.load_manifest(mp), E5, mp)
    (d / "special_tokens_map.json").unlink()
    with pytest.raises(EncoderIntegrityError, match="missing"):
        T.verify_entry(T.load_manifest(mp), E5, mp)


def test_verify_detects_stray_weight_file_after_recording(recorded):
    mp, d = recorded
    (d / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(EncoderIntegrityError, match="extra weight"):
        T.verify_entry(T.load_manifest(mp), E5, mp)


def _edit(mp, fn):
    m = json.loads(mp.read_text())
    fn(m["models"][E5])
    mp.write_text(json.dumps(m, indent=2) + "\n")


def test_verify_detects_manifest_tampering(recorded):
    mp, d = recorded
    original = mp.read_bytes()

    _edit(mp, lambda e: e["files"]["config.json"].update(sha256="0" * 64))
    with pytest.raises(EncoderIntegrityError, match="SHA-256 mismatch"):
        T.verify_entry(T.load_manifest(mp), E5, mp)

    mp.write_bytes(original)
    _edit(mp, lambda e: e.update(revision="0123456789abcdef0123456789abcdef01234567"))
    with pytest.raises(EncoderLoadError, match="differs from expected_revision"):
        T.verify_entry(T.load_manifest(mp), E5, mp)

    mp.write_bytes(original)
    _edit(mp, lambda e: e.update(expected_revision="0123456789abcdef0123456789abcdef01234567"))
    with pytest.raises(EncoderLoadError, match="differs from expected_revision"):
        T.verify_entry(T.load_manifest(mp), E5, mp)

    mp.write_bytes(original)
    _edit(mp, lambda e: e.update(license_id="gpl-3.0"))
    with pytest.raises(EncoderLoadError, match="licence id differs"):
        T.verify_entry(T.load_manifest(mp), E5, mp)

    mp.write_bytes(original)
    _edit(mp, lambda e: e["files"]["model.safetensors"].update(sha256="1" * 64))
    with pytest.raises(EncoderLoadError, match="differs from upstream_sha256"):
        T.verify_entry(T.load_manifest(mp), E5, mp)

    mp.write_bytes(original)
    _edit(mp, lambda e: e.update(final_encoder=True))
    with pytest.raises(EncoderLoadError, match="final_encoder"):
        T.load_manifest(mp)

    mp.write_bytes(original)
    _edit(mp, lambda e: e["encoder"].update(expected_dim=768))
    with pytest.raises(EncoderIntegrityError, match="hidden_size"):
        T.verify_entry(T.load_manifest(mp), E5, mp)

    mp.write_bytes(original)
    _edit(mp, lambda e: e["observed_config"].update(vocab_size=999))
    with pytest.raises(EncoderIntegrityError, match="observed_config"):
        T.verify_entry(T.load_manifest(mp), E5, mp)

    mp.write_bytes(original)
    assert T.verify_entry(T.load_manifest(mp), E5, mp) == d.resolve()      # restored original is fine


def test_unpinned_entry_refuses_verify(tmp_path):
    mp, _, _ = make_env(tmp_path, E5)
    with pytest.raises(EncoderLoadError, match="not pinned"):
        T.verify_entry(T.load_manifest(mp), E5, mp)
    with pytest.raises(EncoderLoadError, match="unknown text manifest entry"):
        T.verify_entry(T.load_manifest(mp), "nope", mp)


# ------------------------------------------------------------------ CLI
def test_cli_status_record_verify_exit_codes(tmp_path, capsys):
    mp, _, _ = make_env(tmp_path, MINI)
    assert T._main(["--manifest", str(mp), "status"]) == 0
    out = capsys.readouterr().out
    assert "NOT PINNED" in out and "candidate" in out and "final_encoder=false" in out
    assert "design open: max_input_tokens" in out                           # e5 entry
    assert T._main(["--manifest", str(mp), "verify"]) == 2                  # unpinned
    assert T._main(["--manifest", str(mp), "record", MINI, "--revision", G0[MINI]["rev"],
                    "--license-id", "mit", "--license", "x"]) == 2          # wrong licence id
    assert T._main(["--manifest", str(mp), "record", MINI, "--revision", G0[MINI]["rev"],
                    "--license-id", "apache-2.0", "--license", "Apache-2.0 (synthetic)"]) == 0
    assert T._main(["--manifest", str(mp), "record", "nope", "--revision", G0[MINI]["rev"],
                    "--license-id", "x", "--license", "x"]) == 2
    assert "recorded" in capsys.readouterr().out
    # E5 is still unpinned, so plain verify fails; verifying the recorded one passes
    assert T._main(["--manifest", str(mp), "verify"]) == 2
    assert T._main(["--manifest", str(mp), "verify", MINI]) == 0
    assert T._main(["--manifest", str(mp), "status"]) == 0
    assert "PINNED" in capsys.readouterr().out


def test_cli_manifest_sha256_tamper_check(tmp_path):
    mp, _, _ = make_env(tmp_path, E5)
    rec(mp, E5)
    good = sha256_file(mp)
    assert T._main(["--manifest", str(mp), "verify", E5, "--manifest-sha256", good]) == 0
    assert T._main(["--manifest", str(mp), "verify", E5, "--manifest-sha256", "0" * 64]) == 2
    assert T._main(["--manifest", str(mp), "verify", E5, "--manifest-sha256", "nothex"]) == 2
    mp.write_text(mp.read_text() + " ")
    assert T._main(["--manifest", str(mp), "verify", E5, "--manifest-sha256", good]) == 2


def test_cli_rejects_invalid_manifest(tmp_path, capsys):
    mp = tmp_path / "m.json"
    m = committed()
    m["models"][E5]["final_encoder"] = True
    mp.write_text(json.dumps(m))
    assert T._main(["--manifest", str(mp), "status"]) == 2
    assert "final_encoder" in capsys.readouterr().err


# ------------------------------------------------------------------ laziness / offline
def test_importing_text_manifest_is_lazy_and_offline_safe():
    # Some environments preload e.g. requests/urllib3 at interpreter start-up (.pth files,
    # sitecustomize). Only modules that importing the text manifest ADDS are attributable to it,
    # so the baseline is captured before the import. The prohibited list is unchanged.
    code = (
        "import sys; sys.path.insert(0, %r);"
        "PROHIBITED=('torch','transformers','tokenizers','safetensors','huggingface_hub','requests','urllib3');"
        "before={m for m in PROHIBITED if m in sys.modules};"
        "import representation_learning.text.manifest as T;"
        "newly=[m for m in PROHIBITED if m in sys.modules and m not in before];"
        "assert not newly, newly; print('lazy-ok')" % str(ROOT.parent))
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "lazy-ok" in r.stdout, r.stderr


def test_text_package_never_touches_the_network_or_hub():
    src = (ROOT / "text" / "manifest.py").read_text()
    for banned in ("import torch", "import transformers", "huggingface_hub", "urllib", "requests",
                   "socket", "from_pretrained", "hf_hub_download"):
        assert banned not in src.replace('"""', "").split("def load_manifest")[1], banned
