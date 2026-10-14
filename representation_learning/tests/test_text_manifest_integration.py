"""Phase 3 / G2: adapters <-> text manifest integration, laziness and offline guarantees.
All offline. Real-looking manifests here are COPIES with tiny synthetic local files; the committed
manifest is only read. Nothing says anything about real E5/MiniLM checkpoints."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from representation_learning.encoders.datatypes import EncoderConfig
from representation_learning.testing.text_mocks import (MockTextBackend, make_pinned_synthetic_manifest,
                                                        make_text_item)
from representation_learning.text import manifest as T
from representation_learning.text.adapters import (ADAPTERS, E5_NAME, E5_SPEC, MINILM_NAME,
                                                   MINILM_SPEC, spec_problems,
                                                   text_encoder_from_manifest)
from representation_learning.text.errors import EncoderIntegrityError, EncoderLoadError

ROOT = Path(__file__).resolve().parents[1]
SPECS = {E5_NAME: E5_SPEC, MINILM_NAME: MINILM_SPEC}


def committed():
    return json.loads(T.DEFAULT_MANIFEST.read_text())


# ------------------------------------------------------------------ code spec == manifest
@pytest.mark.parametrize("name", [E5_NAME, MINILM_NAME])
def test_code_specs_agree_with_the_committed_manifest(name):
    assert spec_problems(SPECS[name], committed()["models"][name]) == []


def test_adapters_cover_exactly_the_two_manifest_candidates():
    assert set(ADAPTERS) == set(committed()["models"]) == {E5_NAME, MINILM_NAME}
    for name, cls in ADAPTERS.items():
        assert cls.SPEC is SPECS[name] and cls.SPEC.name == name


def test_e5_limit_is_decided_in_code_within_the_verified_capacity_and_manifest_is_unchanged():
    e = committed()["models"][E5_NAME]["encoder"]
    assert e["max_input_tokens"] is None                               # G1 file untouched
    assert E5_SPEC.max_input_tokens <= e["max_position_embeddings"] == 512
    assert T.design_open_items(committed()["models"][E5_NAME]) == ["max_input_tokens"]


def _mut(name, fn):
    m = copy.deepcopy(committed()["models"][name])
    fn(m)
    return spec_problems(SPECS[name], m)


@pytest.mark.parametrize("name,fn,expect", [
    (E5_NAME, lambda m: m["encoder"].update(expected_dim=768), "expected_dim"),
    (E5_NAME, lambda m: m["encoder"]["input_template"]["roles"].update(query="q: "), "role prefixes"),
    (E5_NAME, lambda m: m["encoder"]["input_template"].update(role_required=False, roles=None), "role_required"),
    (E5_NAME, lambda m: m["encoder"].update(max_position_embeddings=256), "exceeds"),
    (E5_NAME, lambda m: m["encoder"].update(max_input_tokens=256), "max_input_tokens"),
    (E5_NAME, lambda m: m["encoder"].update(pooling="cls"), "pooling"),
    (E5_NAME, lambda m: m.update(source_url="https://huggingface.co/other/model"), "source_ref"),
    (MINILM_NAME, lambda m: m["encoder"].update(max_input_tokens=256), "max_input_tokens"),
    (MINILM_NAME, lambda m: m["encoder"].update(sentence_bert_max_seq_length=512), "sentence_bert"),
    (MINILM_NAME, lambda m: m["encoder"]["input_template"].update(role_required=True,
                                                                 roles={"query": "q", "passage": "p"}), "role"),
])
def test_spec_drift_against_the_manifest_is_detected(name, fn, expect):
    probs = _mut(name, fn)
    assert probs and any(expect in p for p in probs), probs


# ------------------------------------------------------------------ candidates stay candidates
def test_no_adapter_or_spec_claims_selection():
    for name, spec in SPECS.items():
        assert "final" not in json.dumps(spec.preprocessing_dict()).lower()
        assert not hasattr(spec, "final_encoder") and not hasattr(spec, "selected")
    m = committed()
    assert all(e["status"] == "candidate" and e["final_encoder"] is False for e in m["models"].values())
    assert all(e["revision"] == "TO_RECORD" for e in m["models"].values())      # no pin recorded


def test_committed_manifest_is_never_modified_by_the_adapter_code(tmp_path):
    before = T.DEFAULT_MANIFEST.read_bytes()
    for name in ADAPTERS:
        with pytest.raises(EncoderLoadError, match="not pinned"):
            text_encoder_from_manifest(name)
    assert T.DEFAULT_MANIFEST.read_bytes() == before


# ------------------------------------------------------------------ from_manifest (synthetic pins)
@pytest.mark.parametrize("name", [E5_NAME, MINILM_NAME])
def test_from_manifest_builds_a_lazy_adapter_from_a_pinned_entry(tmp_path, name):
    mp = make_pinned_synthetic_manifest(tmp_path, name)
    e = T.load_manifest(mp)["models"][name]
    enc = text_encoder_from_manifest(name, mp)
    assert isinstance(enc, ADAPTERS[name]) and not enc.loaded
    assert enc.config.checkpoint_id == f"{e['hf_id']}@{e['revision']}"
    assert enc.config.revision == e["revision"] and enc.config.license == e["license"]
    assert enc.config.weights_dir == (tmp_path / e["local_dir"]).resolve()
    assert enc.config.expected_weights_sha256 == e["files"]["model.safetensors"]["sha256"]
    assert enc.file_sha256 == {f: v["sha256"] for f, v in e["files"].items()}


def test_from_manifest_adapter_records_the_pinned_identity_when_given_a_mock_backend(tmp_path):
    mp = make_pinned_synthetic_manifest(tmp_path, E5_NAME)
    enc = text_encoder_from_manifest(E5_NAME, mp, backend=MockTextBackend(384))
    r = enc.encode(make_text_item(), role="query")
    e = T.load_manifest(mp)["models"][E5_NAME]
    assert r.record["encoder"]["version_or_checkpoint"].endswith(e["revision"])
    assert r.record["encoder"]["license"] == e["license"]


def test_from_manifest_refuses_unpinned_unknown_tampered_and_drifted_entries(tmp_path):
    with pytest.raises(EncoderLoadError, match="unknown text manifest entry"):
        text_encoder_from_manifest("nope", make_pinned_synthetic_manifest(tmp_path / "a", E5_NAME))
    mp = make_pinned_synthetic_manifest(tmp_path / "b", E5_NAME)
    (tmp_path / "b" / "models" / "multilingual-e5-small" / "special_tokens_map.json").write_bytes(
        b'{"pad_token": "<pat>"}')                                              # same size, new content
    with pytest.raises(EncoderIntegrityError, match="SHA-256 mismatch"):
        text_encoder_from_manifest(E5_NAME, mp)
    mp2 = make_pinned_synthetic_manifest(tmp_path / "c", MINILM_NAME)
    m = json.loads(mp2.read_text())
    m["models"][MINILM_NAME]["encoder"]["input_template"] = {"role_required": True,
                                                            "roles": {"query": "q: ", "passage": "p: "}}
    mp2.write_text(json.dumps(m, indent=2) + "\n")
    with pytest.raises(EncoderIntegrityError, match="code spec disagrees"):
        text_encoder_from_manifest(MINILM_NAME, mp2)


def test_from_manifest_entry_without_an_adapter_is_refused(tmp_path):
    import shutil
    mp = make_pinned_synthetic_manifest(tmp_path, E5_NAME)
    m = json.loads(mp.read_text())
    e = copy.deepcopy(m["models"][E5_NAME])
    for f in e["files"].values():
        f["sha256"] = "TO_RECORD"
    e.update(revision="TO_RECORD", license_id="TO_RECORD", license="TO_RECORD", observed_config=None,
             recorded_on=None, local_dir=m["models"][E5_NAME]["local_dir"] + "-2")
    m["models"]["new-candidate"] = e
    mp.write_text(json.dumps(m, indent=2) + "\n")
    shutil.copytree(tmp_path / m["models"][E5_NAME]["local_dir"], tmp_path / e["local_dir"])
    T.record_entry(mp, "new-candidate", e["expected_revision"], e["expected_license_id"], "synthetic")
    with pytest.raises(EncoderLoadError, match="no text adapter implemented"):
        text_encoder_from_manifest("new-candidate", mp)


# ------------------------------------------------------------------ real backend: pre-torch paths only
def test_torch_backend_checks_configuration_before_importing_torch(monkeypatch):
    from representation_learning.text.torch_backend import TorchTextBackend
    monkeypatch.setitem(sys.modules, "torch", None)                            # import torch -> ImportError
    with pytest.raises(EncoderLoadError, match="weights_dir is not configured"):
        TorchTextBackend(EncoderConfig(), E5_SPEC)
    with pytest.raises(EncoderLoadError, match="not found"):
        TorchTextBackend(EncoderConfig(weights_dir=Path("/nonexistent/dir")), E5_SPEC)


def test_torch_backend_verifies_files_then_reports_missing_dependencies(tmp_path, monkeypatch):
    from representation_learning.text.torch_backend import TorchTextBackend
    monkeypatch.setitem(sys.modules, "torch", None)
    mp = make_pinned_synthetic_manifest(tmp_path, E5_NAME)
    enc = text_encoder_from_manifest(E5_NAME, mp)
    with pytest.raises(EncoderLoadError, match="torch and transformers are required"):
        enc.encode(make_text_item(), role="query")                             # integrity OK -> deps missing
    assert not enc.loaded
    wrong = dict(enc.file_sha256, **{"tokenizer.json": "0" * 64})
    with pytest.raises(EncoderIntegrityError, match="tokenizer.json"):
        TorchTextBackend(enc.config, E5_SPEC, file_sha256=wrong)               # integrity first
    bad_cfg = EncoderConfig(weights_dir=enc.config.weights_dir, expected_weights_sha256="0" * 64)
    with pytest.raises(EncoderIntegrityError, match="SHA-256 mismatch"):
        TorchTextBackend(bad_cfg, E5_SPEC)
    with pytest.raises(EncoderIntegrityError, match="expected_weights_sha256 is required"):
        TorchTextBackend(EncoderConfig(weights_dir=enc.config.weights_dir), E5_SPEC)


def test_torch_backend_source_is_local_only():
    import ast
    tree = ast.parse((ROOT / "text" / "torch_backend.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "from_pretrained"]
    assert len(calls) == 2
    for c in calls:
        kw = {k.arg: k.value.value for k in c.keywords if isinstance(k.value, ast.Constant)}
        assert kw.get("local_files_only") is True and kw.get("trust_remote_code") is False
    assert any(k.arg == "use_safetensors" and k.value.value is True
               for c in calls for k in c.keywords if isinstance(k.value, ast.Constant))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module and n.level == 0}
    assert not imported & {"urllib", "requests", "socket", "http", "huggingface_hub", "torch", "transformers"}
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"hf_hub_download", "snapshot_download", "urlopen", "urlretrieve"}


# ------------------------------------------------------------------ laziness / offline
PROHIBITED = "('torch','transformers','tokenizers','safetensors','huggingface_hub','requests','urllib3')"
MODULES = ["representation_learning.text.contracts", "representation_learning.text.errors",
           "representation_learning.text.encoder", "representation_learning.text.adapters",
           "representation_learning.text.torch_backend", "representation_learning.testing.text_mocks"]


@pytest.mark.parametrize("module", MODULES)
def test_importing_text_modules_is_lazy_and_offline_safe(module):
    # Only modules that the import ADDS are attributable to it (some environments preload e.g.
    # requests/urllib3 at start-up), so the baseline is captured before importing.
    code = (f"import sys; sys.path.insert(0, {str(ROOT.parent)!r});"
            f"P={PROHIBITED}; before={{m for m in P if m in sys.modules}};"
            f"import {module};"
            "newly=[m for m in P if m in sys.modules and m not in before];"
            "assert not newly, newly; print('lazy-ok')")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "lazy-ok" in r.stdout, r.stderr


def test_constructing_and_using_mock_encoders_never_imports_the_model_stack():
    code = (f"import sys; sys.path.insert(0, {str(ROOT.parent)!r});"
            f"P={PROHIBITED}; before={{m for m in P if m in sys.modules}};"
            "from representation_learning.testing.text_mocks import mock_text_encoder, make_text_item;"
            "e=mock_text_encoder('e5'); e.encode(make_text_item(), role='query');"
            "from representation_learning.text.adapters import E5TextEncoder, MiniLMTextEncoder;"
            "E5TextEncoder(); MiniLMTextEncoder();"
            "newly=[m for m in P if m in sys.modules and m not in before];"
            "assert not newly, newly; print('lazy-ok')")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "lazy-ok" in r.stdout, r.stderr


@pytest.mark.parametrize("fname", ["contracts.py", "errors.py", "encoder.py", "adapters.py"])
def test_core_text_modules_contain_no_network_or_model_loading_code(fname):
    src = (ROOT / "text" / fname).read_text()
    for banned in ("import torch", "import transformers", "huggingface_hub", "urllib", "import requests",
                   "socket", "from_pretrained", "hf_hub_download", "snapshot_download"):
        assert banned not in src, (fname, banned)
