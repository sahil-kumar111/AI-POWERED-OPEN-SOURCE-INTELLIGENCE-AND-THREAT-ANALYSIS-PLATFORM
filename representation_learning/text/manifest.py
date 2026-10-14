"""Text-model manifest (Phase 3, G1): pins source revision, licence, local path and SHA-256 per file
for TEXT encoder candidates. Separate from the visual `models.manifest.json` (Phase 2), which this
module never reads or writes. Nothing here downloads, and nothing imports torch/transformers.

Committed manifest: `representation_learning/text_models.manifest.json`.
 * `expected_*`, per-file `size` and `upstream_sha256` come from verified Hugging Face metadata
   (G0). `upstream_sha256` is null where the Hub reports no LFS hash; it is never invented.
 * `revision`, `license_id`, `license`, per-file `sha256`, `observed_config`, `recorded_on` are
   filled ONLY by `record`, after the files exist locally and every check passes.
 * Every entry is a candidate and `final_encoder` must stay false (model selection is a separate
   decision).

CLI (run from the repository root):
  python -m representation_learning.text.manifest status
  python -m representation_learning.text.manifest record <name> --revision <40-hex> \
         --license-id <id> --license "<licence text from the model card>"
  python -m representation_learning.text.manifest verify [name ...] [--manifest-sha256 <hex>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from ..encoders.errors import EncoderIntegrityError, EncoderLoadError
from ..encoders.integrity import sha256_file

PLACEHOLDER = "TO_RECORD"
MANIFEST_VERSION = "0.1"
KIND = "text"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "text_models.manifest.json"

REQUIRED_FILES = ("model.safetensors", "config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json")
# Evidence files that check_local_semantics() reads. They must be LISTED in `files` so they also go
# through the regular-file, size, SHA-256 and verify checks (not merely a JSON semantic check).
POOLING_FILE = "1_Pooling/config.json"                  # always read (pooling mode + dimension)
SEQ_LEN_FILE = "sentence_bert_config.json"              # read when sentence_bert_max_seq_length is set
FILE_ROLES = ("weights", "config", "tokenizer", "evidence")
WEIGHTS_FILE = "model.safetensors"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_SUFFIXES = (".bin", ".onnx", ".h5", ".pt", ".pth", ".ckpt", ".msgpack", ".pkl",
                       ".pickle", ".gguf", ".safetensors")

_TOP_KEYS = {"manifest_version", "kind", "note", "models"}
_ENTRY_KEYS = {"status", "final_encoder", "family", "hf_id", "source_url", "local_dir",
               "expected_revision", "revision", "expected_license_id", "license_id", "license",
               "encoder", "files", "observed_config", "recorded_on"}
_ENCODER_KEYS = {"expected_dim", "max_position_embeddings", "max_input_tokens",
                 "max_input_tokens_status", "sentence_bert_max_seq_length", "token_limit_policy",
                 "pooling", "upstream_normalize_module", "output_normalization", "input_template"}
_FILE_KEYS = {"role", "size", "upstream_sha256", "sha256"}
_ALLOW_LIST_HINTS = ("allow", "supported_lang", "languages")


# ----------------------------------------------------------------------------- loading
def load_manifest(path: Optional[Path] = None, validate: bool = True) -> Dict[str, Any]:
    p = Path(path or DEFAULT_MANIFEST)
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EncoderLoadError(f"cannot read text manifest {p}: {e}") from e
    if validate:
        validate_manifest(m)
    return m


def _entry(manifest: Dict[str, Any], name: str) -> Dict[str, Any]:
    try:
        return manifest["models"][name]
    except KeyError:
        raise EncoderLoadError(
            f"unknown text manifest entry {name!r}; available: {sorted(manifest.get('models', {}))}")


# ----------------------------------------------------------------------------- validation
def _safe_relpath(p: Any) -> bool:
    if not isinstance(p, str) or not p or "\\" in p or "\x00" in p:
        return False
    pp = PurePosixPath(p)
    return not pp.is_absolute() and ".." not in pp.parts and "." not in pp.parts and str(pp) == p


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def entry_problems(name: str, e: Any) -> List[str]:
    """Structural/consistency problems of one entry (independent of local files)."""
    pre = f"{name}: "
    if not isinstance(e, dict):
        return [pre + "entry must be an object"]
    out: List[str] = []
    extra = set(e) - _ENTRY_KEYS
    missing = _ENTRY_KEYS - set(e)
    if extra:
        hint = " (language allow-lists are not permitted)" if any(
            h in k.lower() for k in extra for h in _ALLOW_LIST_HINTS) else ""
        out.append(pre + f"unknown keys {sorted(extra)}{hint}")
    if missing:
        out.append(pre + f"missing keys {sorted(missing)}")
    if e.get("final_encoder") is not False:
        out.append(pre + "final_encoder must be the boolean false (model selection is a separate decision)")
    if e.get("status") != "candidate":
        out.append(pre + "status must be 'candidate'")
    if not isinstance(e.get("hf_id"), str) or e.get("hf_id", "").count("/") != 1:
        out.append(pre + "hf_id must be 'owner/name'")
    elif e.get("source_url") != "https://huggingface.co/" + e["hf_id"]:
        out.append(pre + "source_url must be https://huggingface.co/<hf_id>")
    ld = e.get("local_dir")
    if not _safe_relpath(ld) or not str(ld).startswith("models/") or str(ld) == "models/":
        out.append(pre + "local_dir must be a safe relative path under models/")
    if not _HEX40.match(str(e.get("expected_revision", ""))):
        out.append(pre + "expected_revision must be a full 40-character lowercase hex commit")
    rev = e.get("revision")
    if rev != PLACEHOLDER and not _HEX40.match(str(rev)):
        out.append(pre + "revision must be TO_RECORD or a 40-hex commit")
    if not isinstance(e.get("expected_license_id"), str) or not e.get("expected_license_id", "").strip():
        out.append(pre + "expected_license_id must be a non-empty string")
    for k in ("license_id", "license"):
        if not isinstance(e.get(k), str) or not e.get(k, "").strip():
            out.append(pre + f"{k} must be a non-empty string (TO_RECORD until recorded)")
    # encoder design block
    enc = e.get("encoder")
    if not isinstance(enc, dict):
        out.append(pre + "encoder block must be an object")
    else:
        if set(enc) - _ENCODER_KEYS:
            hint = " (language allow-lists are not permitted)" if any(
                h in k.lower() for k in set(enc) - _ENCODER_KEYS for h in _ALLOW_LIST_HINTS) else ""
            out.append(pre + f"encoder: unknown keys {sorted(set(enc) - _ENCODER_KEYS)}{hint}")
        if _ENCODER_KEYS - set(enc):
            out.append(pre + f"encoder: missing keys {sorted(_ENCODER_KEYS - set(enc))}")
        if not _is_int(enc.get("expected_dim")) or enc.get("expected_dim", 0) <= 0:
            out.append(pre + "encoder.expected_dim must be a positive integer")
        for k in ("max_position_embeddings", "max_input_tokens", "sentence_bert_max_seq_length"):
            v = enc.get(k)
            if v is not None and (not _is_int(v) or v <= 0):
                out.append(pre + f"encoder.{k} must be null or a positive integer")
        mi, mp = enc.get("max_input_tokens"), enc.get("max_position_embeddings")
        if _is_int(mi) and _is_int(mp) and mi > mp:
            out.append(pre + "encoder.max_input_tokens exceeds max_position_embeddings")
        if enc.get("token_limit_policy") != "reject_over_limit":
            out.append(pre + "encoder.token_limit_policy must be 'reject_over_limit' (no silent truncation)")
        if enc.get("pooling") != "mean":
            out.append(pre + "encoder.pooling must be 'mean'")
        if enc.get("output_normalization") != "l2":
            out.append(pre + "encoder.output_normalization must be 'l2'")
        if enc.get("upstream_normalize_module") not in (True, False, None):
            out.append(pre + "encoder.upstream_normalize_module must be true, false or null")
        if not isinstance(enc.get("max_input_tokens_status"), str):
            out.append(pre + "encoder.max_input_tokens_status must be a string")
        tpl = enc.get("input_template")
        if not isinstance(tpl, dict) or set(tpl) != {"role_required", "roles"} \
                or not isinstance(tpl.get("role_required"), bool):
            out.append(pre + "encoder.input_template must be {role_required: bool, roles: ...}")
        elif tpl["role_required"]:
            roles = tpl.get("roles")
            if not isinstance(roles, dict) or set(roles) != {"query", "passage"} or \
                    not all(isinstance(v, str) and v.strip() for v in roles.values()):
                out.append(pre + "role_required input_template needs non-empty 'query' and 'passage' prefixes")
        elif tpl.get("roles") is not None:
            out.append(pre + "input_template.roles must be null when no role is required")
    # files
    files = e.get("files")
    if not isinstance(files, dict) or not files:
        out.append(pre + "files must be a non-empty object")
    else:
        for req in REQUIRED_FILES:
            if req not in files:
                out.append(pre + f"required file {req!r} is not listed")
        if POOLING_FILE not in files:
            out.append(pre + f"required evidence file {POOLING_FILE!r} is not listed "
                             "(read to validate pooling and dimension)")
        if isinstance(enc, dict) and enc.get("sentence_bert_max_seq_length") is not None \
                and SEQ_LEN_FILE not in files:
            out.append(pre + f"required evidence file {SEQ_LEN_FILE!r} is not listed "
                             "(encoder.sentence_bert_max_seq_length is set, so it is read to validate the limit)")
        for fname, f in files.items():
            fp = f"{pre}files[{fname!r}]: "
            if not _safe_relpath(fname):
                out.append(fp + "unsafe file path")
                continue
            if fname != WEIGHTS_FILE and fname.lower().endswith(_FORBIDDEN_SUFFIXES):
                out.append(fp + "weight/pickle formats other than model.safetensors are not allowed")
            if not isinstance(f, dict) or set(f) != _FILE_KEYS:
                out.append(fp + f"must have exactly keys {sorted(_FILE_KEYS)}")
                continue
            if f["role"] not in FILE_ROLES:
                out.append(fp + f"role must be one of {FILE_ROLES}")
            if (fname == WEIGHTS_FILE) != (f["role"] == "weights"):
                out.append(fp + "only model.safetensors may have role 'weights'")
            if not _is_int(f["size"]) or f["size"] <= 0:
                out.append(fp + "size must be a positive integer")
            up = f["upstream_sha256"]
            if up is not None and not _HEX64.match(str(up)):
                out.append(fp + "upstream_sha256 must be null or 64 lowercase hex")
            if fname == WEIGHTS_FILE and up is None:
                out.append(fp + "weights must carry an upstream SHA-256 (provenance)")
            sh = f["sha256"]
            if sh != PLACEHOLDER and not _HEX64.match(str(sh)):
                out.append(fp + "sha256 must be TO_RECORD or 64 lowercase hex")
    oc = e.get("observed_config")
    if oc is not None and not isinstance(oc, dict):
        out.append(pre + "observed_config must be null or an object")
    ro = e.get("recorded_on")
    if ro is not None and not (isinstance(ro, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", ro)):
        out.append(pre + "recorded_on must be null or YYYY-MM-DD")
    return out


def manifest_problems(m: Any) -> List[str]:
    if not isinstance(m, dict):
        return ["manifest must be an object"]
    out: List[str] = []
    if set(m) != _TOP_KEYS:
        out.append(f"top-level keys must be exactly {sorted(_TOP_KEYS)}")
    if m.get("manifest_version") != MANIFEST_VERSION:
        out.append(f"manifest_version must be {MANIFEST_VERSION!r}")
    if m.get("kind") != KIND:
        out.append(f"kind must be {KIND!r} (this is not the visual manifest)")
    models = m.get("models")
    if not isinstance(models, dict) or not models:
        out.append("models must be a non-empty object")
        return out
    dirs: Dict[str, str] = {}
    for name, e in models.items():
        out.extend(entry_problems(name, e))
        if isinstance(e, dict) and isinstance(e.get("local_dir"), str):
            if e["local_dir"] in dirs:
                out.append(f"{name}: local_dir duplicates {dirs[e['local_dir']]}")
            dirs[e["local_dir"]] = name
    return out


def validate_manifest(m: Any) -> None:
    probs = manifest_problems(m)
    if probs:
        raise EncoderLoadError("invalid text manifest: " + "; ".join(probs))


# ----------------------------------------------------------------------------- pin state
def pin_problems(e: Dict[str, Any]) -> List[str]:
    """What is still missing before this entry may be used (empty list = fully pinned)."""
    out: List[str] = []
    rev = e.get("revision")
    if not _HEX40.match(str(rev)):
        out.append("revision not recorded")
    elif rev != e.get("expected_revision"):
        out.append("recorded revision differs from expected_revision")
    if e.get("license_id") in ("", PLACEHOLDER, None):
        out.append("licence id not recorded")
    elif e.get("license_id") != e.get("expected_license_id"):
        out.append("recorded licence id differs from expected_license_id")
    if e.get("license") in ("", PLACEHOLDER, None):
        out.append("licence text not recorded")
    for fname, f in (e.get("files") or {}).items():
        if not _HEX64.match(str(f.get("sha256", ""))):
            out.append(f"sha256 of {fname} not recorded")
        elif f.get("upstream_sha256") and f["sha256"] != f["upstream_sha256"]:
            out.append(f"sha256 of {fname} differs from upstream_sha256")
    if not isinstance(e.get("observed_config"), dict):
        out.append("observed_config not recorded")
    if not e.get("recorded_on"):
        out.append("recorded_on not set")
    return out


def design_open_items(e: Dict[str, Any]) -> List[str]:
    """Design facts deliberately left undecided (informational; not a pin problem)."""
    enc = e.get("encoder") or {}
    return ["max_input_tokens"] if enc.get("max_input_tokens") is None else []


# ----------------------------------------------------------------------------- local files
def local_dir(entry: Dict[str, Any], manifest_path: Optional[Path] = None) -> Path:
    base = Path(manifest_path or DEFAULT_MANIFEST).resolve().parent
    d = (base / entry["local_dir"]).resolve()
    if base not in d.parents:
        raise EncoderLoadError(f"local_dir {entry['local_dir']!r} escapes the manifest directory")
    return d


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EncoderIntegrityError(f"{what} is not readable JSON: {e}") from e


def _forbidden_weight_files(d: Path) -> List[str]:
    bad = []
    for p in d.rglob("*"):
        if p.is_file() or p.is_symlink():
            rel = p.relative_to(d).as_posix()
            n = p.name.lower()
            if rel == WEIGHTS_FILE:
                continue
            if n.endswith(_FORBIDDEN_SUFFIXES) or n == "model.safetensors.index.json":
                bad.append(rel)
    return sorted(bad)


def _safetensors_summary(path: Path, dim: int) -> Dict[str, Any]:
    """Validate a single-file safetensors container WITHOUT torch: header length, JSON header,
    tensor offsets covering the data buffer exactly, and the word-embedding width if present."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise EncoderIntegrityError("model.safetensors is too small to be a safetensors file")
        n = int.from_bytes(head, "little")
        if n <= 0 or 8 + n > size or n > 100 * 1024 * 1024:
            raise EncoderIntegrityError("model.safetensors header length is invalid")
        raw = f.read(n)
    try:
        hdr = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise EncoderIntegrityError(f"model.safetensors header is not valid JSON: {e}") from e
    if not isinstance(hdr, dict):
        raise EncoderIntegrityError("model.safetensors header must be a JSON object")
    end, count, emb = 0, 0, None
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        if not isinstance(v, dict) or not {"dtype", "shape", "data_offsets"} <= set(v):
            raise EncoderIntegrityError(f"model.safetensors tensor {k!r} entry is malformed")
        o = v["data_offsets"]
        if not (isinstance(o, list) and len(o) == 2 and all(_is_int(x) for x in o)
                and 0 <= o[0] <= o[1]):
            raise EncoderIntegrityError(f"model.safetensors tensor {k!r} has bad data_offsets")
        end = max(end, o[1])
        count += 1
        if k.endswith("word_embeddings.weight"):
            emb = list(v["shape"])
    if count == 0:
        raise EncoderIntegrityError("model.safetensors contains no tensors")
    if 8 + n + end != size:
        raise EncoderIntegrityError("model.safetensors tensor offsets do not match the file size")
    if emb is not None and (len(emb) != 2 or emb[1] != dim):
        raise EncoderIntegrityError(
            f"model.safetensors word-embedding shape {emb} does not have width {dim}")
    return {"safetensors_tensor_count": count, "safetensors_word_embedding_shape": emb}


def _check_pooling(cfg: Any, dim: int) -> None:
    if not isinstance(cfg, dict):
        raise EncoderIntegrityError("1_Pooling/config.json must be a JSON object")
    if cfg.get("pooling_mode_mean_tokens") is not True:
        raise EncoderIntegrityError("1_Pooling/config.json does not declare mean pooling")
    others = sorted(k for k, v in cfg.items()
                    if k.startswith("pooling_mode_") and k != "pooling_mode_mean_tokens" and v)
    if others:
        raise EncoderIntegrityError(f"1_Pooling/config.json enables other pooling modes: {others}")
    wd = cfg.get("word_embedding_dimension")
    if wd is not None and wd != dim:
        raise EncoderIntegrityError(f"1_Pooling/config.json dimension {wd} != expected {dim}")


def check_local_files(entry: Dict[str, Any], d: Path) -> Dict[str, str]:
    """Existence, regular-file, size and SHA-256 for every listed file, plus refusal of stray
    weight formats. Returns {file: sha256}. Raises on the first class of problem found."""
    if not d.is_dir():
        raise EncoderLoadError(f"local directory not found: {d}")
    stray = _forbidden_weight_files(d)
    if stray:
        raise EncoderIntegrityError(
            f"refusing directory with extra weight/pickle files (only model.safetensors is allowed): {stray}")
    hashes: Dict[str, str] = {}
    for fname, f in entry["files"].items():
        fp = d / fname
        if fp.is_symlink() or not fp.is_file():
            raise EncoderLoadError(f"{fname}: missing or not a regular file in {d}")
        if fp.stat().st_size != f["size"]:
            raise EncoderIntegrityError(
                f"{fname}: size mismatch (expected {f['size']}, actual {fp.stat().st_size})")
        hashes[fname] = sha256_file(fp)
    return hashes


def check_local_semantics(entry: Dict[str, Any], d: Path) -> Dict[str, Any]:
    """Cross-check the local files against the entry's design facts; returns observed_config."""
    enc = entry["encoder"]
    dim = enc["expected_dim"]
    cfg = _read_json(d / "config.json", "config.json")
    if not isinstance(cfg, dict):
        raise EncoderIntegrityError("config.json must be a JSON object")
    if cfg.get("hidden_size") != dim:
        raise EncoderIntegrityError(f"config.json hidden_size {cfg.get('hidden_size')!r} != expected {dim}")
    mp = enc["max_position_embeddings"]
    if mp is not None and cfg.get("max_position_embeddings") != mp:
        raise EncoderIntegrityError(
            f"config.json max_position_embeddings {cfg.get('max_position_embeddings')!r} != expected {mp}")
    tok = _read_json(d / "tokenizer.json", "tokenizer.json")
    if not isinstance(tok, dict) or not isinstance(tok.get("model"), dict):
        raise EncoderIntegrityError("tokenizer.json has no 'model' object")
    for aux in ("tokenizer_config.json", "special_tokens_map.json"):
        if not isinstance(_read_json(d / aux, aux), dict):
            raise EncoderIntegrityError(f"{aux} must be a JSON object")
    _check_pooling(_read_json(d / "1_Pooling" / "config.json", "1_Pooling/config.json"), dim)
    want = enc["sentence_bert_max_seq_length"]
    if want is not None:
        sb = _read_json(d / "sentence_bert_config.json", "sentence_bert_config.json")
        if not isinstance(sb, dict) or sb.get("max_seq_length") != want:
            raise EncoderIntegrityError(
                f"sentence_bert_config.json max_seq_length != expected {want}")
    st = _safetensors_summary(d / WEIGHTS_FILE, dim)
    return {"model_type": cfg.get("model_type"), "hidden_size": cfg.get("hidden_size"),
            "max_position_embeddings": cfg.get("max_position_embeddings"),
            "vocab_size": cfg.get("vocab_size"), **st}


# ----------------------------------------------------------------------------- record / verify
def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def record_entry(manifest_path: Path, name: str, revision: str, license_id: str,
                 license_text: str) -> Dict[str, Any]:
    """Hash the local files and write revision/licence/hashes into the manifest. Nothing is
    written unless every check passes; an existing pin is never silently replaced."""
    if not _HEX40.match(revision):
        raise EncoderIntegrityError("revision must be the full 40-character lowercase commit hash")
    if not license_text.strip() or license_text.strip() == PLACEHOLDER:
        raise EncoderIntegrityError("licence text must be provided (copy it from the model card)")
    m = load_manifest(manifest_path)
    e = _entry(m, name)
    if revision != e["expected_revision"]:
        raise EncoderIntegrityError(
            f"revision {revision} does not match the verified expected_revision {e['expected_revision']}")
    if license_id != e["expected_license_id"]:
        raise EncoderIntegrityError(
            f"licence id {license_id!r} does not match the verified id {e['expected_license_id']!r}")
    d = local_dir(e, manifest_path)
    hashes = check_local_files(e, d)
    for fname, h in hashes.items():
        up = e["files"][fname]["upstream_sha256"]
        if up is not None and h != up:
            raise EncoderIntegrityError(
                f"{fname}: SHA-256 {h} does not match upstream_sha256 {up}")
    observed = check_local_semantics(e, d)
    already = {f: v["sha256"] for f, v in e["files"].items() if _HEX64.match(str(v["sha256"]))}
    if already and already != hashes:
        raise EncoderIntegrityError(
            f"{name} is already recorded with different hashes; refusing to overwrite a pin")
    updated = json.loads(json.dumps(e))
    for fname, h in hashes.items():
        updated["files"][fname]["sha256"] = h
    updated.update({"revision": revision, "license_id": license_id, "license": license_text,
                    "observed_config": observed})
    if not already or not e.get("recorded_on"):
        updated["recorded_on"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    m["models"][name] = updated
    validate_manifest(m)
    _atomic_write(Path(manifest_path), json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    return updated


def verify_entry(manifest: Dict[str, Any], name: str, manifest_path: Optional[Path] = None) -> Path:
    """Refuse unless the entry is fully pinned and every local file still matches."""
    validate_manifest(manifest)
    e = _entry(manifest, name)
    probs = pin_problems(e)
    if probs:
        raise EncoderLoadError(f"text manifest entry {name!r} is not pinned yet: " + "; ".join(probs))
    d = local_dir(e, manifest_path)
    bad = []
    if not d.is_dir():
        raise EncoderLoadError(f"local directory not found: {d}")
    stray = _forbidden_weight_files(d)
    if stray:
        raise EncoderIntegrityError(
            f"{name}: extra weight/pickle files present (only model.safetensors is allowed): {stray}")
    for fname, f in e["files"].items():
        fp = d / fname
        if fp.is_symlink() or not fp.is_file():
            bad.append(f"{fname}: missing")
        elif fp.stat().st_size != f["size"]:
            bad.append(f"{fname}: size mismatch")
        elif sha256_file(fp) != f["sha256"]:
            bad.append(f"{fname}: SHA-256 mismatch")
    if bad:
        raise EncoderIntegrityError(f"{name}: " + "; ".join(bad))
    if check_local_semantics(e, d) != e["observed_config"]:
        raise EncoderIntegrityError(f"{name}: local configuration differs from the recorded observed_config")
    return d


# ----------------------------------------------------------------------------- CLI
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("record")
    r.add_argument("name")
    r.add_argument("--revision", required=True)
    r.add_argument("--license-id", required=True, dest="license_id")
    r.add_argument("--license", required=True, dest="license_text")
    v = sub.add_parser("verify")
    v.add_argument("names", nargs="*")
    v.add_argument("--manifest-sha256", dest="manifest_sha256", default=None,
                   help="expected SHA-256 of the manifest file itself (tamper check)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "status":
            m = load_manifest(a.manifest)
            for n, e in m["models"].items():
                p = pin_problems(e)
                openi = design_open_items(e)
                print(f"{n}: {e['status']} (final_encoder={str(e['final_encoder']).lower()}) "
                      f"{'PINNED' if not p else 'NOT PINNED - ' + '; '.join(p)}  "
                      f"[{e['hf_id']} -> {e['local_dir']}]"
                      + (f"  design open: {', '.join(openi)}" if openi else ""))
        elif a.cmd == "record":
            e = record_entry(a.manifest, a.name, a.revision, a.license_id, a.license_text)
            print(f"recorded {a.name}: revision {e['revision']}, "
                  f"model.safetensors sha256 {e['files'][WEIGHTS_FILE]['sha256']}")
        else:
            if a.manifest_sha256 is not None:
                if not _HEX64.match(a.manifest_sha256.lower()):
                    raise EncoderIntegrityError("--manifest-sha256 must be 64 hex characters")
                actual = sha256_file(Path(a.manifest))
                if actual != a.manifest_sha256.lower():
                    raise EncoderIntegrityError(
                        f"manifest file SHA-256 mismatch: expected {a.manifest_sha256}, actual {actual}")
            m = load_manifest(a.manifest)
            for n in (a.names or list(m["models"])):
                verify_entry(m, n, a.manifest)
                print(f"{n}: OK (all files match recorded SHA-256; candidate, final_encoder=false)")
    except (EncoderLoadError, EncoderIntegrityError) as ex:
        print(f"error: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main())
