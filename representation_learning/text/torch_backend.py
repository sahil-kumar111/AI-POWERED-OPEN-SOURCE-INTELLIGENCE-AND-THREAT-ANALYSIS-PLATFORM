"""Real-checkpoint text backend (PyTorch + Hugging Face `transformers`). Phase 3, G2.

STATUS: written but NOT yet exercised against any real or tiny-random checkpoint (torch and
transformers are not installed in the G2/G2.1 environment and real weights are out of scope).
Tested: the pre-torch paths (configuration and integrity checks, missing-dependency error) and, with
FAKE torch/transformers stand-ins injected by the tests, the load-time integrity sequence and the
arguments passed to `from_pretrained`. The real-checkpoint checks this module still needs are listed
in docs/phase3_text_encoder.md.

Imported lazily by `TextEncoder._make_backend`; importing this module itself imports neither torch
nor transformers. Only LOCAL directories are loaded (`local_files_only=True`, no remote code, no
pickle: `use_safetensors=True`). Nothing here downloads anything.

Behaviour:
  * file integrity first, THEN the optional dependency stack is imported. When manifest hashes are
    given (`file_sha256`), EVERY listed file is hashed and compared with its pin, and the files that
    can affect loading (`REQUIRED_FILES`: weights, config.json, tokenizer.json, tokenizer_config.json,
    special_tokens_map.json) MUST be pinned or the load is refused;
  * the same files (every pinned file; weights, config.json and tokenizer.json even without pins) are
    hashed AGAIN after `from_pretrained` returns; any difference means a file
    changed while loading and the loaded objects are discarded. This detects (it does not prevent)
    modification during the load window; the directory must still be trusted not to be written to by
    others. A change that is reverted before the second hash cannot be detected;
  * the tokenizer is loaded with `use_fast=True` and must come back as a fast tokenizer. (The pinned
    transformers 5.18.0 pops and ignores `use_fast`, so the `is_fast` check is what enforces it there);
  * tokenizer sidecars (G2.2): the optional files the fast-tokenizer loader reads in addition to the
    five required files are `added_tokens.json`, `chat_template.jinja` and
    `additional_chat_templates/*.jinja`. If any is PRESENT it must be listed in the manifest and
    SHA-256-pinned (so it is hashed before and after loading like every other pinned file); present but
    unpinned means refuse. A `tokenizer_config.json` that carries `fast_tokenizer_files` is refused,
    because that key lets the loader pick a versioned tokenizer file instead of the pinned
    `tokenizer.json`. Vocabulary/sentencepiece files (`sentencepiece.bpe.model`, `vocab.txt`, ...) are
    NOT required to be pinned: with a pinned `tokenizer.json` present the loader resolves but does not
    open them. All of this comes from reading the transformers 5.18.0 source (docs/phase3_text_encoder.md
    section 6); it has not been confirmed by running the real loader;
  * the weight file itself (G2.4): `<weights_dir>/model.safetensors` that is a symlink, dangling or
    not, is refused first of all, with or without manifest hashes;
  * model-side files (G2.3): only a single-file `model.safetensors` checkpoint is supported. Before the
    stack is imported, and again after loading, any top-level adapter file (`adapter_config*`,
    `adapter_model*`), weight index file (`*.index.json`) or weight-format file other than the pinned
    `model.safetensors` (`.bin`, other `.safetensors` shards/variants, `.pt`, `.h5`, `.onnx`, ...) is
    refused. See the constants above for how the list was derived (transformers 5.18.0 source, not a
    real load);
  * `count_tokens` tokenizes with special tokens and WITHOUT truncation or padding;
  * `forward` tokenizes the same way (padding only, never truncation), refuses any sequence longer
    than the spec limit as a second line of defence, and returns attention-masked MEAN pooling of
    the last hidden state as float32 (normalisation is done by the encoder base class).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..encoders.datatypes import BackendInfo, EncoderConfig
from ..encoders.integrity import locate_weights, sha256_file, verify_weights
from ..encoders.torch_backends import _import_torch_stack, resolve_device
from .encoder import TextEncoderSpec
from .errors import EncoderIntegrityError, EncoderLoadError, TextTooLongError
from .manifest import _HEX64, REQUIRED_FILES, _safe_relpath

WEIGHTS = "model.safetensors"
# Optional files the transformers fast-tokenizer loader reads from a local directory beyond the
# required files (REQUIRED_FILES). Present => must be pinned. See the module docstring.
TOKENIZER_SIDECAR_FILES = ("added_tokens.json", "chat_template.jinja")
CHAT_TEMPLATE_DIR = "additional_chat_templates"      # every `*.jinja` in it is read by the loader
VERSIONED_TOKENIZER_KEY = "fast_tokenizer_files"     # tokenizer_config.json key that redirects tokenizer.json

# G2.3 single-file checkpoint contract. The directory must hold exactly ONE weight file,
# `model.safetensors`, and no adapter. Derived by reading the transformers 5.18.0 source:
#   * constants: WEIGHTS_NAME `pytorch_model.bin`, WEIGHTS_INDEX_NAME `pytorch_model.bin.index.json`,
#     SAFE_WEIGHTS_NAME `model.safetensors`, SAFE_WEIGHTS_INDEX_NAME `model.safetensors.index.json`
#     (`transformers/utils/__init__.py`);
#   * `_add_variant` inserts a variant before the extension (`model.<variant>.safetensors`,
#     `model.<variant>.safetensors.index.json`), and sharded checkpoints are named
#     `model-0000N-of-0000M.safetensors` / `.bin` (`modeling_utils.py`); shard names inside an index file
#     are chosen by the index, so every `*.index.json` is refused;
#   * PEFT adapters: `adapter_config.json`, `adapter_model.bin`, `adapter_model.safetensors`
#     (`utils/peft_utils.py`). `AutoModel.from_pretrained` calls `find_adapter_config_file` and, when an
#     adapter config exists, can read `base_model_name_or_path` from it (`models/auto/auto_factory.py`).
# Rather than trusting that list to be complete, EVERY top-level entry with a weight-format suffix, an
# index suffix or an adapter prefix is refused unless it is the pinned `model.safetensors`. TensorFlow /
# Flax names (`tf_model.h5`, `flax_model.msgpack`) are not constants of 5.18.0 but are refused by suffix.
WEIGHT_FORMAT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".h5", ".onnx",
                          ".gguf", ".pkl", ".pickle", ".npz", ".npy", ".tflite", ".mlmodel", ".pb")
INDEX_SUFFIX = ".index.json"
ADAPTER_PREFIXES = ("adapter_config", "adapter_model")


def _check_pins(wdir: Path, file_sha256: Mapping[str, str],
                known: Mapping[str, str]) -> Dict[str, str]:
    """Hash every pinned file under `wdir` (reusing hashes already computed in `known`) and compare
    with the pin. Refuses unsafe names, malformed pins, symlinks/non-files and a pin set that omits
    a file which can affect loading. Returns {file: sha256}."""
    missing = [f for f in REQUIRED_FILES if not file_sha256.get(f)]
    if missing:
        raise EncoderIntegrityError(
            f"no pinned SHA-256 for files that affect loading: {sorted(missing)}")
    actual: Dict[str, str] = {}
    for fname, want in file_sha256.items():
        if not _safe_relpath(fname):
            raise EncoderIntegrityError(f"unsafe pinned file name {fname!r}")
        if not isinstance(want, str) or not _HEX64.match(want.lower()):
            raise EncoderIntegrityError(f"{fname}: pinned SHA-256 is not 64 hex characters")
        fp = wdir / fname
        if fp.is_symlink() or not fp.is_file():
            raise EncoderLoadError(f"{fname}: missing or not a regular file in {wdir}")
        got = known.get(fname) or sha256_file(fp)
        if got.lower() != want.lower():
            raise EncoderIntegrityError(f"{fname}: SHA-256 {got} differs from the pinned {want}")
        actual[fname] = got
    return actual


def _check_tokenizer_sidecars(wdir: Path, pins: Mapping[str, str]) -> None:
    """Refuse tokenizer sidecars the loader would read but that are not pinned (`pins` is empty when
    the backend was given no manifest hashes, so any sidecar is then refused), and a
    `tokenizer_config.json` that can redirect loading to an unpinned versioned tokenizer file."""
    present = [n for n in TOKENIZER_SIDECAR_FILES if os.path.lexists(wdir / n)]
    cdir = wdir / CHAT_TEMPLATE_DIR
    if cdir.is_symlink():
        raise EncoderIntegrityError(f"{CHAT_TEMPLATE_DIR} is a symlink; refusing (its files cannot be pinned)")
    if cdir.is_dir():
        present += [f"{CHAT_TEMPLATE_DIR}/{n}" for n in sorted(os.listdir(cdir)) if n.endswith(".jinja")]
    unpinned = [n for n in present if not pins.get(n)]
    if unpinned:
        raise EncoderIntegrityError(
            f"tokenizer sidecar file(s) present but not pinned in the manifest: {unpinned}; the loader "
            "would read them. List them in the manifest files (role 'tokenizer') and record, or remove them")
    cfg = wdir / "tokenizer_config.json"
    if cfg.is_file():
        raw = cfg.read_bytes()                 # parse exactly the bytes that are compared with the pin
        want = pins.get("tokenizer_config.json")
        if want and hashlib.sha256(raw).hexdigest() != want.lower():
            raise EncoderIntegrityError("tokenizer_config.json: SHA-256 differs from the pinned value")
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise EncoderIntegrityError(f"tokenizer_config.json is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise EncoderIntegrityError("tokenizer_config.json must be a JSON object")
        if VERSIONED_TOKENIZER_KEY in data:
            raise EncoderIntegrityError(
                f"tokenizer_config.json contains {VERSIONED_TOKENIZER_KEY!r}, which lets the loader select "
                "a versioned tokenizer file other than the pinned tokenizer.json; refusing")


def _refuse_symlinked_weights(weights_dir) -> None:
    """G2.4: `<weights_dir>/model.safetensors` must not be a symlink, dangling or not, with or without
    manifest hashes. `locate_weights` and `verify_weights` follow symlinks (a link to identical bytes
    elsewhere passes the hash check), and `_check_pins` only looks at files that have a pin, so the weight
    file is checked here explicitly, by the same `is_symlink()` convention as `_check_pins`. Runs first,
    before any hashing and before the model stack is imported. A missing or unconfigured directory is
    left to `locate_weights` to report."""
    if weights_dir is None:
        return
    if (Path(weights_dir) / WEIGHTS).is_symlink():
        raise EncoderIntegrityError(
            f"{WEIGHTS} in {weights_dir} is a symlink; refusing (the weight file must be a regular file "
            "in the model directory; a link, even to identical bytes or to a missing target, is not accepted)")


def _model_side_problems(wdir: Path) -> List[str]:
    """Top-level entries of `wdir` that could change WHAT the loader loads beyond the one pinned
    `model.safetensors`: adapter files, other weight formats/variants/shards and weight index files.
    Matching is case-insensitive and counts symlinks and directories (`lexists`/listing, not `is_file`).
    Only the top level is scanned because the loader is never given `subfolder`."""
    out: List[str] = []
    for n in sorted(os.listdir(wdir)):
        low = n.lower()
        if low.startswith(ADAPTER_PREFIXES):
            out.append(f"{n} (PEFT adapter file; the loader would read the adapter and may follow its "
                       "base_model_name_or_path)")
        elif low.endswith(INDEX_SUFFIX):
            out.append(f"{n} (weight index file; it can redirect loading to extra shards)")
        elif low.endswith(WEIGHT_FORMAT_SUFFIXES) and n != WEIGHTS:
            out.append(f"{n} (weight-format file other than the pinned {WEIGHTS}; only a single-file "
                       "safetensors checkpoint is supported)")
    return out


def _check_model_side_files(wdir: Path) -> None:
    probs = _model_side_problems(wdir)
    if probs:
        raise EncoderIntegrityError(
            f"unpinned model-side file(s) present in {wdir}: {probs}; remove them (single-file "
            f"{WEIGHTS} checkpoints only, no adapters, shards or alternative weight formats)")


def _changed_files(wdir: Path, hashes: Mapping[str, str]) -> List[str]:
    """Files whose current SHA-256 differs from `hashes` (missing, symlinked or unreadable counts)."""
    changed = []
    for fname, want in hashes.items():
        fp = wdir / fname
        try:
            ok = not fp.is_symlink() and fp.is_file() and sha256_file(fp).lower() == want.lower()
        except OSError:
            ok = False
        if not ok:
            changed.append(fname)
    return sorted(changed)


class TorchTextBackend:
    def __init__(self, config: EncoderConfig, spec: TextEncoderSpec, *,
                 file_sha256: Optional[Mapping[str, str]] = None):
        if config.precision != "float32":
            raise EncoderLoadError("only float32 is supported")
        _refuse_symlinked_weights(config.weights_dir)        # G2.4: before anything reads or hashes it
        wfile = locate_weights(config.weights_dir)
        wdir: Path = wfile.parent
        if not (wdir / "tokenizer.json").is_file():
            raise EncoderLoadError(f"tokenizer.json not found in {wdir}")
        weights_sha, warns = verify_weights(wfile, config.expected_weights_sha256,
                                            config.allow_unverified)
        hashes = {WEIGHTS: weights_sha}
        for fname in ("config.json", "tokenizer.json"):
            hashes[fname] = sha256_file(wdir / fname)
        if file_sha256:                           # re-check at load time (files may have changed
            hashes.update(_check_pins(wdir, file_sha256, hashes))     # since the manifest was verified)
        elif config.expected_weights_sha256 is None:
            warns = list(warns) + ["tokenizer_and_config_not_verified_against_expected_values"]
        _check_tokenizer_sidecars(wdir, file_sha256 or {})
        _check_model_side_files(wdir)                         # G2.3: single-file checkpoint contract
        torch, transformers = _import_torch_stack()          # only now is the stack needed
        device = resolve_device(config.device, torch)
        try:
            tok = transformers.AutoTokenizer.from_pretrained(
                str(wdir), local_files_only=True, trust_remote_code=False, use_fast=True)
            model = transformers.AutoModel.from_pretrained(
                str(wdir), local_files_only=True, trust_remote_code=False, use_safetensors=True)
        except Exception as e:
            raise EncoderLoadError(f"failed to load checkpoint from {wdir}: "
                                   f"{type(e).__name__}: {e}") from e
        if not getattr(tok, "is_fast", False):
            raise EncoderLoadError(f"a fast tokenizer (tokenizer.json) is required in {wdir}; "
                                   f"got {type(tok).__name__}")
        changed = _changed_files(wdir, hashes)    # nothing hashed above may have changed while loading
        if changed:
            raise EncoderIntegrityError(
                f"files changed during load (loaded objects discarded): {changed}")
        try:                                      # a sidecar may also have APPEARED while loading
            _check_tokenizer_sidecars(wdir, file_sha256 or {})
        except EncoderIntegrityError as e:
            raise EncoderIntegrityError(
                f"tokenizer sidecar state changed during load (loaded objects discarded): {e}") from e
        try:                                      # ... and so may a model-side file (adapter, shard, index)
            _check_model_side_files(wdir)
        except EncoderIntegrityError as e:
            raise EncoderIntegrityError(
                f"model-side file state changed during load (loaded objects discarded): {e}") from e
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        hidden = getattr(model.config, "hidden_size", None)
        if hidden != spec.expected_dim:
            raise EncoderIntegrityError(
                f"checkpoint hidden_size {hidden!r} != expected {spec.expected_dim} for {spec.name}")
        self._torch, self._tok, self._model = torch, tok, model
        self._device, self._limit = device, spec.max_input_tokens
        self.info = BackendInfo(
            weights_sha256=weights_sha, device=device, precision="float32", dim=spec.expected_dim,
            framework_versions={"torch": torch.__version__, "transformers": transformers.__version__},
            extra={"config_sha256": hashes["config.json"], "tokenizer_sha256": hashes["tokenizer.json"]},
            warnings=tuple(warns))

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        enc = self._tok(list(texts), add_special_tokens=True, truncation=False, padding=False)
        return [len(ids) for ids in enc["input_ids"]]

    def forward(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        enc = self._tok(list(texts), add_special_tokens=True, truncation=False, padding=True,
                        return_tensors="pt")
        longest = int(enc["input_ids"].shape[1])
        if longest > self._limit:             # unreachable via the encoder (it rejects first)
            raise TextTooLongError(f"batch has a {longest}-token sequence (limit {self._limit}); "
                                   "refusing to truncate", token_count=longest, limit=self._limit)
        with torch.inference_mode():
            batch = {k: v.to(self._device) for k, v in enc.items()}
            hidden = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return pooled.detach().to("cpu", dtype=torch.float32).numpy()
