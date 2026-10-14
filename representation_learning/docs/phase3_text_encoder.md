# Phase 3 (G2): text representation runtime

Status: **G2 only.** The text input contract, the encoder interface and one adapter per manifest
candidate exist and are tested **offline with mock components**. No real checkpoint was loaded, no
weights were downloaded, no retrieval-quality evaluation ran, and neither candidate is selected:
the manifest still says `status: candidate`, `final_encoder: false`, revisions `TO_RECORD`.
No Phase 2 file and no G1 file was modified. Cross-modal alignment and G3 are not started.

| File | Purpose |
|---|---|
| `text/contracts.py` | `TextEvidenceItem` + validation rules |
| `text/errors.py` | Reuses the visual error types; adds `TextTooLongError` |
| `text/encoder.py` | `TextEncoder` base, `TextBackend` protocol, `TextEncoderSpec`, `TextRepresentation` |
| `text/adapters.py` | `E5TextEncoder`, `MiniLMTextEncoder` (candidate-specific specs), manifest cross-check, `text_encoder_from_manifest` |
| `text/torch_backend.py` | Lazy real backend (torch + transformers). **Not yet executed against any checkpoint** |
| `testing/text_mocks.py` | Mock backend, `mock_text_encoder`, synthetic pinned-manifest builder |
| `tests/test_text_contract.py`, `test_text_encoder.py`, `test_text_manifest_integration.py` | Offline tests |

## 1. Input contract: `TextEvidenceItem`

The contract is `docs/data_contracts.md` section 3 (unchanged). `TextEvidenceItem.from_dict(d)`
validates a dict against it and returns an immutable item holding deep copies; constructing the
dataclass directly is validated too. Rules **beyond the JSON Schema** (the Python validator may be
stricter than the schema, never looser; a test cross-checks against the schema parsed from the doc):

- `original_text_sha256` must equal SHA-256 of `original_text` as UTF-8, no Unicode/newline
  normalisation (the contract only says "hash of original_text"; this is the assumption made).
  Same for `translation.text_sha256`.
- Texts must not be blank, contain NUL, or be non-UTF-8-encodable. Over 20000 characters is rejected,
  never truncated (the same bound is applied to `translation.text`, which the schema leaves
  unbounded). Text is never trimmed or normalised.
- `evidence_id`: **generated upstream**. The encoder validates form only: exactly `te_` followed by
  16 lowercase hexadecimal characters (`^te_[0-9a-f]{16}$`; G2.1 -- G2 accepted any 8..256-character
  token). The format is stricter than the Phase 1 schema (`minLength: 8`), which is unchanged. The
  encoder does not verify, and does not claim to verify, how the id was derived from
  (`source_id`, `original_text_sha256`). Within one batch, an id used for two different text hashes
  rejects every item of that id.
- `language`: preserved verbatim, non-blank string. **No allow-list and no BCP-47 syntax check**; it
  never changes the model input, the prefix or the vector, and is copied to the record.
- Timestamps must be ISO-8601 with a UTC offset; unknown top-level keys are rejected.
- Raw strings are refused: every encoded text needs an upstream evidence item (no ad-hoc text).
- `nlp_context` and `location` are metadata. They are never concatenated into the encoded text.
- `upstream.nlp_model_version` in the embedding record is always `null` (G2.1): the encoder does not
  read `nlp_context`, so it cannot attribute an upstream NLP model version. `nlp_context` is still
  validated (must be an object) and kept on the item.

## 2. Encoder interface

```python
enc = E5TextEncoder(config)                       # lazy: nothing loaded yet
rep = enc.encode(item, role="query")              # item: TextEvidenceItem or dict
reps = enc.encode_batch(items, role="passage", batch_size=8, on_error="collect")
rep.vector   # float32 (384,), read-only, L2-normalised
rep.record   # Phase 1 EmbeddingRecord (modality "text", text_variant set)
```

`text_variant` is `original_text` (default) or `translation` (only if the item has one); the record
always states which was embedded. Flow per item: validate item -> choose variant -> check role ->
format input -> **count tokens on the full model input** -> reject if over limit -> forward -> check
shape/dtype/finite/non-zero -> L2-normalise -> build record. Failures: `InvalidInputError`
(contract, role, variant, id clash), `TextTooLongError` (subclass of `InvalidInputError`, carries
`token_count` and `limit`), `EmbeddingValidationError` (bad model/tokenizer output),
`EncoderLoadError` / `EncoderIntegrityError` (loading). `encode_batch` collects per-item failures as
`EncodeFailure` (or raises with `on_error="raise"`).

Backends implement `count_tokens(texts) -> [int]` (with special tokens, no truncation) and
`forward(texts) -> (N, dim)`; tests inject `MockTextBackend`.

## 3. Candidate-specific handling (explicit, in `text/adapters.py`)

| | multilingual-e5-small | paraphrase-multilingual-MiniLM-L12-v2 |
|---|---|---|
| Role | **required**: `query` / `passage` | optional; validated and recorded, **no effect** on the input |
| Model input | `"query: " + text` / `"passage: " + text` (trailing space) | `text` |
| Token limit | **512**, prefix + text + special tokens | **128**, text + special tokens |
| Over limit | `TextTooLongError`, never truncated | same |
| Dimension / pooling | 384 / mean, then L2 | 384 / mean, then L2 |

The E5 limit was left open by G1 ("decided in the encoder step"). It is decided **here, in code**, as
512 = the verified `max_position_embeddings`; `adapters.spec_problems` refuses to run if the code
limit ever exceeds the manifest capacity, or disagrees with any manifest value (dimension, pooling,
policy, prefixes, role requirement, 128 for MiniLM). The manifest's `max_input_tokens: null` is left
untouched because G1 files are frozen (see section 6).

`role` is not part of `space_id` (query and passage vectors of one model share a space) but is part of
`embedding_id` through the hash of the exact model input. Where it is stored: `source_ref.input_role`
(the EmbeddingRecord has no first-class role field; see section 6).

## 4. Manifest integration and laziness

`text_encoder_from_manifest(name)` loads the text manifest, runs `verify_entry` (refuses unless the
entry is pinned and every local file verifies; against the committed manifest it therefore always
refuses), cross-checks the code spec, and returns a **lazy** adapter whose config carries the pinned
revision, licence, weights hash and per-file hashes. The real backend re-hashes the weights, config
and tokenizer at load time. Importing any text module imports neither torch nor transformers (tested
in subprocesses); the real backend uses `local_files_only=True`, `trust_remote_code=False`,
`use_safetensors=True`. `conftest.py` fails any test that opens a socket.

## 5. Test strategy and what it proves

All tests are offline, with synthetic data and mocks (`MockTextBackend`: whitespace "tokenizer" + 2
special tokens, seeded random un-normalised vectors, fault injection). They verify: contract
validation and schema cross-check; exact prefixes; verbatim text; boundary behaviour at exactly the
limit and limit+1 (and that over-limit text never reaches the model); output shape/dtype/L2 norm;
determinism; record conformance to the Phase 1 `EmbeddingRecord` schema; no text copy in records;
every error path; code-spec vs manifest agreement and drift detection; manifest integration using
tiny synthetic pinned manifests in temp directories; laziness. They say **nothing** about the
behaviour of the real tokenizers/models or embedding quality.

### Tokenizer sidecars (G2.2)

What the supported loading path can consume, from reading the **transformers 5.18.0** source
(the version in `requirements-phase2.txt`; source only, no model or tokenizer was loaded, so this is
reading, not observed loader behaviour). `AutoTokenizer.from_pretrained(<local dir>, local_files_only=True,
trust_remote_code=False)` reads, beyond `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json` and `config.json` (all already required and pinned):

| File | Read by the loader | Policy in `TorchTextBackend` |
|---|---|---|
| `added_tokens.json` | yes: converted to added tokens when `tokenizer_config.json` has no `added_tokens_decoder` (changes the token table) | present => must be manifest-listed and pinned, else refuse |
| `chat_template.jinja`, `additional_chat_templates/*.jinja` | yes: stored as the tokenizer's chat template (not used by plain tokenization, but read) | present => must be listed and pinned, else refuse; a symlinked template directory is refused |
| a versioned `tokenizer.<version>.json` | only if `tokenizer_config.json` contains `fast_tokenizer_files`, which makes the loader pick that file instead of `tokenizer.json` | a `tokenizer_config.json` with that key is refused (the pinned `tokenizer.json` would no longer be the file consumed) |
| `sentencepiece.bpe.model`, `vocab.txt`, `vocab.json`, `merges.txt`, `spiece.model`, `tokenizer.model`, `tekken.json`, `tiktoken.model` | resolved and passed on as `vocab_file`/`merges_file`, but opened only when `tokenizer.json` is absent (conversion path) | not required to be pinned: the backend requires a pinned `tokenizer.json`, so they are not consumed. Real snapshots contain e.g. `sentencepiece.bpe.model`, which the manifest does not list |

Notes. Listing a sidecar needs no manifest change: add it to the entry's `files` with role `tokenizer`
before `record` (the G1.2 schema already allows extra listed files; a test does exactly this). Without
manifest hashes (`file_sha256` not given) no sidecar can be pinned, so any present sidecar is refused.
Pinned sidecars are hashed before and after loading like all pinned files, and the sidecar check is
repeated after loading, so a sidecar that appears during the load is detected. In transformers 5.18.0
`AutoTokenizer` pops and ignores `use_fast` (always tokenizers-backed); `use_fast=True` stays in the
call as the explicit intent and for older versions, and the `is_fast` check is what enforces the result.
The real loader has not been run (tests use fake stand-ins that record arguments and touch files;
they establish our checks, not Hugging Face behaviour). Model-side files are covered in the next
subsection.

### Model-side files: the single-file checkpoint contract (G2.3)

G2.2 did not analyse model-side files; G2.3 does, again from reading the **transformers 5.18.0**
source only (no model was loaded). What `AutoModel.from_pretrained(<local dir>, use_safetensors=True)`
can pick up from a directory:

| Loader input (5.18.0 source) | Names | Why it matters |
|---|---|---|
| weights and weight indexes (`utils/__init__.py`) | `model.safetensors`, `model.safetensors.index.json`, `pytorch_model.bin`, `pytorch_model.bin.index.json` | an index file makes the loader read the shard files it names |
| variants and shards (`modeling_utils.py`, `_add_variant`, save path) | `model.<variant>.safetensors`, `model.<variant>.safetensors.index.json`, `model-0000N-of-0000M.safetensors` / `.bin` | alternative or additional weights |
| PEFT adapters (`utils/peft_utils.py`, `models/auto/auto_factory.py`) | `adapter_config.json`, `adapter_model.safetensors`, `adapter_model.bin` | `find_adapter_config_file` detects an adapter config, and the loader can read `base_model_name_or_path` from it |

The supported contract is **one weight file, the pinned `model.safetensors`, and no adapter**. In
`TorchTextBackend` (`_model_side_problems`, `_check_model_side_files`) the top level of the model
directory is listed (case-insensitive; symlinks and directories count) and the following are refused
with `EncoderIntegrityError`, even if a pin exists for them:

- adapter files: any name starting `adapter_config` or `adapter_model`
  (so `adapter_config.json`, `adapter_model.safetensors`, `adapter_model.bin` and their variants/shards);
- weight index files: every `*.index.json` (so `pytorch_model.bin.index.json` and variant indexes;
  the shard names inside an index are chosen by the index, so no allow-list is possible);
- every other weight-format file: names ending `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`,
  `.msgpack`, `.h5`, `.onnx`, `.gguf`, `.pkl`, `.pickle`, `.npz`, `.npy`, `.tflite`, `.mlmodel`, `.pb`,
  other than `model.safetensors` itself (stray `pytorch_model.bin`, `model.fp16.safetensors`, shards,
  TF/Flax names, ...).

The suffix rule is deliberately broader than the constants in the source (TF/Flax constants no longer
exist in 5.18.0), so an alternative format is refused rather than silently ignored. The check runs
after pin validation and the tokenizer-sidecar check and **before** the model stack is imported, and
again after `from_pretrained` (a file that appears while loading is reported as
"model-side file state changed during load"). `model.safetensors.index.json` is additionally refused
earlier by the approved Phase 2 `locate_weights` ("sharded checkpoints"); that check follows symlinks
via `exists()`, so a dangling symlink of that name is caught by ours. Other files (`README.md`,
`modules.json`, `generation_config.json`, vocabulary files, ...) are not refused by this check.

#### The weight file itself must not be a symlink (G2.4)

`<weights_dir>/model.safetensors` that is a symlink, **including a dangling one**, is refused with
`EncoderIntegrityError` ("is a symlink"), with or without manifest hashes, as the first step of
`TorchTextBackend.__init__` (`_refuse_symlinked_weights`): before the weights are located or hashed and
before the model stack is imported. Why it was needed: `locate_weights` and `verify_weights` follow
symlinks, so a link to identical bytes elsewhere satisfies `expected_weights_sha256`, and `_check_pins`
(which does reject symlinks) only runs when manifest hashes were supplied. A dangling link was already
refused, but as a generic "not found"; it now gets the explicit message. The post-load hash comparison
is unchanged and also detects the weight file being replaced by a symlink while loading. Scope: only the
weight file is checked; a symlinked weights directory is unchanged behaviour. As before this detects
state at check time and does not prevent a swap between the check and the load.

Remaining limitations. Only the top level is scanned (the loader is never given `subfolder`; files in
`onnx/`, `openvino/`, ... are not consumed, and are not checked). The post-load check detects a file
that is still present after loading; one that appears and is removed within the load window cannot be
detected. The list was derived from source, not from a real load; real snapshots were not inspected and
the real loader was not run, so behaviour for a real checkpoint is unconfirmed. The G1 manifest layer is
unchanged: it already refuses stray weight files and `model.safetensors.index.json` at `record`/`verify`,
but not `adapter_config.json` or other `*.index.json`, which only this backend check covers.

## 6. Remaining real-checkpoint validation (not done; needs separate approval)

1. Obtain both checkpoints under the existing manifest rules, then `record` the pins (revision,
   licence text, hashes). Not done here; nothing was downloaded.
2. Run `TorchTextBackend` against the real checkpoints. It has **never been executed** against a real
   or tiny-random checkpoint (G2.1 tests drive it only with fake torch/transformers stand-ins that
   record the `from_pretrained` arguments). Load-time integrity (G2.1): every manifest-listed file is
   hashed and compared with its pin before loading, files that can affect loading
   (`REQUIRED_FILES`) must be pinned, the tokenizer is loaded with `use_fast=True` and must be a fast
   tokenizer, and the pinned files are hashed again after loading (a change during the load window is
   detected, not prevented; a change reverted before the second hash is not detectable). Confirm: tokenizer/model load locally, `hidden_size` 384, pooling equals the
   Sentence-Transformers mean pooling, E5 output equals the checkpoint's own pipeline (pooling +
   Normalize) up to tolerance, MiniLM likewise.
3. Confirm that `count_tokens` (special tokens, no truncation) equals the tokenizer length the
   models' own pipelines apply their limits to: 512 for E5 (also that a 512-token sequence runs) and
   128 for MiniLM; confirm that prefix text such as `"query: "` tokenizes as assumed.
4. Check multilingual behaviour on real text (scripts, normalisation forms, very short/long inputs),
   padding invariance (batch vs single vectors within tolerance) and CPU/GPU parity.
5. Decide `max_input_tokens` for E5 in the manifest (G1 amendment) and whether a role field should
   exist on `EmbeddingRecord` (contract amendment).
6. Retrieval-quality evaluation and model selection: separate approval; not started.

## 7. Known limitations and open decisions for review

- The stale G1 statements ("no encoder ... implemented") in `text/__init__.py` and
  `docs/phase3_text_manifest.md`, and Phase 2 statements in `__init__.py`, `README.md` and
  `docs/phase2_encoders.md`, were not edited (frozen files).
- `pipeline_version` in records reads `0.2.0-phase2` because `__init__.py` is frozen; use
  `EncoderConfig.pipeline_version` to override.
- Assumptions flagged above: UTF-8/no-normalisation for text hashes; id length bound 256; role stored
  in `source_ref`; E5 limit 512 decided in code; whitespace-only text rejected.
