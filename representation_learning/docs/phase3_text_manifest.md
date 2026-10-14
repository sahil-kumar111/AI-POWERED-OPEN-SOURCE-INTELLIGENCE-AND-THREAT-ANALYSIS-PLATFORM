# Phase 3 (G1): text-model manifest

Status: **G1 only.** A separate manifest and its validation exist. No text encoder, tokenizer
loading or inference is implemented, no weights were downloaded, and no real-checkpoint validation
has run. Both candidates stay `status: candidate`, `final_encoder: false`.

Files (all new; no Phase 2 file was modified):

| File | Purpose |
|---|---|
| `text_models.manifest.json` | Two text candidates, unpinned |
| `text/manifest.py` | `status`, `record`, `verify`, validation (no torch, no network) |
| `tests/test_text_manifest.py` | Manifest tests with tiny synthetic local files |
| `tests/test_phase2_unchanged.py`, `tests/phase2_baseline.sha256` | Guard: approved Phase 2 files stay byte-identical |

## Where each fact comes from

- **Hugging Face metadata (G0, user-supplied output)**: commit SHA, licence identifier, file sizes
  and the LFS SHA-256 of `model.safetensors` and `tokenizer.json`. Small files have no upstream
  SHA-256 on the Hub and carry `null`; none were invented.
- **Stated by the project owner for G1** (configuration-verified by the owner; the contents of the
  config files were not shown to the implementer): dimension 384, e5 `max_position_embeddings`
  512, mean pooling, e5 normalization module present, e5 `query: ` / `passage: ` prefixes (the
  exact whitespace follows the E5 convention and is locked by a test), MiniLM 128-token limit.
  `record` re-checks what it can against the real local files (hidden size, max positions,
  mean-pooling flags, `max_seq_length`, safetensors container, embedding width) and refuses on any
  mismatch. That code path has only been exercised with synthetic files.
- **Not yet decided**: e5 `max_input_tokens` (left `null`; shown as "design open"). Model type and
  tokenizer class are recorded as observed values at `record` time, not asserted.

## Commands (from the repository root)

```bash
python -m representation_learning.text.manifest status
python -m representation_learning.text.manifest record <name> --revision <40-hex> \
       --license-id <id> --license "<licence text from the model card>"
python -m representation_learning.text.manifest verify [name ...] [--manifest-sha256 <hex>]
```

`record` refuses unless: the revision equals the verified `expected_revision`; the licence id equals
the verified id; every listed file exists as a regular file with the expected size; every file with
an upstream SHA-256 matches it; no other weight/pickle format is present (only `model.safetensors`);
and the local configuration agrees with the design facts. Nothing is written on failure, and an
existing pin is never replaced by different hashes. `final_encoder` can only be `false`; selecting
an encoder is a separate decision.
