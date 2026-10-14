# Model weights: pinning, local paths, verification, real runs

**Status: PENDING.** No real weights were available to Phase 2 (the build environment could not
reach Hugging Face). Nothing below has been executed against real checkpoints. The manifest
(`models.manifest.json`) is a committed template with every pin set to `TO_RECORD`; the code
refuses to use an unpinned entry.

Weights are never downloaded by this code or its tests and never committed
(`representation_learning/.gitignore` excludes `models/`, `*.safetensors`, `*.bin`, `*.pt`).

## 1. What gets pinned (per checkpoint)

| Field | DINOv2-small | CLIP ViT-B/32 |
|---|---|---|
| Manifest name | `dinov2-small` | `clip-vit-b-32` |
| Source | https://huggingface.co/facebook/dinov2-small | https://huggingface.co/openai/clip-vit-base-patch32 |
| Revision (40-hex commit) | `TO_RECORD` | `TO_RECORD` |
| Licence (text from the model card) | `TO_RECORD` (expected Apache-2.0, **unverified**) | `TO_RECORD` (expected MIT, **unverified**) |
| Local path (git-ignored) | `representation_learning/models/dinov2-small/` | `representation_learning/models/clip-vit-base-patch32/` |
| Files hashed | `model.safetensors`, `config.json`, `preprocessor_config.json` | same |
| Output dim (checked on load) | 384 | 512 |
| Why this variant | Resource-based CPU baseline. **Not a quality ranking.** `dinov2-base` (768) stays selectable. | same |

The expected licences are from memory and are not authoritative; you record what the model card
actually says. Only a single `model.safetensors` is accepted (no pickle `.bin`, no shards). That
these repositories ship such a file under that name is **unverified**; if not, the loader refuses
and you must choose or convert a compatible file and record that provenance.

## 2. Exact commands (CPU, run from the repository root)

```bash
# 0. CPU install (see requirements-phase2.txt; the +cpu wheel index was unreachable in my sandbox)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r representation_learning/requirements-phase2.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect 2.14.1+cpu False

# 1. Choose and write down the exact commit of each checkpoint (40 hex characters)
python -c "from huggingface_hub import HfApi as A; print('dinov2-small', A().model_info('facebook/dinov2-small').sha); print('clip-b32', A().model_info('openai/clip-vit-base-patch32').sha)"
export DINO_REV=<40-hex printed above>
export CLIP_REV=<40-hex printed above>

# 2. Download exactly those revisions (manual step; flags confirmed in `hf download --help`, hf CLI 1.33.0)
hf download facebook/dinov2-small config.json model.safetensors preprocessor_config.json \
    --revision "$DINO_REV" --local-dir representation_learning/models/dinov2-small
hf download openai/clip-vit-base-patch32 config.json model.safetensors preprocessor_config.json \
    --revision "$CLIP_REV" --local-dir representation_learning/models/clip-vit-base-patch32

# 3. Read each model card, then record revision, licence text and SHA-256 of the three files
python -m representation_learning.encoders.manifest record dinov2-small --revision "$DINO_REV" --license "<licence text from the model card>"
python -m representation_learning.encoders.manifest record clip-vit-b-32 --revision "$CLIP_REV" --license "<licence text from the model card>"

# 4. Check pins and local files (every file must match its recorded SHA-256)
python -m representation_learning.encoders.manifest status
python -m representation_learning.encoders.manifest verify

# 5. Real-checkpoint tests (CPU). Expected: 8 passed, 0 skipped.
python -m pytest representation_learning/tests/test_real_checkpoints.py -v -rs

# 6. Real-model smoke test on the committed video (weapon/fire/smoke crops only), one encoder per
#    process so the memory figures are not mixed. Whole frames are not used.
python -m representation_learning.evaluation.run_eval --source video \
  --video input/YOUTUBE_YouTubeCCTV196_gun_2.mp4 \
  --report output/YOUTUBE_YouTubeCCTV196_gun_2_report.json \
  --manifest representation_learning/models.manifest.json --encoders dinov2 --device cpu \
  --output representation_learning/evaluation/runs/real_video_smoke_dinov2-small.json
python -m representation_learning.evaluation.run_eval --source video \
  --video input/YOUTUBE_YouTubeCCTV196_gun_2.mp4 \
  --report output/YOUTUBE_YouTubeCCTV196_gun_2_report.json \
  --manifest representation_learning/models.manifest.json --encoders clip --device cpu \
  --output representation_learning/evaluation/runs/real_video_smoke_clip-vit-b-32.json
```

`evaluation/runs/` is git-ignored. Optional diagnostic only: add `--include-diagnostic-frames` to
also encode whole frames as a separately labelled pipeline diagnostic.

## 3. How to read the outcome

- **A skipped real-checkpoint test is not a pass.** If step 5 reports skips, the manifest is not
  pinned or the files are missing/mismatched and Phase 2 is still not validated.
- A load failure raises `EncoderIntegrityError` (hash/model-type/dimension/preprocessing mismatch)
  or `EncoderLoadError`. Fix the cause; do not use `allow_unverified`.
- If the checkpoint's `preprocessor_config.json` differs from the preprocessing implemented here,
  loading refuses with the exact differences. That is a finding to report, not something to bypass.
- The smoke-run JSON states `real_pretrained_inference_executed: true`, `representation_quality_evaluated: false`
  and `winner_declared: false`. Report the numbers; do not rank the encoders from them.

## 4. Safety properties

Local directories only (`local_files_only=True`); `safetensors` only; every file hash checked
against the manifest before loading; no network access is attempted by the library; the manifest
holds no secrets. Re-check licences before any redistribution of results or models.
