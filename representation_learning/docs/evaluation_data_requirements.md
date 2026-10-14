# Evaluation: what Phase 2 can and cannot show

## 1. Two different things (kept separate)

| | Pipeline smoke test (done) | Representation-quality evaluation (NOT done) |
|---|---|---|
| Question | Does the code path run and produce valid vectors? | Are the vectors useful for this project's purpose? |
| Needs labels | No | Yes |
| Data used | Synthetic fixtures; weapon/fire/smoke crops from the one committed video via its existing report | A labelled, representative evaluation set (below) |
| May conclude | Extraction succeeds/fails, dim, finiteness, unit norm, determinism, timing | Retrieval/grouping quality, robustness, domain fit |
| Can pick an encoder | **No** | Only with pre-registered criteria |
| Status | Real-model run **pending** | Not possible yet |

## 2. Measurements possible without labels (implemented in `evaluation/measure.py`)

Extraction success/failure and reasons; output dimension and dtype; finite-value and unit-norm
checks; repeatability (identical input repeated) and batch-vs-single agreement; end-to-end
latency per input with a preprocessing/forward split and per-batch-size timing; load time and
process RSS (CUDA peak if on GPU); hardware context; a descriptive within-encoder spread of
pairwise cosines (explicitly **not** a quality metric and not comparable across encoders).

Not possible without labels: any claim of semantic quality, which encoder is better, whether
embeddings separate weapon/fire/smoke from background, robustness to blur/compression/low
light, or generalisation beyond the one video.

## 3. Requirements for a future labelled evaluation set (documented; the set does not exist yet)

**Unit of evaluation: target-object crops** (weapon, fire, smoke) cut with the existing detectors'
bounding boxes or by hand, each carrying Phase 1 `VisualEvidenceItem` provenance (original video hash,
frame index, bbox, detector settings). Whole frames are not evaluation inputs (diagnostics only). **No
person crops, no faces, no identity labels.**

| Requirement | Detail |
|---|---|
| Independent sources | Crops from **multiple independent videos/sources** (different cameras, scenes, lighting, compression, resolution). One video is never enough. The number of sources and crops per class is fixed by the project owner **before** any run and recorded with the set; this document promises no size. |
| Classes | At least weapon, fire, smoke. Sub-labels where available (weapon type; smoke vs steam/fog/dust). |
| Hard negatives | Look-alike non-targets for each class (e.g. tools/phones vs handguns, lights/sunsets/flames-on-screens vs fire, steam/fog/exhaust/dust vs smoke), plus ordinary background crops. Labelled as negatives with the confusable type. |
| Source-level separation | Train/validation/test split **by source (video/camera), not by frame or crop**, so near-duplicate frames never straddle the split. Split recorded in `labels.csv` and fixed before any run. Phase 2 trains nothing; the split matters for later learned components and for honest retrieval evaluation. |
| Leakage control | Keep detector training data (`weapon.v2i.yolov11`) out unless overlap is analysed and reported. |
| Quality flags | Per crop: small size, blur, occlusion, low light, compression. Report results per flag. |
| Labelling | Human labels; a second reviewer on a sample to estimate agreement; ambiguous items marked, not forced. |
| Rights and privacy | Confirm redistribution rights; avoid identifiable people; do not retain pixels in embedding records. |

## 4. Proposed criteria (to be fixed before any run, then not changed)

Within-encoder tasks on held-out *sources*: label-based retrieval recall@k and class separability,
including hard-negative confusion; robustness deltas under fixed perturbations (resize, JPEG, blur);
runtime and memory on the target CPU as secondary criteria. Cross-encoder comparison uses the same items
and the same metrics, never direct comparison of DINOv2 and CLIP vectors. A winner requires a
pre-agreed margin with uncertainty reported. (Retrieval code is a later phase; Phase 2 has none.)

## 5. Data format and commands

Layout (labels are **not read in Phase 2**; they are for the later evaluation phase):

```text
eval_set/
  crops/       <source_id>__<item_id>.png       # one target-object crop per file, no whole frames
  labels.csv   path,label,hard_negative_of,source_id,split,quality_flags
```

Real-model smoke run on the committed video (pinned weights; crops only; commands and setup in
`weights_and_models.md` section 2): `run_eval --source video ... --manifest representation_learning/models.manifest.json`.

Measurement-only run on a folder of crops:

```bash
python -m representation_learning.evaluation.run_eval --source images --images eval_set/crops \
  --manifest representation_learning/models.manifest.json --encoders dinov2 --device cpu \
  --output representation_learning/evaluation/runs/real_crops_dinov2-small.json
```

Code-path check without any weights (tiny random checkpoints, **not pretrained**): add `--tiny-random`.

## 6. Results so far

`../evaluation/reports/` holds two JSON reports produced with **tiny random checkpoints** (code-path
checks only). No result from DINOv2 or CLIP exists yet; real-model results are **pending** the
steps in `weights_and_models.md`. Nothing here says which encoder is better or good.
