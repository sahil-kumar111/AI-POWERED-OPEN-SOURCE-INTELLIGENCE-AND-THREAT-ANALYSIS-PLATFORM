# Phase roadmap, dependencies, hardware and risks

Status: Phase 1 (this document set) is design only. Later phases are plans;
none has been started and each requires explicit approval.

## 1. Phases

| Phase | Goal | Exit criteria | Gate |
|---|---|---|---|
| 1 | Design and architecture docs (done when approved) | Docs reviewed; open questions in section 6 answered or accepted | User approval |
| 2 | Visual encoder evaluation: DINOv2 vs CLIP image tower on this project's crops/frames | Visual adapter + selector + encoder wrapper with contract and unit tests; comparison report with identical crops, metrics and settings; one visual `space_id` chosen or "none adequate" | Approval after report |
| 3 | Text representation: define the `TextEvidenceItem` adapter against whatever OSINT/NLP interface exists by then; choose text encoder | Adapter contract tests; text retrieval sanity results | Approval |
| 4 | Comparison/alignment: native joint space check (CLIP) and/or learned alignment; build paired validation set | `AlignmentArtifact` with `status: passed` against section 6 protocol of `architecture.md`, or a documented decision to stay symbolic for cross-modal | Approval |
| 5 | Registry/index and event-level aggregation; integration with Event Manager (if it exists) | Store, similarity API, provenance enforcement, event grouping tests | Approval |
| 6 | Hand-off to fusion/explainability layers; privacy review; performance pass | Documented interface; privacy checklist signed off | Approval |

Phases 3 and 4 can be reordered if the text pipeline matures later than the
visual side; Phase 4 cannot precede both encoders being chosen.

## 2. Proposed directory layout (future, not created)

```text
representation_learning/
  README.md
  docs/
  contracts/        # dataclasses / JSON Schemas (Phase 2+)
  adapters/         # pipeline-report adapter, future Event Manager / NLP adapters
  encoders/         # one wrapper per encoder, hash-verified weights
  registry/         # EmbeddingRecord storage, space registry
  compare/          # same-space similarity, alignment gate
  tests/
```

Use a distinct top-level package name. Existing modules each define their own
top-level `backend` package (`backend/app/interfaces`), so reusing `backend`
here would collide on import.

## 3. Dependencies

Phase 1 adds **none**.

Known from the repo: `ultralytics`, `opencv-python` (`cv2`), `numpy`
(imported by `threat_video_pipeline.py`); `pytest` is used per module. There is
no requirements file, so versions are unpinned and unknown.

Likely in Phase 2 (to be decided, not committed): PyTorch (already a transitive
dependency of `ultralytics`), and one loader for the candidate encoders (for
example Hugging Face `transformers`, `open_clip`, or `torch.hub`). Selection
criteria: pinned versions, licence compatibility, weights available as
verifiable files, no code execution on load. Vector search (FAISS or similar)
and a database are Phase 5 decisions; the diagram mentions MongoDB but no
database code exists in the repo.

## 4. Hardware constraints

The target hardware is **not stated** anywhere in the repo and has not been
provided; it must be confirmed (open question 1). Facts that bear on it:

- Existing detectors are nano-size YOLO checkpoints (`yolo11n`, `best_nano_111.pt`), and the pipeline supports `frame_skip`, so detection already shares compute with any encoder run on the same machine.
- Encoder cost is per crop/frame, not per video frame, if the selector samples sparsely. Sampling policy is the main lever.
- Approximate figures to **verify in Phase 2** (from general knowledge of the published models, not measured here): DINOv2 backbones range from roughly 20M parameters (ViT-S) to over 1B (ViT-g) with embedding sizes of about 384 to 1536; common CLIP ViT-B variants are roughly 150M parameters in total with 512-dim embeddings. A 768-dim float32 vector is about 3 KB, so storage is minor next to any decision to retain pixels.
- Plan to benchmark on the actual machine: latency per crop, peak memory, CPU-only vs GPU, batch size 1 vs N, float16 impact on embedding stability.
- Offline operation is assumed (no implicit downloads); weights must be placed locally and hash-verified.

## 5. Risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| R1 | Text side has no upstream interface (no OSINT/NLP code) | Certain / high for cross-modal goals | Contract-first `TextEvidenceItem`; do not block visual work |
| R2 | Event Manager does not exist; "event" has no authoritative definition | Certain / medium | `interim_grouping` flag; adapter swap when Event Manager lands |
| R3 | Pipeline report omits person crops, video hash, model hash | Certain / medium | Adapter recomputes video hash; future read-only hook for persons; do not edit pipeline in this project |
| R4 | Annotated video accidentally encoded (boxes/banner burned in) | Medium / high | Schema rejects `_analyzed.mp4`; adapter reads original by frame index |
| R5 | Cross-modal retrieval looks plausible but is meaningless or biased | High / high | Alignment gate; baselines; refuse by default |
| R6 | Domain shift (CCTV, low light, small objects, 320px weapon inference) hurts encoder quality | High / medium | Quality fields, min-crop-size filter, per-domain evaluation in Phase 2 |
| R7 | No paired image-text validation data exists; the weapon dataset (`weapon.v2i.yolov11`) is detection data, not paired text | Certain / high for Phase 4 | Budget time to build/source a small labelled paired set; consider symbolic matching as fallback |
| R8 | Small or biased evaluation data gives false confidence in encoder choice | High / medium | Report sample sizes and uncertainty; avoid single-metric decisions |
| R9 | Privacy: persons in crops, personal data in OSINT text | Certain / high | Pointer-not-pixels default, retention limits, no re-identification, access control |
| R10 | Model supply chain / licence issues | Medium / medium | Hash-verify, licence check recorded in `encoder.license` |
| R11 | Encoder or preprocessing change silently invalidates stored vectors | Medium / high | `space_id` includes weights hash and preprocessing; regression golden vectors |
| R12 | Unpinned dependencies; no manifest in repo | Certain / medium | Pin versions when Phase 2 introduces them, documented separately |
| R13 | Inconsistent configs/types across entry points (pipeline `imgsz=320` vs `WeaponDetector` 640; `WeaponDetection` vs `Detection`; duplicated `backend` packages) | Certain / low-medium | Record effective settings in provenance; adapters normalise types |
| R14 | Stale docstrings in `smoke_detector.py` (class ids described opposite to code) could mislead future work | Certain / low | Code treated as authoritative; report to repo owner; not modified here |
| R15 | Scope creep into fusion/scoring | Medium / medium | Non-goals; interface to fusion defined only in Phase 6 |

## 6. Decisions recorded after Phase 1 review

| # | Decision (from the project owner) |
|---|---|
| D1 | Hardware: CPU-only for now, no hard latency target; keep encoders device-configurable; record hardware, time per input, batch size, memory; no real-time claims. |
| D2 | Evaluation data: the committed video is a pipeline smoke test only. Without additional labelled data no encoder is declared better. |
| D3 | A separate pinned Phase 2 requirements note is allowed; weights stay local and out of Git; tests never download. |
| D4 | Person crops/embeddings excluded; no face recognition, person identification or person-level tracking. |
| D5 | No event grouping, temporal aggregation, event IDs or threat scoring in Phase 2. |
| D6 | Future text must stay multilingual-compatible; original text and language preserved, translation kept separate. |
| D7 | Assume no paired image-text dataset. Phase 4 needs a small curated labelled paired set if feasible; otherwise symbolic matching remains the fallback for cross-modal needs. |
| D8 | Initial baseline: DINOv2-small and CLIP ViT-B/32 (resource-based, not a quality ranking); variants stay configurable (DINOv2-base later). |
| D9 | Whole CCTV frames are not embedded as target-object inputs (diagnostics only); weapon/fire/smoke crops with provenance only; no pixel retention in records. |
| D10 | Owner downloads weights locally; revision/source/licence/path/SHA-256 pinned in `models.manifest.json`; real-model validation stays PENDING until run. |

Open questions 1, 2, 3, 4 (partly), 6 and 8 of the Phase 1 list are answered or superseded
by these decisions. Remaining open items are listed in `phase2_encoders.md` section 9.

## 6b. Original Phase 1 open questions (kept for history)

1. What hardware will Phase 2 run on (CPU only, GPU model and VRAM)? Is there a latency budget?
2. Which video should be the Phase 2 evaluation set? Only one video is committed under `input/`, plus the report/annotated outputs; is more data available?
3. Is it acceptable to add a small `requirements` note for Phase 2 (outside existing modules), and may weights be placed locally by you?
4. Will OSINT/NLP code exist before Phase 3, and in what format will its output be? If not, should Phase 3 use manually curated text samples?
5. Languages: will OSINT text be English only, or multilingual (relevant to text encoder choice)?
6. Should person crops be in scope for encoding at all, given privacy risk, or restrict Phase 2 to weapon/fire/smoke crops and whole frames?
7. Preferred "event" definition until an Event Manager exists (single frame, time window, per-track)?
8. Is there any source of paired visual-text data for Phase 4, or should the project plan to create one?
9. Should the repo owner be told about the `smoke_detector.py` docstring inconsistency (separately from this work)?

## 7. Explicit non-goals

Same as `../README.md` section 5 and `architecture.md` section 10. In
particular: no Phase 2+ work begins without approval.
