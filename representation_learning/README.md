# representation_learning (Phase 1 design + Phase 2 visual encoder baseline)

Multimodal representation learning for the AI-Powered OSINT and Threat Analysis
Platform (project **Option C**).

> **Status: Phase 2 implemented (visual encoders: DINOv2 + CLIP image encoders and a
> smoke-test harness). Phases 3+ are not started.**
> Phase 2 code is tested with mocks and tiny *random* checkpoints only. **PENDING:** pinning
> the real weights, the real-checkpoint tests and the real-model smoke run (commands in
> `docs/weights_and_models.md`); until they are executed Phase 2 is not fully validated and **no
> real DINOv2/CLIP inference has been run**. Initial CPU baseline: DINOv2-small + CLIP ViT-B/32
> (resource-based, not a quality ranking). No encoder is declared better, and nothing here is
> integrated into the live video pipeline.
> The text encoder, alignment, search and event components described in the Phase 1
> design remain *proposed*; see `docs/phase2_encoders.md` for what exists.

## 1. Purpose

Represent visual evidence from the video pipeline and textual intelligence from
OSINT/NLP in embedding spaces so the platform can eventually support:

- similarity search within a modality (frame/crop to frame/crop, text to text),
- cross-modal retrieval (text to visual evidence and back), **only after a
  validated comparison method exists** (see `docs/architecture.md` section 6),
- event-level intelligence (grouping evidence into events with traceable
  provenance).

## 2. Scope of Phase 1

| In scope | Out of scope |
|---|---|
| Inspect repository, record what actually exists | Any encoder (DINOv2, CLIP, text) implementation or evaluation |
| Architecture, modality boundaries, contracts | Downloading weights, adding dependencies |
| Metadata/provenance requirements | Embedding storage, vector index, API endpoints |
| Risks, privacy, testing strategy, roadmap | Changes to any existing module |

## 3. What the repository actually contains (inspected at commit `f258199`)

| Area | Finding | Evidence (path) |
|---|---|---|
| Combined video pipeline | **Exists.** Weapon + fire/smoke + person detection, writes an annotated video and a JSON report. | `threat_video_pipeline.py` |
| Weapon detector | **Exists.** Hash-verified YOLO11n wrapper. Returns its own `WeaponDetection` (with `class_id`), not the shared `Detection`. | `weapon_detection_project/weapon_detector.py` |
| Person detector | **Exists.** Implements `VideoDetector`, returns shared `Detection`. | `person_detection/person_detection/backend/app/modules/video/person_detector.py` |
| Fire / smoke detectors | **Exist** as separate wrappers; return shared `Detection`. | `fire_smoke_detection/fire_smoke_detection/backend/app/modules/video/{fire,smoke}_detector.py` |
| Crowd counter | **Exists** as a pure per-frame count over `List[Detection]`; **not used** by `threat_video_pipeline.py`. | `crowd_counter/crowd_counter/backend/app/modules/video/crowd_counter.py` |
| ByteTrack adapter | **Exists** (`Tracker` contract + per-class Ultralytics BYTETracker). **Not used** by `threat_video_pipeline.py`, which calls Ultralytics `model.track(..., tracker="bytetrack.yaml")` for persons only. | `bytetrack/backend/app/modules/video/byte_tracker.py` |
| Shared contracts | `Detection(bbox, class_name, confidence)`, `VideoDetector`, `Tracker`, `TrackedDetection` | `*/backend/app/interfaces/` (duplicated per module) |
| Event Manager | **Not found.** Appears in the architecture diagram only. The pipeline's `threat_events` list is per-frame, not an aggregated event. | - |
| OSINT collectors, NLP pipeline | **Not found** (no GDELT/NewsAPI/RSS/spaCy/transformers code). Appear in README and diagram only. | - |
| XAI / SHAP, FastAPI, MongoDB, Streamlit/React | **Not found** in code. | - |
| Dependency manifest | **Not found** (no requirements/pyproject). Pipeline imports `ultralytics`, `cv2`, `numpy`. | `threat_video_pipeline.py` |
| Docs convention | Single root `README.md`; no `docs/` directory. This module introduces `docs/` locally. | `README.md` |

Consequence: the visual side can be fed today from the pipeline's JSON report
(with limits, see `docs/data_contracts.md` section 2). The text side has **no
upstream interface**; it is specified as a future adapter.

## 4. Documents

Phase 2 additions: [`docs/phase2_encoders.md`](docs/phase2_encoders.md) (implementation, dims,
preprocessing, limitations, open decisions), [`docs/weights_and_models.md`](docs/weights_and_models.md)
(sources, licences, SHA-256 workflow), [`docs/evaluation_data_requirements.md`](docs/evaluation_data_requirements.md)
(what can and cannot be concluded, labelled-set requirements). Tests:
`python -m pytest representation_learning/tests -q` (from the repo root). Optional installs:
`requirements-phase2.txt` (CPU install commands inside). Weights manifest: `models.manifest.json`.

Phase 1 documents:

- [`docs/architecture.md`](docs/architecture.md) - layers, diagrams, modality boundaries, comparison/alignment policy, errors, privacy, testing
- [`docs/data_contracts.md`](docs/data_contracts.md) - input/output schemas, embedding metadata, provenance
- [`docs/phase_roadmap.md`](docs/phase_roadmap.md) - phases, dependencies, hardware, risks, open questions

## 5. Explicit non-goals (all phases unless a later phase says otherwise)

- No retraining, fine-tuning, or replacement of existing detectors.
- No edits to existing weapon, person, fire/smoke, crowd, ByteTrack, Event Manager, XAI, or OSINT/NLP code.
- No face recognition or person re-identification.
- No claim that vectors from different encoders, or from different modalities, are comparable because they are vectors.
- No person crops/embeddings, event grouping, text encoder or cross-modal alignment in Phase 2.
- No threat score or classification is produced by this module (the platform's fusion/ATI layer is a separate concern).
