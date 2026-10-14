# Architecture Decision Record: Cross-Modal Alignment (Gate 3)

**Status:** APPROVED DECISION — OPTION C (Separate Representation Spaces & Symbolic / Event-Level Fusion)  
**Date:** 2026-10-13  
**Module:** `representation_learning`  
**Related Documents:** [`architecture.md`](architecture.md), [`phase_roadmap.md`](phase_roadmap.md), [`data_contracts.md`](data_contracts.md)

---

## 1. Context & Architectural Policy

AEGIS requires representing both visual intelligence (CCTV video streams, bounding box crops, detector outputs) and textual intelligence (OSINT reports, intelligence advisories, normalized video evidence descriptions).

Per `architecture.md` Section 6 ("Comparison and alignment policy"):
> *"Two vectors are comparable only if they live in the same space AND that space is meaningful for the comparison being made. Being numeric arrays of equal length is not sufficient."*

Gate 3 requires deciding among three fundamental architectural paths:
- **Option A:** The visual and text encoders produce a compatible native joint space, supported by verified evidence.
- **Option B:** A learned alignment method (e.g., cross-modal projection layer or contrastive fine-tuning) is trained and validated.
- **Option C:** Retain separate representation spaces and use symbolic/event-level fusion instead.

---

## 2. Evaluation of Candidates

### Option A: Native Joint Space — REJECTED
- **Dimension Fallacy:** Both `multilingual-e5-small` and `dinov2-small` output 384-dimensional float32 L2-normalized vectors. However, having equal dimensionality ($d = 384$) does NOT mean the coordinates reside in a common semantic manifold. DINOv2 was trained via self-supervised vision objectives (DINO/iBOT on images); E5 was trained via weakly-supervised text contrastive learning (InfoNCE on text pairs).
- **No Joint Tower:** Neither candidate text encoder (`multilingual-e5-small` nor `paraphrase-multilingual-MiniLM-L12-v2`) possesses a native image encoder tower.
- **Visual Encoder State:** In `models.manifest.json`, visual models (`dinov2-small`, `clip-vit-b-32`) remain unrecorded (`TO_RECORD`). As recorded in `outputs/aegis_evidence_metadata.json`, `visual_embeddings_generated: false`.
- **Conclusion:** Option A is mathematically and empirically false. Direct cosine similarity across modalities is noise.

### Option B: Learned Alignment Method — DEFERRED (NOT AUTHORIZED)
- **Data Prerequisite:** Per `architecture.md` Section 6 and `phase_roadmap.md` Decision D7, learned alignment requires a curated, human-verified paired dataset of visual crops and corresponding textual intelligence, split by independent video source to prevent data leakage.
- **Current Repository Inventory:** Zero paired visual-text datasets exist in the repository. The detector training data (`weapon.v2i.yolov11`) provides bounding boxes, not natural language descriptions.
- **Non-Negotiable Rule 4 & 8:** Fabricating synthetic paired vectors or claiming a trained alignment without genuine paired ground truth violates project integrity.
- **Conclusion:** Option B cannot be completed honestly at this stage and is deferred until a validated paired dataset is formally created.

### Option C: Separate Spaces with Symbolic / Event-Level Fusion — APPROVED
- **Strict Vector Isolation:** Text embeddings reside exclusively in `space_e8d2691a145d41eb`. Visual embeddings, once generated, will reside in their respective visual `space_id`.
- **Refusal Contract:** Any cross-modal comparison attempted directly in vector space without an approved `AlignmentArtifact` raises `IncompatibleSpaceError` as designed in `architecture.md` Section 7.
- **Symbolic Linkage:** Cross-modal association is achieved through structured metadata attributes:
  1. `video_hash` (SHA-256 integrity of the source video)
  2. `timestamp` / `first_seen_sec` / `last_seen_sec`
  3. `bounding_box` and detector classification (`category`, `class_name`)
  4. `event_id` / `source_report`
- **Auditability:** Avoids hallucinations and deceptive similarity scores. Fusion layers receive auditable symbolic facts rather than uncalibrated cross-modal cosine distances.

---

## 3. Decision Summary & Governance

| Gate / Requirement | Status | Technical Action |
|---|---|---|
| **Joint Vector Space** | Disabled | `IncompatibleSpaceError` raised on cross-modal vector comparison |
| **Cross-Modal Alignment** | Deferred | Awaits human-curated paired dataset and project owner authorization |
| **Cross-Modal Association** | Active | Symbolic & metadata matching via report provenance and event temporal windows |
| **Manifest State** | Preserved | No unverified models or alignment weights introduced |

---

## 4. Acceptance Criteria for Future Alignment (Phase 4 Exit Gate)

Before Option B can be reconsidered:
1. Sourcing or annotation of at least 100 paired visual crops and verified multilingual intelligence texts across $\ge 5$ independent video sources.
2. Source-level train/validation/test split without temporal frame overlap.
3. Bidirectional retrieval validation (Image $\to$ Text and Text $\to$ Image) achieving Recall@5 $\ge 0.70$ over a random/shuffled pair baseline.
4. Verification against modality-gap bias.
5. Storage of a versioned, hash-verified `AlignmentArtifact`.
