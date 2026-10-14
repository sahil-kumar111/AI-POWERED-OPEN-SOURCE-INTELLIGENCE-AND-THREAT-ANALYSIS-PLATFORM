from pathlib import Path
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "representation_learning" / "outputs"
MODEL_DIR = ROOT / "representation_learning" / "models" / "multilingual-e5-small"

SOURCE = OUT / "aegis_normalized_video_evidence.json"
VECTOR_OUT = OUT / "aegis_evidence_embeddings_v2.npz"
META_OUT = OUT / "aegis_evidence_metadata_v2.json"

def clean_description(record):
    category = record["category"]
    detected = record["detected"]
    parts = []

    if category == "person":
        count = record.get("person_count")
        peak = record.get("peak_concurrent_persons")
        average = record.get("avg_concurrent_persons")

        if detected:
            parts.append("Person detection confirmed.")
            if count is not None:
                parts.append(f"Total people detected: {count}.")
            if peak is not None:
                parts.append(f"Peak concurrent people: {peak}.")
            if average is not None:
                parts.append(f"Average concurrent people: {average:.2f}.")
        else:
            parts.append("No person detected in the video.")
    else:
        class_name = record.get("class_name")
        label = str(class_name).strip() if class_name else category

        if detected:
            parts.append(f"{category.capitalize()} detected.")
            if label.lower() != category:
                parts.append(f"Detected class: {label}.")
        else:
            parts.append(f"No {category} detected in the video.")

    confidence = record.get("avg_confidence")
    if detected and confidence is not None:
        parts.append(f"Average detector confidence: {confidence:.4f}.")

    frames = record.get("frames_detected")
    if detected and isinstance(frames, int) and not isinstance(frames, bool):
        parts.append(f"Frames with detections: {frames}.")

    first = record.get("first_seen_sec")
    last = record.get("last_seen_sec")
    if detected and first is not None:
        parts.append(f"First detected at {first:.2f} seconds.")
    if detected and last is not None:
        parts.append(f"Last detected at {last:.2f} seconds.")

    return " ".join(parts)

def main():
    if not SOURCE.exists():
        raise SystemExit(f"Normalized evidence file not found: {SOURCE}")

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = payload["records"]

    if not records:
        raise SystemExit("No normalized evidence records.")

    ids = [record["evidence_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate evidence IDs.")

    texts = [clean_description(record) for record in records]

    if any(not text.strip() for text in texts):
        raise SystemExit("Empty clean description detected.")

    print("Loading local E5 model...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()
    model.to("cpu")

    vectors = []
    batch_size = 8

    for start in range(0, len(texts), batch_size):
        batch = ["passage: " + text for text in texts[start:start + batch_size]]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings = F.normalize(pooled, p=2, dim=1)

        vectors.append(embeddings.cpu().numpy().astype(np.float32))

    matrix = np.concatenate(vectors, axis=0)

    if matrix.shape != (len(records), 384):
        raise SystemExit(f"Unexpected matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise SystemExit("Non-finite embeddings.")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
        raise SystemExit("Embedding normalization check failed.")

    np.savez_compressed(
        VECTOR_OUT,
        embeddings=matrix,
        evidence_ids=np.asarray(ids, dtype=str),
        source_types=np.asarray(
            [record["source_type"] for record in records], dtype=str
        ),
    )

    metadata = {
        "schema_version": "2.0",
        "model": "intfloat/multilingual-e5-small",
        "embedding_dimension": 384,
        "pooling": "attention-mask mean pooling",
        "normalization": "L2",
        "embedding_kind": "clean_normalized_video_evidence",
        "source_file": SOURCE.name,
        "record_count": len(records),
        "records": [
            {
                **record,
                "embedding_text": text,
            }
            for record, text in zip(records, texts)
        ],
    }

    META_OUT.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("AEGIS CLEAN EMBEDDINGS V2")
    print("=" * 60)
    print("Records:", len(records))
    print("Matrix shape:", matrix.shape)
    print("All finite:", bool(np.isfinite(matrix).all()))
    print(
        "All normalized:",
        bool(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4))
    )
    print("Unique IDs:", len(set(ids)) == len(ids))
    print("Vectors:", VECTOR_OUT)
    print("Metadata:", META_OUT)
    print("Original embeddings and frozen manifest unchanged.")
    print("RESULT: PASS")

if __name__ == "__main__":
    main()
