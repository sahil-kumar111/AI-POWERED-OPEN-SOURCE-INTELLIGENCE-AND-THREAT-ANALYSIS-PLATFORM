from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "representation_learning" / "outputs"
MODEL_DIR = ROOT / "representation_learning" / "models" / "multilingual-e5-small"

NORMALIZED = OUT / "aegis_normalized_video_evidence.json"
EMBEDDINGS = OUT / "aegis_evidence_embeddings.npz"
METADATA = OUT / "aegis_evidence_metadata.json"

CATEGORIES = {"person", "weapon", "fire", "smoke"}

def load_inputs():
    normalized = json.loads(NORMALIZED.read_text(encoding="utf-8"))
    old_metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    with np.load(EMBEDDINGS, allow_pickle=False) as saved:
        matrix = saved["embeddings"]
        ids = saved["evidence_ids"].astype(str)
        sources = saved["source_types"].astype(str)

    old_records = old_metadata["records"]
    old_by_id = {
        str(record["evidence_id"]): record
        for record in old_records
    }

    if len(old_by_id) != len(old_records):
        raise SystemExit("ERROR: duplicate IDs in embedding metadata.")

    if len(ids) != len(matrix):
        raise SystemExit("ERROR: vector/ID count mismatch.")

    if len(set(ids)) != len(ids):
        raise SystemExit("ERROR: duplicate embedding IDs.")

    if matrix.ndim != 2 or matrix.shape[1] != 384:
        raise SystemExit(f"ERROR: unexpected embedding shape {matrix.shape}")

    if not np.isfinite(matrix).all():
        raise SystemExit("ERROR: embeddings contain non-finite values.")

    normalized_by_id = {
        str(record["evidence_id"]): record
        for record in normalized["records"]
    }

    # Match by ID; never assume row order is the same.
    aligned = []
    for row, evidence_id in enumerate(ids):
        if str(sources[row]) == "synthetic_demo":
            continue

        record = normalized_by_id.get(str(evidence_id))
        old_record = old_by_id.get(str(evidence_id))

        if record is None or old_record is None:
            raise SystemExit(
                f"ERROR: no matching normalized/metadata record for {evidence_id}"
            )

        aligned.append({
            "vector": matrix[row],
            "record": record,
            "old_description": old_record.get("description", ""),
        })

    missing = set(normalized_by_id) - {
        str(ids[i]) for i in range(len(ids))
        if str(sources[i]) != "synthetic_demo"
    }
    if missing:
        raise SystemExit(
            "ERROR: normalized records missing embeddings: "
            + ", ".join(sorted(missing))
        )

    return aligned

def encode_query(model, tokenizer, query):
    tokens = tokenizer(
        ["query: " + query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vector = F.normalize(pooled, p=2, dim=1)

    return vector.cpu().numpy()[0].astype(np.float32)

def main():
    for path in (NORMALIZED, EMBEDDINGS, METADATA):
        if not path.exists():
            raise SystemExit(f"Required file not found: {path}")

    evidence = load_inputs()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()
    model.to("cpu")

    print("=" * 64)
    print("AEGIS CATEGORY-AWARE RETRIEVAL")
    print("=" * 64)
    print("Real evidence records with aligned vectors:", len(evidence))

    results = {}

    for category in ("person", "crowd", "weapon", "fire", "smoke"):
        query = {
            "person": "A person is detected in surveillance video.",
            "crowd": "A crowd of people is present, with high concurrent occupancy.",
            "weapon": "A weapon such as a knife or rifle is detected.",
            "fire": "Fire is detected in video footage.",
            "smoke": "Smoke is detected in video footage.",
        }[category]

        mode = "confirmed"
        query_vector = encode_query(model, tokenizer, query)

        candidates = []
        for item in evidence:
            record = item["record"]

            if category == "crowd":
                if record["category"] != "person":
                    continue
                # Provisional heuristic, not a trained crowd classifier.
                peak = record.get("peak_concurrent_persons")
                average = record.get("avg_concurrent_persons")
                is_crowd = (
                    (peak is not None and peak >= 10)
                    or (average is not None and average >= 5)
                )
                if not is_crowd:
                    continue
            elif record["category"] != category:
                continue

            if mode == "confirmed" and record["detected"] is not True:
                continue

            candidates.append((item, record))

        ranked = []
        for item, record in candidates:
            score = float(np.dot(query_vector, item["vector"]))
            ranked.append((score, record, item["old_description"]))

        ranked.sort(key=lambda x: x[0], reverse=True)

        print(f"\n{category.upper()} — confirmed detections")
        print("Eligible records:", len(ranked))

        results[category] = {
            "query": query,
            "mode": mode,
            "results": [],
        }

        for rank, (score, record, old_description) in enumerate(ranked[:5], 1):
            result = {
                "rank": rank,
                "evidence_id": record["evidence_id"],
                "category": record["category"],
                "detected": record["detected"],
                "semantic_similarity": score,
                "detector_confidence": record["avg_confidence"],
                "source_report": record["source_report"],
                "description": old_description,
                "input_video": record["input_video"],
            }
            results[category]["results"].append(result)

            print(
                f"{rank}. {record['evidence_id']} | "
                f"similarity={score:.4f} | "
                f"detector_confidence={record['avg_confidence']}"
            )

    output_path = OUT / "aegis_category_retrieval.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:", output_path)
    print("Retrieval completed. Crowd filtering remains heuristic.")
    print("Original reports, embeddings, and manifests were not modified.")

if __name__ == "__main__":
    main()
