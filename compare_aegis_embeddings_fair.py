from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "representation_learning" / "outputs"
MODEL_DIR = ROOT / "representation_learning" / "models" / "multilingual-e5-small"

def load_vectors(path):
    with np.load(path, allow_pickle=False) as saved:
        matrix = saved["embeddings"]
        ids = saved["evidence_ids"].astype(str)

    if matrix.ndim != 2 or len(ids) != len(matrix):
        raise ValueError(f"Invalid vector file: {path.name}")
    if len(set(ids)) != len(ids) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid IDs or values: {path.name}")

    return {evidence_id: matrix[i] for i, evidence_id in enumerate(ids)}

def encode_query(model, tokenizer, text):
    tokens = tokenizer(
        ["query: " + text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled, p=2, dim=1).cpu().numpy()[0].astype(np.float32)

def is_eligible(record, query_name):
    if record.get("detected") is not True:
        return False

    if query_name == "crowd":
        if record.get("category") != "person":
            return False
        peak = record.get("peak_concurrent_persons")
        average = record.get("avg_concurrent_persons")
        return (
            (peak is not None and peak >= 10)
            or (average is not None and average >= 5)
        )

    return record.get("category") == query_name

def main():
    v1_path = OUT / "aegis_evidence_embeddings.npz"
    v2_path = OUT / "aegis_evidence_embeddings_v2.npz"
    normalized_path = OUT / "aegis_normalized_video_evidence.json"
    v1_meta_path = OUT / "aegis_evidence_metadata.json"
    v2_meta_path = OUT / "aegis_evidence_metadata_v2.json"

    for path in (v1_path, v2_path, normalized_path, v1_meta_path, v2_meta_path):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    v1 = load_vectors(v1_path)
    v2 = load_vectors(v2_path)
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    meta1 = json.loads(v1_meta_path.read_text(encoding="utf-8"))
    meta2 = json.loads(v2_meta_path.read_text(encoding="utf-8"))

    records = {str(r["evidence_id"]): r for r in normalized["records"]}
    old_records = {str(r["evidence_id"]): r for r in meta1["records"]}
    new_records = {str(r["evidence_id"]): r for r in meta2["records"]}

    # Require identical real-evidence IDs across both embedding versions.
    real_ids = set(records)
    if not real_ids.issubset(v1) or not real_ids.issubset(v2):
        raise SystemExit("V1/V2 do not both contain every normalized evidence ID.")

    if not real_ids.issubset(old_records) or not real_ids.issubset(new_records):
        raise SystemExit("Metadata IDs do not align with normalized evidence.")

    if v1[next(iter(real_ids))].shape != v2[next(iter(real_ids))].shape:
        raise SystemExit("V1/V2 embedding dimensions differ.")

    queries = {
        "person": "A person is detected in surveillance video.",
        "crowd": "A large crowd of people is present.",
        "weapon": "A weapon such as a knife or rifle is detected.",
        "fire": "Fire is detected in video footage.",
        "smoke": "Smoke is detected in video footage.",
    }

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()
    model.to("cpu")

    output = {}
    print("=" * 64)
    print("FAIR AEGIS V1 VS V2 COMPARISON")
    print("Same evidence IDs, categories and detection status")
    print("=" * 64)
    print("Shared real-evidence records:", len(real_ids))

    for name, query in queries.items():
        qvec = encode_query(model, tokenizer, query)
        eligible_ids = [
            evidence_id for evidence_id, record in records.items()
            if is_eligible(record, name)
        ]

        output[name] = {
            "query": query,
            "eligible_count": len(eligible_ids),
            "V1": [],
            "V2": [],
        }

        print(f"\n{name.upper()} — eligible: {len(eligible_ids)}")

        for version, vectors, version_records in (
            ("V1", v1, old_records),
            ("V2", v2, new_records),
        ):
            ranked = sorted(
                (
                    (float(np.dot(qvec, vectors[evidence_id])), evidence_id)
                    for evidence_id in eligible_ids
                ),
                reverse=True,
            )

            print(f"  {version}:")
            for rank, (score, evidence_id) in enumerate(ranked[:5], 1):
                record = records[evidence_id]
                description = (
                    version_records[evidence_id].get("embedding_text")
                    or version_records[evidence_id].get("description", "")
                )
                item = {
                    "rank": rank,
                    "evidence_id": evidence_id,
                    "similarity": score,
                    "category": record["category"],
                    "detected": record["detected"],
                    "description": description,
                }
                output[name][version].append(item)
                print(f"    {rank}. {evidence_id} | {score:.4f}")

    result_path = OUT / "aegis_v1_v2_fair_comparison.json"
    result_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSaved:", result_path)
    print("Comparison complete. Existing artifacts unchanged.")

if __name__ == "__main__":
    main()
