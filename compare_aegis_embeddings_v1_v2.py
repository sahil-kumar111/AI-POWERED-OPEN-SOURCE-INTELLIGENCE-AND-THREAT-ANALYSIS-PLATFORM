from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "representation_learning" / "outputs"
MODEL_DIR = ROOT / "representation_learning" / "models" / "multilingual-e5-small"

V1 = OUT / "aegis_evidence_embeddings.npz"
V1_META = OUT / "aegis_evidence_metadata.json"
V2 = OUT / "aegis_evidence_embeddings_v2.npz"
V2_META = OUT / "aegis_evidence_metadata_v2.json"

queries = {
    "person": ("person", "A person is detected in surveillance video."),
    "crowd": ("person", "A large crowd of people is present."),
    "weapon": ("weapon", "A weapon such as a knife or rifle is detected."),
    "fire": ("fire", "Fire is detected in video footage."),
    "smoke": ("smoke", "Smoke is detected in video footage."),
}

def load_vectors(path):
    with np.load(path, allow_pickle=False) as saved:
        matrix = saved["embeddings"]
        ids = saved["evidence_ids"].astype(str)

    if matrix.ndim != 2 or len(matrix) != len(ids):
        raise ValueError(f"Invalid vector/ID alignment in {path.name}")
    if len(set(ids)) != len(ids) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid IDs or values in {path.name}")

    return {evidence_id: matrix[i] for i, evidence_id in enumerate(ids)}

def query_vector(model, tokenizer, text):
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
        vector = F.normalize(pooled, p=2, dim=1)
    return vector.cpu().numpy()[0].astype(np.float32)

def eligible(record, query_name, target_category):
    if record.get("detected") is not True:
        return False

    if query_name != "crowd":
        return record.get("category") == target_category

    if record.get("category") != "person":
        return False

    peak = record.get("peak_concurrent_persons")
    average = record.get("avg_concurrent_persons")

    return (
        (peak is not None and peak >= 10)
        or (average is not None and average >= 5)
    )

def main():
    paths = (V1, V1_META, V2, V2_META)
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")

    v1 = load_vectors(V1)
    v2 = load_vectors(V2)

    meta1 = json.loads(V1_META.read_text(encoding="utf-8"))
    meta2 = json.loads(V2_META.read_text(encoding="utf-8"))

    records1 = {
        str(r["evidence_id"]): r for r in meta1["records"]
        if not r.get("synthetic", False)
        and r.get("source_type") != "synthetic_demo"
    }
    records2 = {
        str(r["evidence_id"]): r for r in meta2["records"]
    }

    ids1 = set(v1) & set(records1)
    ids2 = set(v2) & set(records2)

    if ids1 != ids2:
        raise SystemExit(
            f"Evidence ID mismatch. V1-only={ids1-ids2}, V2-only={ids2-ids1}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()
    model.to("cpu")

    output = {}
    print("=" * 68)
    print("AEGIS V1 vs V2 RETRIEVAL COMPARISON")
    print("Only confirmed real evidence; category-filtered")
    print("=" * 68)

    for name, (category, query) in queries.items():
        qvec = query_vector(model, tokenizer, query)
        output[name] = {"query": query, "V1": [], "V2": []}

        print(f"\n{name.upper()}")

        for version, vectors, records in (
            ("V1", v1, records1),
            ("V2", v2, records2),
        ):
            candidates = []

            for evidence_id in ids1:
                record = records[evidence_id]
                if not eligible(record, name, category):
                    continue

                score = float(np.dot(qvec, vectors[evidence_id]))
                candidates.append((score, evidence_id, record))

            candidates.sort(key=lambda x: x[0], reverse=True)

            print(f"  {version}: {len(candidates)} eligible records")
            for rank, (score, evidence_id, record) in enumerate(candidates[:3], 1):
                result = {
                    "rank": rank,
                    "evidence_id": evidence_id,
                    "similarity": score,
                    "category": record.get("category"),
                    "detected": record.get("detected"),
                    "description": (
                        record.get("embedding_text")
                        or record.get("description", "")
                    ),
                    "source_report": record.get("source_report"),
                }
                output[name][version].append(result)
                print(f"    {rank}. {evidence_id} | {score:.4f}")
                print("       ", result["description"][:160])

    result_path = OUT / "aegis_v1_v2_comparison.json"
    result_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved comparison:", result_path)
    print("COMPARISON COMPLETE — no existing artifacts modified.")

if __name__ == "__main__":
    main()
