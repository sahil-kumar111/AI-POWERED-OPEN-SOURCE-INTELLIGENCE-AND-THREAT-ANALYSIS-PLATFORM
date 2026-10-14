from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

root = Path(__file__).resolve().parent
model_dir = root / "representation_learning" / "models" / "multilingual-e5-small"
out_dir = root / "representation_learning" / "outputs"

npz_path = out_dir / "aegis_evidence_embeddings.npz"
metadata_path = out_dir / "aegis_evidence_metadata.json"

if not npz_path.exists() or not metadata_path.exists():
    raise SystemExit("Unified embedding files not found. Run the generation script first.")

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
records = metadata["records"]

with np.load(npz_path, allow_pickle=False) as saved:
    matrix = saved["embeddings"]
    ids = saved["evidence_ids"].astype(str)
    sources = saved["source_types"].astype(str)

if len(records) != len(matrix) or len(ids) != len(matrix):
    raise SystemExit("Record/vector counts do not match.")
if len(set(ids)) != len(ids):
    raise SystemExit("Duplicate evidence IDs found.")
if not np.isfinite(matrix).all():
    raise SystemExit("Non-finite embedding values found.")

tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
model.eval()
model.to("cpu")

queries = {
    "person": "A person is detected in surveillance video.",
    "crowd": "A large crowd of people gathers and causes congestion.",
    "weapon": "A weapon is detected in video footage.",
    "fire": "Fire is detected in a building.",
    "smoke": "Smoke is detected in video footage.",
}

def encode_query(text):
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

results = {}
print("=" * 68)
print(" AEGIS - SEMANTIC RETRIEVAL TEST")
print("=" * 68)
print("Saved evidence vectors:", matrix.shape)
print("Queries:", len(queries))

for name, query in queries.items():
    query_vector = encode_query(query)
    scores = matrix @ query_vector
    order = np.argsort(scores)[::-1][:5]

    print(f"\nQUERY: {name.upper()} — {query}")
    top_results = []

    for rank, i in enumerate(order, start=1):
        record = records[int(i)]
        item = {
            "rank": rank,
            "evidence_id": str(ids[i]),
            "source_type": str(sources[i]),
            "category": record.get("category"),
            "score": float(scores[i]),
            "description": record.get("description", ""),
            "source_report": record.get("source_report"),
        }
        top_results.append(item)
        print(
            f"{rank}. score={scores[i]:.4f} | "
            f"{item['source_type']} | {item['evidence_id']}"
        )
        print("   ", item["description"][:220])

    results[name] = {
        "query": query,
        "top_results": top_results,
    }

result_path = out_dir / "aegis_retrieval_test.json"
result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\nSaved retrieval report:", result_path)
print("RETRIEVAL TEST COMPLETE")
print("Note: Similarity scores are not probabilities.")
print("Inspect top results to assess semantic correctness.")
