from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "representation_learning" / "models" / "multilingual-e5-small"
OUT_DIR = ROOT / "representation_learning" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not (MODEL_DIR / "model.safetensors").exists():
    raise FileNotFoundError(f"Model weights not found: {MODEL_DIR}")

print("=" * 58)
print(" AEGIS REPRESENTATION LEARNING - SAVE EMBEDDINGS")
print("=" * 58)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR), local_files_only=True
)
model = AutoModel.from_pretrained(
    str(MODEL_DIR), local_files_only=True
)
model.eval()
model.to("cpu")

labels = ["weapon", "fire", "crowd", "flood"]
descriptions = [
    "Security camera footage reports a person carrying a weapon near an entrance.",
    "A fire breaks out inside a building and smoke spreads through the area.",
    "A large crowd gathers at an entrance, causing congestion and unusual movement.",
    "Heavy rainfall causes flooding on nearby roads and disrupts traffic.",
]
query = "A person carrying a weapon is seen near a building entrance."

def embed(texts, role):
    prefixed = [f"{role}: {text}" for text in texts]
    tokens = tokenizer(
        prefixed,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.inference_mode():
        output = model(**tokens)
        hidden = output.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vectors = F.normalize(pooled, p=2, dim=1)
    return vectors.cpu().numpy().astype(np.float32)

evidence_vectors = embed(descriptions, "passage")
query_vector = embed([query], "query")[0]
scores = evidence_vectors @ query_vector
ranking = np.argsort(scores)[::-1]

npz_path = OUT_DIR / "mentor_demo_embeddings.npz"
np.savez_compressed(
    npz_path,
    evidence_embeddings=evidence_vectors,
    query_embedding=query_vector,
    labels=np.asarray(labels),
)

metadata = {
    "model": "intfloat/multilingual-e5-small",
    "model_source": "local checkpoint",
    "embedding_dimension": int(evidence_vectors.shape[1]),
    "pooling": "attention-mask mean pooling",
    "normalization": "L2",
    "similarity_metric": "cosine similarity",
    "query": query,
    "evidence": [
        {
            "label": labels[i],
            "description": descriptions[i],
            "cosine_similarity": float(scores[i]),
        }
        for i in range(len(labels))
    ],
    "ranking": [
        {
            "rank": rank + 1,
            "label": labels[i],
            "cosine_similarity": float(scores[i]),
        }
        for rank, i in enumerate(ranking)
    ],
    "data_type": "synthetic demonstration data",
    "scores_are_probabilities": False,
}

json_path = OUT_DIR / "mentor_demo_metadata.json"
json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"Model loaded: {MODEL_DIR}")
print(f"Evidence embedding shape: {evidence_vectors.shape}")
print(f"Query embedding shape: {query_vector.shape}")
print(f"All vectors finite: {bool(np.isfinite(evidence_vectors).all() and np.isfinite(query_vector).all())}")
print(f"Evidence vector norms: {np.round(np.linalg.norm(evidence_vectors, axis=1), 4)}")
print("\nSIMILARITY RANKING")
for rank, i in enumerate(ranking, start=1):
    print(f"{rank}. {labels[i]:8s} cosine similarity = {scores[i]:.4f}")

print("\nSAVED FILES")
print(npz_path)
print(json_path)
print("\nDEMO RESULT: PASS")
