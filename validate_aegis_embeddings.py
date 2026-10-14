from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
out = root / "representation_learning" / "outputs"

npz_path = out / "aegis_evidence_embeddings.npz"
json_path = out / "aegis_evidence_metadata.json"

if not npz_path.exists() or not json_path.exists():
    raise SystemExit("FAIL: Unified embedding output files are missing.")

data = json.loads(json_path.read_text(encoding="utf-8"))
with np.load(npz_path, allow_pickle=False) as z:
    vectors = z["embeddings"]
    ids = z["evidence_ids"].astype(str)
    sources = z["source_types"].astype(str)

print("=" * 60)
print(" AEGIS REPRESENTATION LEARNING - VALIDATION")
print("=" * 60)

print("Vector matrix:", vectors.shape)
print("Metadata records:", len(data.get("records", [])))
print("Vector IDs:", len(ids))
print("Source labels:", len(sources))
print("Finite values:", bool(np.isfinite(vectors).all()))

norms = np.linalg.norm(vectors, axis=1)
print("All vectors normalized:", bool(np.allclose(norms, 1.0, atol=1e-4)))
print("Unique evidence IDs:", len(set(ids)) == len(ids))
print("Vector/metadata count matches:", len(vectors) == len(data.get("records", [])))

records = data.get("records", [])
counts = {}
for record in records:
    source = record.get("source_type", "unknown")
    counts[source] = counts.get(source, 0) + 1

print("\nSOURCE COUNTS")
for source, count in sorted(counts.items()):
    print(f"{source}: {count}")

print("\nDETECTION RECORD QUALITY")
for source in ("person_detection", "crowd_detection"):
    selected = [r for r in records if r.get("source_type") == source]
    missing_description = sum(not r.get("description", "").strip() for r in selected)
    missing_report = sum(not r.get("source_report") for r in selected)
    unknown_description = sum(
        "unknown" in r.get("description", "").lower()
        for r in selected
    )
    duplicate_descriptions = len(selected) - len({
        r.get("description", "").strip() for r in selected
    })
    print(f"{source}: {len(selected)} records")
    print(f"  Empty descriptions: {missing_description}")
    print(f"  Missing source report: {missing_report}")
    print(f"  Descriptions containing 'unknown': {unknown_description}")
    print(f"  Duplicate descriptions: {duplicate_descriptions}")

print("\nVECTOR/METADATA LABEL MATCH:")
print("IDs match metadata:", ids.tolist() == [
    str(r.get("evidence_id", "")) for r in records
])
print("Source labels match metadata:", sources.tolist() == [
    str(r.get("source_type", "")) for r in records
])

passed = (
    vectors.ndim == 2
    and vectors.shape[1] == 384
    and len(vectors) == len(records) == len(ids) == len(sources)
    and np.isfinite(vectors).all()
    and np.allclose(norms, 1.0, atol=1e-4)
    and len(set(ids)) == len(ids)
    and ids.tolist() == [str(r.get("evidence_id", "")) for r in records]
    and sources.tolist() == [str(r.get("source_type", "")) for r in records]
)

print("\nSTRUCTURAL VALIDATION:", "PASS" if passed else "FAIL")
print("Note: PASS checks structure and numeric validity, not semantic correctness.")
