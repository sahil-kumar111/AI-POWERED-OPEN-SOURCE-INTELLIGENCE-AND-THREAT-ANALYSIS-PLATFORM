from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
out_dir = root / "representation_learning" / "outputs"

with np.load(out_dir / "aegis_evidence_embeddings.npz", allow_pickle=False) as saved:
    matrix = saved["embeddings"]
    ids = saved["evidence_ids"].astype(str)
    sources = saved["source_types"].astype(str)

metadata = json.loads(
    (out_dir / "aegis_evidence_metadata.json").read_text(encoding="utf-8")
)
records = metadata["records"]

if len(matrix) != len(records) or len(ids) != len(records):
    raise SystemExit("ERROR: vector and metadata counts differ.")

if len(set(ids)) != len(ids):
    raise SystemExit("ERROR: duplicate evidence IDs.")

if not np.isfinite(matrix).all():
    raise SystemExit("ERROR: non-finite embeddings.")

queries = {
    "PERSON": "person",
    "CROWD": "person",
    "WEAPON": "weapon",
    "FIRE": "fire",
    "SMOKE": "smoke",
}

real_indices = [
    i for i, source in enumerate(sources)
    if source != "synthetic_demo"
]

print("=" * 60)
print("AEGIS STRUCTURED DETECTION FILTER — V2")
print("=" * 60)
print("Real evidence records:", len(real_indices))
print("Synthetic records excluded:", len(records) - len(real_indices))

for query_name, target_category in queries.items():
    confirmed = []
    negative = []
    unavailable = []

    for i in real_indices:
        record = records[i]
        category = str(record.get("category") or "").strip().lower()

        if category != target_category:
            continue

        # Support either metadata layout without assuming a missing
        # detection status means False.
        detection_metadata = record.get("detection_metadata") or {}
        detected = detection_metadata.get("detected")

        if detected is None:
            detected = record.get("detected")

        item = {
            "id": str(ids[i]),
            "category": category,
            "detected": detected,
            "description": str(record.get("description") or ""),
            "report": record.get("source_report"),
        }

        if detected is True:
            confirmed.append(item)
        elif detected is False:
            negative.append(item)
        else:
            unavailable.append(item)

    # Crowd is a person-detection subtype, not a separate raw category.
    if query_name == "CROWD":
        confirmed = [
            item for item in confirmed
            if "peak concurrent persons: 15" in item["description"].lower()
            or "large crowd" in item["description"].lower()
        ]
        print("\nCROWD uses a provisional text rule; validate against actual counts.")

    print(f"\n{query_name}")
    print(f"  Confirmed detections: {len(confirmed)}")
    print(f"  Negative detections:  {len(negative)}")
    print(f"  Status unavailable:   {len(unavailable)}")

    for item in confirmed[:5]:
        print(f"  [DETECTED] {item['id']} | {item['category']}")
        print("   ", item["description"][:160])

    for item in negative[:2]:
        print(f"  [NOT DETECTED] {item['id']} | {item['category']}")

print("\nSTRUCTURED FILTER TEST COMPLETE")
print("No embedding files or manifests were modified.")
