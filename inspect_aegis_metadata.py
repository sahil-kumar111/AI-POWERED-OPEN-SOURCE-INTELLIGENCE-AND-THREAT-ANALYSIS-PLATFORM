from pathlib import Path
import json

root = Path(__file__).resolve().parent
out_dir = root / "representation_learning" / "outputs"

files = {
    "Embedding metadata": out_dir / "aegis_evidence_metadata.json",
    "Retrieval results": out_dir / "aegis_retrieval_test.json",
}

for label, path in files.items():
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    if not path.exists():
        print("NOT FOUND:", path)
        continue

    data = json.loads(path.read_text(encoding="utf-8"))
    print("Top-level keys:", list(data.keys()))

    records = data.get("records")
    if isinstance(records, list):
        print("Record count:", len(records))
        if records:
            print("First record:")
            print(json.dumps(records[0], indent=2, ensure_ascii=False))
    elif isinstance(data, dict):
        print("Sample:")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])

print("\nREAD-ONLY INSPECTION COMPLETE")
