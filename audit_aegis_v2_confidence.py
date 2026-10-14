import json
from pathlib import Path

out = Path(__file__).resolve().parent / "representation_learning" / "outputs"

normalized = json.loads(
    (out / "aegis_normalized_video_evidence.json").read_text(encoding="utf-8")
)
metadata = json.loads(
    (out / "aegis_evidence_metadata_v2.json").read_text(encoding="utf-8")
)

normalized_by_id = {
    str(r["evidence_id"]): r for r in normalized["records"]
}

print("=" * 72)
print("V2 EMBEDDING DESCRIPTION AUDIT")
print("=" * 72)

checked = 0
confidence_present = 0

for item in metadata["records"]:
    evidence_id = str(item.get("evidence_id", ""))
    record = normalized_by_id.get(evidence_id)

    if not record or record.get("confidence") is None:
        continue

    description = (
        item.get("embedding_text")
        or item.get("description")
        or item.get("text")
        or ""
    )

    checked += 1
    contains_value = str(record["confidence"]) in str(description)

    if contains_value:
        confidence_present += 1

    print("\nID:", evidence_id)
    print("Category:", record.get("category"))
    print("Source confidence:", record["confidence"])
    print("Confidence included in description:", contains_value)
    print("Embedding description:", description)

print("\n" + "=" * 72)
print("Records with confidence available:", checked)
print("Descriptions containing confidence value:", confidence_present)
print("Audit complete; no files modified.")
