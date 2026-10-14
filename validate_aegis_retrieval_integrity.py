import json
from pathlib import Path

out = Path(__file__).resolve().parent / "representation_learning" / "outputs"

audit = json.loads(
    (out / "aegis_evidence_quality_audit.json").read_text(encoding="utf-8")
)
normalized = json.loads(
    (out / "aegis_normalized_video_evidence.json").read_text(encoding="utf-8")
)

evidence = {
    str(r["evidence_id"]): r
    for r in normalized["records"]
}

errors = []
checked = 0

for query, results in audit.get("retrieval_summary", {}).items():
    for item in results:
        evidence_id = item.get("evidence_id")
        source = evidence.get(str(evidence_id))

        if source is None:
            errors.append(f"{query}: unmatched ID {evidence_id}")
            continue

        checked += 1

        if item.get("source_category") != source.get("category"):
            errors.append(f"{query}: category mismatch for {evidence_id}")

        if item.get("detected") != source.get("detected"):
            errors.append(f"{query}: detection-status mismatch for {evidence_id}")

print("=" * 60)
print("AEGIS RETRIEVAL INTEGRITY CHECK")
print("=" * 60)
print("Normalized evidence IDs:", len(evidence))
print("Retrieved entries checked:", checked)
print("Integrity errors:", len(errors))

for error in errors:
    print("ERROR:", error)

if errors:
    print("RESULT: FAIL")
else:
    print("RESULT: PASS")

print("No files modified.")
