from pathlib import Path
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "representation_learning" / "models" / "multilingual-e5-small"
OUT_DIR = ROOT / "representation_learning" / "outputs"
REPORT_DIR = ROOT / "Output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def add_field(parts, label, value):
    if value is not None and value != "":
        parts.append(f"{label}: {clean(value)}.")

records = []
seen_reports = set()

# Keep four clearly labelled synthetic examples for the demo.
for label, description in [
    ("weapon", "Security camera footage reports a person carrying a weapon near an entrance."),
    ("fire", "A fire breaks out inside a building and smoke spreads through the area."),
    ("crowd", "A large crowd gathers at an entrance, causing congestion and unusual movement."),
    ("flood", "Heavy rainfall causes flooding on nearby roads and disrupts traffic."),
]:
    records.append({
        "evidence_id": f"synthetic-demo-{label}",
        "source_type": "synthetic_demo",
        "description": description,
        "synthetic": True,
    })

if REPORT_DIR.exists():
    for path in sorted(REPORT_DIR.rglob("*.json")):
        try:
            canonical = str(path.resolve()).casefold()
            if canonical in seen_reports:
                continue
            seen_reports.add(canonical)
            report = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(report, dict):
            continue

        detections = report.get("detections")
        if not isinstance(detections, list):
            continue

        video = report.get("input_video")
        if not video:
            video = path.name

        for index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue

            category = str(detection.get("category") or "unknown").lower()
            class_name = detection.get("class_name")
            message = detection.get("message")
            count = detection.get("count")
            peak = detection.get("peak_concurrent_persons")
            average = detection.get("avg_concurrent_persons")

            if category == "person":
                # Only call this crowd evidence when actual crowd statistics
                # indicate a substantial concurrent crowd.
                crowd_like = (
                    (isinstance(peak, (int, float)) and peak >= 10)
                    or (isinstance(average, (int, float)) and average >= 5)
                    or "crowd" in str(message or "").lower()
                )
                source_type = "crowd_detection" if crowd_like else "person_detection"
            else:
                source_type = "video_detection"

            parts = [
                f"Video: {video}.",
                f"Detection category: {category}.",
            ]
            add_field(parts, "Class", class_name)
            add_field(parts, "Detected", detection.get("detected"))
            add_field(parts, "Message", message)
            add_field(parts, "Count", count)
            add_field(parts, "Peak concurrent persons", peak)
            add_field(parts, "Average concurrent persons", average)
            add_field(parts, "Average confidence", detection.get("avg_confidence"))
            add_field(parts, "Frames detected", detection.get("frames_detected"))
            add_field(parts, "First seen seconds", detection.get("first_seen_sec"))
            add_field(parts, "Last seen seconds", detection.get("last_seen_sec"))
            add_field(parts, "Average bounding box", detection.get("avg_bbox"))

            report_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem)
            records.append({
                "evidence_id": f"{report_id}-detection-{index}",
                "source_type": source_type,
                "source_report": str(path.relative_to(ROOT)),
                "video": str(video),
                "category": category,
                "description": " ".join(parts),
                "detection_metadata": detection,
                "synthetic": False,
            })

if not (MODEL_DIR / "model.safetensors").exists():
    raise SystemExit(f"Model weights not found: {MODEL_DIR}")

ids = [r["evidence_id"] for r in records]
if len(ids) != len(set(ids)):
    raise SystemExit("FAIL: Duplicate evidence IDs remain; outputs were not regenerated.")

print("Loading local E5 model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
model.eval()
model.to("cpu")

all_vectors = []
for start in range(0, len(records), 16):
    batch = records[start:start + 16]
    tokens = tokenizer(
        ["passage: " + r["description"] for r in batch],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors = F.normalize(pooled, p=2, dim=1)
    all_vectors.append(vectors.cpu().numpy().astype(np.float32))

matrix = np.concatenate(all_vectors, axis=0)
norms = np.linalg.norm(matrix, axis=1)

if matrix.shape != (len(records), 384):
    raise SystemExit(f"Unexpected vector shape: {matrix.shape}")
if not np.isfinite(matrix).all() or not np.allclose(norms, 1, atol=1e-4):
    raise SystemExit("Embedding numeric validation failed.")

source_counts = {}
for record in records:
    source = record["source_type"]
    source_counts[source] = source_counts.get(source, 0) + 1

npz_path = OUT_DIR / "aegis_evidence_embeddings.npz"
json_path = OUT_DIR / "aegis_evidence_metadata.json"

np.savez_compressed(
    npz_path,
    embeddings=matrix,
    evidence_ids=np.asarray(ids),
    source_types=np.asarray([r["source_type"] for r in records]),
)

metadata = {
    "model": "intfloat/multilingual-e5-small",
    "embedding_dimension": 384,
    "pooling": "attention-mask mean pooling",
    "normalization": "L2",
    "embedding_kind": "text embeddings from evidence descriptions and detection reports",
    "visual_embeddings_generated": False,
    "reports_scanned": sorted(seen_reports),
    "record_count": len(records),
    "source_counts": source_counts,
    "records": records,
}
json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\n========== CORRECTED AEGIS EMBEDDINGS ==========")
print("Unique report files:", len(seen_reports))
print("Evidence records:", len(records))
print("Matrix shape:", matrix.shape)
print("All finite:", bool(np.isfinite(matrix).all()))
print("All normalized:", bool(np.allclose(norms, 1, atol=1e-4)))
print("Unique IDs:", len(ids) == len(set(ids)))
print("\nSource counts:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count}")
print("\nSaved:", npz_path)
print("Saved:", json_path)
print("RESULT: PASS")
