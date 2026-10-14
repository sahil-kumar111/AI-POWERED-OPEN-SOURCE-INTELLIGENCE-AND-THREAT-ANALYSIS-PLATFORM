import json
from pathlib import Path

root = Path(__file__).resolve().parent
path = root / "representation_learning" / "outputs" / "aegis_normalized_video_evidence.json"
data = json.loads(path.read_text(encoding="utf-8"))

for r in data["records"]:
    if r.get("detected") is True:
        print("-" * 72)
        print("ID:", r.get("evidence_id"))
        print("Source:", r.get("source_report"))
        print("Category:", r.get("category"))
        print("Confidence:", r.get("confidence"))
        print("Message:", r.get("message"))
        print("Frames:", r.get("frames_detected"))
        print("First seen:", r.get("first_seen_sec"))
        print("Last seen:", r.get("last_seen_sec"))
        print("Peak persons:", r.get("peak_concurrent_persons"))
        print("Average persons:", r.get("avg_concurrent_persons"))
