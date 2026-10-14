from pathlib import Path
import json

root = Path(__file__).resolve().parent
report_dir = root / "Output"

if not report_dir.exists():
    raise SystemExit(f"Report directory not found: {report_dir}")

files = sorted(report_dir.glob("*report.json"))

if not files:
    raise SystemExit("No *report.json files found in Output.")

print("=" * 70)
print("AEGIS — ORIGINAL VIDEO REPORT AUDIT")
print("=" * 70)

for path in files:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"\nERROR reading {path.name}: {exc}")
        continue

    print(f"\nREPORT: {path.name}")
    print("Input video:", data.get("input_video"))
    detections = data.get("detections", [])

    if not isinstance(detections, list):
        print("WARNING: detections is not a list")
        continue

    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            print(f"  [{index}] Invalid detection record")
            continue

        print(
            f"  [{index}] category={detection.get('category')!r}"
            f" | class={detection.get('class_name')!r}"
            f" | detected={detection.get('detected')!r}"
            f" | confidence={detection.get('avg_confidence')!r}"
        )
        if detection.get("message"):
            print("      message:", detection["message"])
        if "count" in detection:
            print("      count:", detection["count"])
        if "peak_concurrent_persons" in detection:
            print(
                "      peak_concurrent_persons:",
                detection["peak_concurrent_persons"]
            )

print("\nAUDIT COMPLETE — source reports were read only.")
