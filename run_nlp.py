
import json
from pathlib import Path

from nlp_intelligence import process_record


def main():
    report_path = Path(
        "output/YOUTUBE_YouTubeCCTV196_gun_2_report.json"
    )

    if not report_path.exists():
        raise FileNotFoundError(
            f"Video report not found: {report_path}\n"
            "Run threat_video_pipeline.py first."
        )

    result = process_record(report_path)

    output_path = Path(
        "output/YOUTUBE_YouTubeCCTV196_gun_2_nlp.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\nNLP INTELLIGENCE RESULTS")
    print("=" * 35)
    print("Record ID:", result["record_id"])
    print("Threats:")

    for threat in result["threats"]:
        print(
            f'  {threat["category"]}: '
            f'{threat["status"]}'
        )

    print("Events:", len(result["events"]))
    print("Entities:", len(result["entities"]))
    print("Keywords:", result["keywords"])
    print("Sentiment:", result["sentiment"]["label"])
    print("Output:", output_path.resolve())


if __name__ == "__main__":
    main()
