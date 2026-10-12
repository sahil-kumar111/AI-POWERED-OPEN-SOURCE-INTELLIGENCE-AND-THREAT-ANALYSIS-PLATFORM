# NLP Intelligence Agent

Input: JSON report produced by the existing Video Intelligence Agent. This module does not run detection, process video frames, or collect OSINT.

## 1. Environment

Use the repository's activated virtual environment:

```bash
python3 -m pip install -r nlp_intelligence_complete/requirements.txt
python3 -m spacy download en_core_web_sm
```

The spaCy model is used for NER; VADER is used for sentiment only when real natural-language text is provided.

## 2. Run on an existing report

Run from the main repository root after copying `nlp_intelligence/` into the root:

```bash
python3 - <<'PY'
import json
from nlp_intelligence import process_record

report = "output/YOUTUBE_YouTubeCCTV196_gun_2_report.json"
result = process_record(report)

with open("output/YOUTUBE_YouTubeCCTV196_gun_2_nlp.json", "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2)

print(json.dumps(result, indent=2))
PY
```

## 3. Supply genuine text if available

```python
result = process_record(
    "output/YOUTUBE_YouTubeCCTV196_gun_2_report.json",
    incident_description="A witness reported a person carrying a pistol near Central Station in Chennai."
)
```

The example description is only illustrative. Do not invent a description for an actual incident. Pass a transcript/caption or human-provided description only when available, and preserve its provenance.

## 4. Components

1. Input validation and JSON loading.
2. Analysis context construction.
3. Named entity recognition using spaCy.
4. Threat extraction from detector fields plus contextual keyword candidates.
5. Event detection by grouping adjacent video frames with the same threat set.
6. VADER sentiment for genuine text only.
7. Keyword extraction.
8. Output serialization and batch error handling.

## 5. Caveats

- The input video report is structured data, not a natural-language article. NER and sentiment are not run on the generated template because that would produce misleading results.
- VADER scores describe textual sentiment, not incident severity.
- Threat confidence copied from the video report is the detector's average confidence, not an NLP confidence.
- Rule-based threat mentions are candidates and may need human review.
- spaCy's small English model may miss domain-specific entities. Evaluate with labelled project data before deployment.
