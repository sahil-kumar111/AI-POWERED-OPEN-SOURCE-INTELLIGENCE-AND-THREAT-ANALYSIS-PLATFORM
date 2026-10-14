# `threat_fusion_v1` — AEGIS Phase 4

Use the project-root wrapper `run_aegis_fusion.py` to read saved video and OSINT JSON reports and write `output/threat_score.json`. Do not run the example command below against live evidence without replacing all sample inputs with real, verified values.

The project-level command, actual available report filename in the provided snapshot, timestamp/location requirements, publication-time policy and test commands are documented in [`../../PHASE4_INTEGRATION_GUIDE.md`](../../PHASE4_INTEGRATION_GUIDE.md).

## Preserved scoring rules

| Evidence | Weight | Eligibility |
| --- | ---: | --- |
| Weapon | 0.40 | Confidence strictly greater than 0.60 |
| Crowd over capacity | 0.20 | Single-measurement count greater than an explicitly supplied capacity |
| Fire | 0.25 | Confidence strictly greater than 0.60 |
| Smoke | 0.25 | Confidence strictly greater than 0.60 |
| Qualifying nearby OSINT/NLP | 0.15 | Threat-keyword match, event time within the configured window, valid event coordinates within the configured radius of a video-location anchor |

The score is `min(1.0, sum(weight × confidence))`. It is an explainable weighted rule score, not a calibrated probability.

## Data safeguards

- One contribution per threat category per selected video report; per-frame records can recover timing but are not individually summed.
- Measured `max_confidence` is used when available. Adapter code does not adjust reported confidence or the strict threshold.
- `publishedAt` is article publication time, not verified incident occurrence time. Using it as a recency proxy requires the explicit `--allow-publication-time-proxy` flag.
- Missing/invalid timestamps, coordinates and unmatched evidence are preserved in `excluded_events` with source references where possible.
- Crowd totals or unique track counts accumulated across a full clip are not treated as concurrent occupancy. A peak concurrent count is eligible only when a peak timestamp and capacity are available.
- Cross-modal embeddings are not compared directly; fusion is symbolic/event-level.

## Tests

From `aegis_threat_fusion_v1`:

```powershell
python -m unittest discover -s threat_fusion_v1/tests -v
```
