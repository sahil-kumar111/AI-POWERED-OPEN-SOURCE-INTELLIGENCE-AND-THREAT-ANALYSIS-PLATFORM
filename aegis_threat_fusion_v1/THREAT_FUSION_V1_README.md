# AEGIS Threat Fusion v1 — Phase 4

This is the existing rule-based fusion engine used by `run_aegis_fusion.py`. For the exact current project paths and PowerShell commands, see [`../PHASE4_INTEGRATION_GUIDE.md`](../PHASE4_INTEGRATION_GUIDE.md).

## Scoring rules (preserved)

- Weapon: weight `0.40`; confidence must be strictly `> 0.60`.
- Crowd over capacity: weight `0.20`; verified count must be greater than a configured capacity.
- Fire: weight `0.25`; confidence must be strictly `> 0.60`.
- Smoke: weight `0.25`; confidence must be strictly `> 0.60`.
- Qualifying nearby OSINT/NLP: weight `0.15`; needs keywords, valid event coordinates, a video-location anchor, and the configured temporal and geographic windows.

The score is `min(1.0, sum(weight × confidence))`. This is an explainable weighted rule score, not a calibrated probability. LOW is a valid outcome when no/limited evidence qualifies.

## Important data policies

- At most one category contribution is generated per selected video report; per-frame records recover timing but are not independently summed.
- Current video reports may include `max_confidence` and `average_confidence` both in summary rows and top-level dictionaries. The adapter uses measured max confidence when available, without changing it or changing the `> 0.60` threshold.
- Article `publishedAt` is not treated as incident occurrence time. `--allow-publication-time-proxy` is an explicit opt-in for a recency proxy only.
- Missing incident times, invalid coordinates, unknown camera coordinates, out-of-window/out-of-radius records, duplicates and insufficient confidence are returned in `excluded_events` with source traceability.
- Crowd counts are never inferred from whole-video person detection totals or unique track counts. A peak concurrent count is used only when the report also provides the peak timestamp and the caller provides capacity.
- No text and visual embeddings are directly compared. Fusion is symbolic/event-level.

Run the standard-library fusion tests from this directory:

```powershell
python -m unittest discover -s threat_fusion_v1/tests -v
```
