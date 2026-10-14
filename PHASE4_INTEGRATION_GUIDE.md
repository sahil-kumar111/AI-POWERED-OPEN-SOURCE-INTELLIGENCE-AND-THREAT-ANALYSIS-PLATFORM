# AEGIS Phase 4 — Saved-Output Integration

This phase connects the **already implemented** video detection pipeline, Location Threat Monitor export, and existing `aegis_threat_fusion_v1` rule engine. It does not modify or rerun weapon/person/fire/smoke detection, tracking, NLP inference, embedding generation, or training.

## Data flow

1. Run the Location Threat Monitor. When **Analyze Location** succeeds, the app exports the exact returned list to `output/osint_results.json` (UTF-8, no field removal). An empty result is saved as `[]`. Analysis/export failures are shown in the UI.
2. Select one existing video report JSON. The adapter supports both the dictionary-based metrics (`detection_counts`, `max_confidence`, `average_confidence`) and the summary `detections` list. For the current repository pipeline, `max_confidence` is preserved and used when available; the engine continues to require confidence **strictly greater than 0.60** for weapon/fire/smoke.
3. Supply the actual recording start timestamp and actual camera coordinates for real temporal/spatial correlation. Relative `timestamp_sec` values from saved per-frame `threat_events` can be added to that start time. If no relative event timestamp exists, the category event is anchored only to the supplied start time and this basis is recorded.
4. The engine creates `output/threat_score.json`, with contributions and exclusions traceable to source report paths / OSINT URLs. A score is an explainable weighted rule score, **not a calibrated probability**.

## Run the UI (from the project root)

```powershell
streamlit run .\location_threat_monitor\app.py
```

Enter a location and click **Analyze Location**. The app calls its existing `analyze_threats(...)` workflow once, then exports its returned records to `output\osint_results.json`. It does not run a second collector.

## Run real video + OSINT fusion

Use a video report that exists in your current `output` directory. The repository snapshot bundled with this guide contains `output\YOUTUBE_YouTubeCCTV196_gun_2_report.json`; if your current report has a different name, pass that actual path.

```powershell
$videoStartUtc = Read-Host "Actual video recording start time in UTC (ISO-8601, e.g. 2026-10-13T12:00:00Z)"
$cameraLat = [double](Read-Host "Actual camera latitude")
$cameraLon = [double](Read-Host "Actual camera longitude")
$zone = Read-Host "Actual shared zone identifier"

python .\run_aegis_fusion.py `
  --video-report ".\output\YOUTUBE_YouTubeCCTV196_gun_2_report.json" `
  --osint-json ".\output\osint_results.json" `
  --zone $zone `
  --event-time $videoStartUtc `
  --latitude $cameraLat `
  --longitude $cameraLon `
  --window-seconds 3600 `
  --osint-radius-meters 1000 `
  --output ".\output\threat_score.json"
```

Use the real camera coordinates, not the Location Threat Monitor's searched-location coordinates unless they are confirmed to be the same place. The start time must describe the actual recording; do not copy the example timestamp into a live run. The default time window is 3600 seconds and radius is 1000 metres; specify values appropriate to the deployment.

### Article publication times

`publishedAt` is not treated as the incident occurrence time. When a result provides no explicit `incident_timestamp`, `event_timestamp`, `incident_time`, `event_time`, `occurred_at`, `timestamp_utc`, or `timestamp`, it is excluded with an explanation. Only add `--allow-publication-time-proxy` when you deliberately accept article publication time as a **recency proxy**, not a verified incident time. The output's run metadata records that opt-in. It should not be used to claim that a camera event and a news article refer to the same event.

### Crowd capacity

A crowd-over-capacity score requires a legitimate configured capacity and a simultaneous count from one measurement. Use `--crowd-count N --capacity C` only when both correspond to the same measurement/venue. A total person detection count or a unique-person count accumulated across the clip is not current occupancy. The adapter can use `peak_concurrent_persons` only when the report also has a timestamp for that peak; otherwise it records an exclusion rather than inventing the peak time.

## Video-only software validation

The bundled source report does not contain a verified absolute recording start time or camera coordinates. To validate the wrapper without faking real-world correlation, run a clearly marked test-only timestamp and skip OSINT:

```powershell
python .\run_aegis_fusion.py `
  --video-report ".\output\YOUTUBE_YouTubeCCTV196_gun_2_report.json" `
  --video-only `
  --validation-only `
  --zone "validation_only" `
  --event-time "2000-01-01T00:00:00Z" `
  --output ".\output\threat_score.json"
```

The result includes `run_metadata.validation_only = true` and a warning. **Do not use this validation output for OSINT correlation or a real-world threat decision.** Rerun the real fusion command above with the actual recording time, camera coordinates and a fresh saved OSINT export to replace it.

## Weights and exclusions

The engine's configured weights remain unchanged: weapon `0.40`, crowd-over-capacity `0.20`, fire `0.25`, smoke `0.25`, and qualifying nearby OSINT `0.15`. The score is the sum of qualifying weighted contributions capped at `1.0`. Weapon/fire/smoke confidence remains strictly `> 0.60`. Missing timestamps/coordinates, time-window mismatches, out-of-radius OSINT, unsupported/low-confidence events, duplicate evidence, and invalid crowd measurements are explained under `excluded_events` where the input record can be evaluated.

OSINT rows without source IDs get a deterministic ID: preserve upstream `event_id`/`id`; otherwise use `osint:sha256:<first 20 hex characters of SHA-256(url or canonical row JSON)>`. Source URLs and source report paths are retained as `raw_ref` values.

## Tests

From the project root:

```powershell
python -m unittest discover -s .\aegis_threat_fusion_v1\threat_fusion_v1\tests -v
python -m unittest discover -s .\location_threat_monitor -p "test_osint_export.py" -v
```

These tests use small local fixtures. They do not make network/API calls, run video inference, download model weights, train models, or regenerate embeddings.
