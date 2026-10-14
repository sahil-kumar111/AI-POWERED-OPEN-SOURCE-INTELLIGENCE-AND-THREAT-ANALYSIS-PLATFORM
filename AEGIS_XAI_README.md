# AEGIS Explainable AI (XAI) module

This module adds a post-hoc evidence trace for existing AEGIS outputs. It is designed to explain saved detector reports, saved NLP/source-reconciliation output, the existing deterministic Phase 4 fusion score, and saved OSINT rows. It does **not** retrain or alter the weapon detector, NLP pipeline, embedding encoder, or fusion engine.

## Included capabilities

- **Video/event explanation:** exposes category-level detections, detector counts, frame coverage, mean/max reported confidence, and example frame-level boxes/timestamps from `*_report.json`.
- **NLP/source explanation:** traces the existing `combined_intelligence.json` reconciliation statuses, text matches, negation/uncertainty flags, and extracted entities.
- **Fusion explanation:** exposes each reported weighted contribution, checks `weight × confidence` against each saved sub-score, lists excluded events and their reasons, and verifies the contribution sum against the saved score without rewriting it.
- **OSINT result explanation:** shown inside an “Explainable AI” expander under each Location Threat Monitor result, using only values returned by the current analysis pipeline.
- **Optional visual EigenCAM:** runs on one image and an existing Ultralytics YOLO checkpoint. It does not rerun or retrain a video pipeline.

## Create an explanation report (no detector rerun)

Run from the project root in the same Python environment used for AEGIS:

```powershell
python -m explainable_ai `
  --video-report "output\YOUTUBE_YouTubeCCTV196_gun_2_report.json" `
  --combined-intelligence "output\YOUTUBE_YouTubeCCTV196_gun_2_combined_intelligence.json" `
  --threat-score "output\threat_score.json" `
  --output "output\xai_explanation.json"
```

To trace saved OSINT results too, add `--osint-results "output\osint_results.json"` if that file contains the saved result array/object. An empty `[]` is accepted and produces an empty OSINT explanation list.

## Optional one-image visual explanation

Use a real image/frame and the already-trained weapon checkpoint. This does only one image inference, not whole-video inference:

```powershell
python -m explainable_ai `
  --video-report "output\YOUTUBE_YouTubeCCTV196_gun_2_report.json" `
  --combined-intelligence "output\YOUTUBE_YouTubeCCTV196_gun_2_combined_intelligence.json" `
  --threat-score "output\threat_score.json" `
  --visual-video "input\YOUTUBE_YouTubeCCTV196_gun_2.mp4" `
  --visual-frame-index 0 `
  --weapon-checkpoint "weapon_detection_project\ml_models\weapon\best.pt" `
  --visual-output "output\xai_eigencam_overlay.jpg" `
  --output "output\xai_explanation.json"
```

The environment needs `ultralytics`, `torch`, `opencv-python`, and `numpy`. Install these into the existing project environment only if missing; do not replace model checkpoints. If a checkpoint is a custom architecture not loadable by the installed Ultralytics version, the function fails with an explicit message rather than silently generating a fake heatmap.

**Interpretation note:** EigenCAM is class-agnostic activation visualization, not a causal explanation of one class score. It can highlight background/context and must be visually reviewed. This runtime lacked the `ultralytics` package and had no package-index network access, so I could test the rest of the XAI module here but could not validate the visual heatmap against the checkpoint in this environment.

## Launch the Streamlit monitor

Keep each app in its existing folder:

```powershell
streamlit run .\threat_detector\app.py
```

or, for the original bundle layout:

```powershell
streamlit run .\location_threat_monitor\app.py
```

Each OSINT result now has an expandable explanation showing which fields were available and what they mean. The app explains visible output fields; it does not claim access to undocumented internal classifier logic.

## Run tests

```powershell
python -m unittest -v explainable_ai.tests.test_explainers
```

The tests use small JSON examples and do not call the model, access the network, or change the saved score.

## Design and safety constraints

- No detector changes, retraining, synthetic evidence, guessed time/location, or embedding-space comparison.
- No SHAP claim: current NLP output uses rule matching/entity extraction/source reconciliation; no trained text-classifier prediction interface is present in the supplied output schema.
- Scores remain exactly as emitted upstream and retain `validation_only` warnings. In particular, a synthetic timestamp must never be presented as a real incident time.
- Model confidence is not described as a calibrated probability unless independently calibrated.
- Human verification remains required before operational decisions.
