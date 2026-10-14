# Applying the AEGIS XAI patch

1. Close the running Streamlit app.
2. Back up `threat_detector\app.py` (the patch updates that app to add the XAI expander).
3. Extract `AEGIS_XAI_Implementation_Patch.zip` into the project root, preserving folder paths and allowing `threat_detector\app.py` to be replaced.
4. From the project root, run:

   ```powershell
   python -m compileall -q explainable_ai threat_detector\app.py
   python -m unittest -v explainable_ai.tests.test_explainers
   ```

5. Generate the XAI report from your current artifacts (adjust paths to match your real filenames):

   ```powershell
   python -m explainable_ai `
     --video-report "output\YOUTUBE_YouTubeCCTV196_gun_2_report.json" `
     --combined-intelligence "output\YOUTUBE_YouTubeCCTV196_gun_2_combined_intelligence.json" `
     --threat-score "output\threat_score.json" `
     --output "output\xai_explanation.json"
   ```

6. Launch your existing monitor:

   ```powershell
   streamlit run .\threat_detector\app.py
   ```

   Each returned OSINT threat row now has an **Explainable AI — Why is this result flagged?** expander.

## Visual XAI (optional dependency)

If `ultralytics` is available in the project's active virtual environment, use the one-frame EigenCAM command in `AEGIS_XAI_README.md`. This sandbox could not install `ultralytics` because the package index was unreachable, so the visual heatmap was not validated here. The non-visual XAI report and all regression tests are validated. Do not interpret the example JSON as a live score: its fusion input is explicitly marked `validation_only`.
