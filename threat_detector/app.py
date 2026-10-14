import json
import sys
from pathlib import Path

import streamlit as st

# Resolve the app's own folder, the legacy monitor folder, and the project root.
# Local sibling modules take precedence; the legacy path is a fallback for the integrated bundle.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _candidate in (_PROJECT_ROOT, _PROJECT_ROOT / "location_threat_monitor", Path(__file__).resolve().parent):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from incident_geocoder import geocode_location
from analysis_engine import analyze_threats

try:
    from explainable_ai.osint_explainer import explain_osint_result
except ImportError:
    explain_osint_result = None  # Keep the monitor usable if the XAI module is not installed.


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Location-Based Threat Monitor",
    page_icon="🌍",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("🌍 Location-Based Threat Monitor")

st.write(
    "Enter any location to identify and analyze "
    "threat-related news from the last 48 hours."
)

st.divider()


# ==========================================
# AEGIS XAI EVIDENCE DASHBOARD
# ==========================================

def render_xai_evidence_dashboard():
    """Render previously generated XAI artifacts without rerunning models."""
    output_dir = _PROJECT_ROOT / "Output"
    report_names = [
        "xai_model_evidence_validation.json",
        "xai_explanation.json",
        "xai_explanations.json",
    ]
    available_reports = [output_dir / name for name in report_names
                         if (output_dir / name).is_file()]

    with st.expander("🧠 AEGIS Explainable AI — Evidence Dashboard", expanded=False):
        st.caption(
            "Reads saved reports only. Opening this panel does not rerun detection, "
            "NLP, representation learning, or fusion. Validation-only data must not "
            "be interpreted as a live incident."
        )
        if not available_reports:
            st.info(
                "No saved XAI report found in the project Output folder. Generate "
                "one with `python -m explainable_ai`, then reopen this panel."
            )
            return

        selected_report = st.selectbox(
            "Saved XAI report",
            options=available_reports,
            format_func=lambda path: path.name,
            key="aegis_xai_report_select",
        )
        try:
            with selected_report.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read this XAI report: {exc}")
            return

        explanations = payload.get("explanations", {})
        if not isinstance(explanations, dict):
            st.warning("This report does not contain an explanations object.")
            st.json(payload)
            return

        fusion = explanations.get("fusion", {})
        video = explanations.get("video", {})
        osint = explanations.get("osint", {})
        visual = explanations.get("visual", {})

        if fusion.get("validation_only") is True:
            st.warning(
                "VALIDATION-ONLY RESULT — this score uses synthetic/demo or "
                "software-validation evidence. Do not use it as a real-world threat assessment."
            )
        elif fusion:
            st.info(
                "The fusion output is an automated heuristic assessment, not a "
                "calibrated probability or ground truth. Human verification is required."
            )

        score_col, level_col, section_col = st.columns(3)
        with score_col:
            score = fusion.get("threat_score")
            st.metric("Saved fusion score", f"{score:.3f}" if isinstance(score, (int, float)) else "Not included")
        with level_col:
            st.metric("Saved threat level", str(fusion.get("threat_level", "Not included")))
        with section_col:
            st.metric("Explanation sections", len(explanations))

        if fusion:
            with st.expander("Fusion: contributions and exclusions", expanded=True):
                st.write("**Score semantics:**", fusion.get("score_semantics", "Not specified"))
                st.write("**Threshold consistency:**", fusion.get("threshold_consistency_check", "Not provided"))
                contributions = fusion.get("contributions", [])
                if contributions:
                    st.markdown("**Contributing events**")
                    st.dataframe(contributions, use_container_width=True, hide_index=True)
                else:
                    st.info("No events contributed to this saved score.")
                exclusions = fusion.get("exclusions", fusion.get("excluded_events", []))
                if exclusions:
                    st.markdown("**Excluded events**")
                    st.dataframe(exclusions, use_container_width=True, hide_index=True)
                else:
                    st.caption("No exclusions were recorded in this explanation.")
                reasons = fusion.get("reported_reasons", fusion.get("reasons", []))
                if reasons:
                    st.markdown("**Engine-reported reasons**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                limitations = fusion.get("limitations", [])
                for limitation in limitations:
                    st.caption(f"Limitation: {limitation}")

        if visual:
            st.markdown("#### Visual explanation")
            overlay_raw = visual.get("overlay_path")
            overlay_path = Path(overlay_raw) if isinstance(overlay_raw, str) and overlay_raw else None
            if overlay_path and overlay_path.is_file():
                st.image(
                    str(overlay_path),
                    caption=(
                        f"EigenCAM-style feature activation map — frame "
                        f"{visual.get('frame_index', 'unknown')} at "
                        f"{visual.get('frame_timestamp_seconds', 'unknown')} seconds"
                    ),
                    use_container_width=True,
                )
            else:
                st.warning(f"Visual overlay path is not available: {overlay_raw or 'not recorded'}")
            visual_detections = visual.get("detections", [])
            if visual_detections:
                st.markdown("**Frame-level model predictions**")
                st.dataframe(visual_detections, use_container_width=True, hide_index=True)
            for limitation in visual.get("limitations", []):
                st.caption(f"Visual limitation: {limitation}")
            st.caption(
                "EigenCAM is class-agnostic. Its heatmap visualizes internal feature "
                "activations; it is not a class-specific causal explanation or proof of an incident."
            )

        if video:
            with st.expander("Video evidence details"):
                st.json(video)
        if osint:
            with st.expander("OSINT evidence explanation"):
                st.json(osint)
        st.caption(f"Loaded report: {selected_report}")


render_xai_evidence_dashboard()


# ==========================================
# LOCATION INPUT
# ==========================================

st.subheader("📍 Enter Monitoring Location")

location_input = st.text_input(
    "Location",
    placeholder="Example: Kelambakkam, Tamil Nadu, India"
)


# ==========================================
# ANALYZE LOCATION
# ==========================================

if st.button(
    "🚨 Analyze Location",
    type="primary",
    use_container_width=True
):

    if not location_input.strip():

        st.warning(
            "⚠️ Please enter a location first."
        )

    else:

        entered_location = location_input.strip()

        # --------------------------------------
        # Geocode user-entered location
        # --------------------------------------

        with st.spinner(
            "📍 Finding location coordinates..."
        ):

            location_data = geocode_location(
                entered_location
            )

        if not location_data:

            st.error(
                "❌ Location could not be found. "
                "Please enter a more specific location."
            )

        else:

            latitude = location_data["latitude"]
            longitude = location_data["longitude"]

            display_name = location_data.get(
                "display_name",
                entered_location
            )

            # ----------------------------------
            # Show resolved location
            # ----------------------------------

            st.success(
                "✅ Location found successfully!"
            )

            st.subheader(
                "📌 Monitoring Location"
            )

            st.write(
                f"**{display_name}**"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Latitude",
                    f"{latitude:.6f}"
                )

            with col2:

                st.metric(
                    "Longitude",
                    f"{longitude:.6f}"
                )

            st.divider()

            # ==================================
            # THREAT ANALYSIS
            # ==================================

            with st.spinner(
                "📰 Collecting and analyzing "
                "recent threat-related news..."
            ):

                results = analyze_threats(
                    entered_location,
                    latitude,
                    longitude
                )

            # ==================================
            # SUMMARY
            # ==================================

            st.subheader(
                "📊 Threat Analysis Summary"
            )

            total_threats = len(results)

            local_threats = sum(
                1
                for result in results
                if result.get(
                    "geographic_classification"
                ) == "Local"
            )

            nearby_threats = sum(
                1
                for result in results
                if result.get(
                    "geographic_classification"
                ) == "Nearby"
            )

            outside_threats = sum(
                1
                for result in results
                if result.get(
                    "geographic_classification"
                ) == "Outside"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "🚨 Threats Found",
                    total_threats
                )

            with col2:

                st.metric(
                    "🔴 Local",
                    local_threats
                )

            with col3:

                st.metric(
                    "🟠 Nearby",
                    nearby_threats
                )

            with col4:

                st.metric(
                    "⚪ Outside",
                    outside_threats
                )

            st.divider()

            # ==================================
            # THREAT RESULTS
            # ==================================

            st.subheader(
                "🚨 Detected Threats"
            )

            if not results:

                st.success(
                    "✅ No threat-related incidents "
                    "were identified for this location."
                )

            else:

                for number, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"### 🚨 Threat {number}"
                    )

                    st.write(
                        f"**{result.get('title', 'Unknown')}**"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.write("**Category**")

                        st.info(
                            result.get(
                                "category",
                                "Unknown"
                            )
                        )

                    with col2:

                        st.write("**Severity**")

                        severity = result.get(
                            "severity",
                            "Unknown"
                        )

                        if severity == "High":

                            st.error(
                                f"🔴 {severity}"
                            )

                        elif severity == "Medium":

                            st.warning(
                                f"🟠 {severity}"
                            )

                        else:

                            st.success(
                                f"🟢 {severity}"
                            )

                    with col3:

                        st.write(
                            "**Geographic Status**"
                        )

                        status = result.get(
                            "geographic_classification",
                            "Unknown"
                        )

                        if status == "Local":

                            st.error(
                                f"🔴 {status}"
                            )

                        elif status == "Nearby":

                            st.warning(
                                f"🟠 {status}"
                            )

                        else:

                            st.info(
                                f"⚪ {status}"
                            )

                    st.write(
                        f"📍 **Incident Location:** "
                        f"{result.get('incident_location', 'Unknown')}"
                    )

                    distance = result.get(
                        "distance_km"
                    )

                    if distance is not None:

                        st.write(
                            f"📏 **Distance from monitoring "
                            f"location:** "
                            f"{distance:.2f} km"
                        )

                    st.write(
                        f"📰 **Source:** "
                        f"{result.get('source', 'Unknown')}"
                    )

                    st.write(
                        f"🕐 **Published:** "
                        f"{result.get('publishedAt', 'Unknown')}"
                    )

                    description = result.get(
                        "description"
                    )

                    if description:

                        st.write(description)

                    article_url = result.get("url")

                    if article_url:

                        st.markdown(
                            f"[🔗 Read Full Article]"
                            f"({article_url})"
                        )

                    with st.expander("🔎 Explainable AI — Why is this result flagged?"):
                        if explain_osint_result is None:
                            st.info("Place the explainable_ai folder in the project root to enable this evidence trace.")
                        else:
                            try:
                                explanation = explain_osint_result(result)
                                st.write(explanation["summary"])
                                st.caption("Evidence trace from saved output fields; not a hidden-model or causal explanation.")
                                for evidence_item in explanation["evidence"]:
                                    st.markdown(f"**{evidence_item['field']}**: {evidence_item['value']}")
                                    st.caption(evidence_item["meaning"])
                                st.markdown("**Limitations and verification notes**")
                                for limitation in explanation["limitations"]:
                                    st.caption(f"• {limitation}")
                            except (TypeError, ValueError, KeyError) as exc:
                                st.warning(f"Could not explain this result's available fields: {exc}")

                    st.divider()