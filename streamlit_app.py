"""AEGIS - Streamlit Threat Intelligence Dashboard.

A clean, modern, responsive interface providing:
- Video upload (MP4, AVI, MOV) & location search with presets
- Progress bar and stage indicators
- Section A: Event Overview (ATI score, level, threat detection, calculation rationale)
- Section B: Person Detection & Crowd Analysis (ByteTrack unique individuals vs detections count, timeline chart)
- Section C: Threat Detection (Weapon, Fire, Smoke, Suspicious Activity cards, dominant category, severity ranking)
- Section D: Heatmaps & Visualizations (Spatial EigenCAM overlay, temporal charts, annotated video player)
- Section E: Evidence Fusion & Explainable AI (SHAP feature attribution, tactical reasoning, 3-tier directives)
- Section F: Final Intelligence Report (Answers to all 7 questions, markdown viewer, exportable artifacts)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import altair as alt
import pandas as pd
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="AEGIS | AI Threat Analysis Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    run_complete_pipeline,
)

# Custom Styling for modern dark cybersecurity aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .threat-badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .threat-badge-high {
        background-color: #f97316;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .threat-badge-medium {
        background-color: #eab308;
        color: black;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .threat-badge-low {
        background-color: #22c55e;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown('<div class="main-header">🛡️ AEGIS Threat Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Multimodal Threat Detection, ByteTrack Tracking, OSINT Fusion & Explainable AI</div>',
    unsafe_allow_html=True,
)

# Session state initialization
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False

# Sidebar: Video & Location Configuration
st.sidebar.header("⚙️ Surveillance Feeds & Location")

location_preset = st.sidebar.selectbox(
    "Location Presets",
    [
        "Custom Input",
        "Connaught Place, New Delhi",
        "Colaba, Mumbai",
        "MG Road, Bengaluru",
        "Park Street, Kolkata",
    ],
    index=1,
)

if location_preset == "Custom Input":
    location_input = st.sidebar.text_input("Surveillance Location", "Connaught Place, New Delhi")
else:
    location_input = location_preset

video_source = st.sidebar.radio(
    "Video Source",
    ["Use Bundled Scenario Video (CCTV Armed Assault)", "Upload Custom Video"],
)

uploaded_video_path = None
if video_source == "Upload Custom Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Video (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_video_path = temp_path
        st.sidebar.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.1f} MB)")
else:
    sample_path = DEFAULT_INPUT_DIR / "YOUTUBE_YouTubeCCTV196_gun_2.mp4"
    if sample_path.is_file():
        uploaded_video_path = sample_path
        st.sidebar.info("Using verified sample footage: `YOUTUBE_YouTubeCCTV196_gun_2.mp4` (90 frames)")

venue_capacity = st.sidebar.number_input("Venue Crowd Capacity Limit", min_value=1, max_value=500, value=15)
frame_skip = st.sidebar.slider("Frame Skip Interval (1 = High Fidelity, 2-3 = Fast)", 1, 5, 2)

analyze_btn = st.sidebar.button(
    "🚨 Analyze Video & Threats",
    type="primary",
    disabled=st.session_state.is_analyzing,
    use_container_width=True,
)

# Trigger Analysis
if analyze_btn:
    if not location_input.strip():
        st.error("Please provide a surveillance location.")
    elif not uploaded_video_path or not uploaded_video_path.is_file():
        st.error("Please provide a valid video file.")
    else:
        st.session_state.is_analyzing = True
        progress_bar = st.progress(0)
        status_text = st.empty()

        def stream_progress(stage: str, pct: int):
            progress_bar.progress(pct)
            status_text.info(f"**[{pct}%]** {stage}")

        try:
            with st.spinner("Executing AEGIS Multimodal Threat Pipeline..."):
                result = run_complete_pipeline(
                    location=location_input.strip(),
                    video_path=uploaded_video_path,
                    venue_capacity=venue_capacity,
                    frame_skip=frame_skip,
                    progress_callback=stream_progress,
                )
            st.session_state.analysis_result = result
            progress_bar.progress(100)
            status_text.success("✅ Analysis completed successfully!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        finally:
            st.session_state.is_analyzing = False

# Render Dashboard
res = st.session_state.analysis_result

# If no active session result, load recent saved result from output directory if available
if not res and (DEFAULT_OUTPUT_DIR / "threat_summary.json").is_file():
    try:
        with open(DEFAULT_OUTPUT_DIR / "threat_summary.json", "r", encoding="utf-8") as f:
            res = json.load(f)
            st.info("Loaded most recent verified analysis from disk.")
    except Exception:
        pass

if res:
    overview = res.get("event_overview", {})
    ati = overview.get("aegis_threat_index", 0.0)
    level = overview.get("threat_level", "LOW")
    dominant = res.get("threat_detection", {}).get("dominant_threat_category", "none")
    p_analysis = res.get("person_and_crowd_analysis", {})
    ranked_threats = res.get("threat_detection", {}).get("ranked_threat_categories", [])
    xai = res.get("evidence_fusion", {})
    report_data = res.get("final_intelligence_report", {})
    seven_q = report_data.get("seven_questions", {})

    # Top Status Banner
    badge_class = (
        "threat-badge-critical" if level == "CRITICAL"
        else "threat-badge-high" if level == "HIGH"
        else "threat-badge-medium" if level == "MEDIUM"
        else "threat-badge-low"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("AEGIS Threat Index (ATI)", f"{ati} / 100")
    with col2:
        st.markdown(f"**Threat Level**<br><span class='{badge_class}'>{level}</span>", unsafe_allow_html=True)
    with col3:
        st.metric("Dominant Threat", dominant.upper())
    with col4:
        st.metric("Unique Tracked People", p_analysis.get("unique_people_count", 0))
    with col5:
        st.metric("Pipeline Duration", f"{res.get('execution_time_seconds', 0)}s")

    st.markdown("---")

    # Structured Tabs
    tabs = st.tabs([
        "📊 Event Overview",
        "👥 Person & Crowd Tracking",
        "🎯 Threat Detection",
        "🗺️ Heatmaps & Visualizations",
        "🧠 Evidence Fusion & XAI",
        "📋 Final Intelligence Report",
    ])

    # TAB 1: EVENT OVERVIEW
    with tabs[0]:
        st.subheader("Event Overview")
        ov_col1, ov_col2 = st.columns([1, 1])

        with ov_col1:
            st.markdown(f"**Video Feed:** `{overview.get('video_name')}`")
            st.markdown(f"**Surveillance Location:** {overview.get('location')}")
            st.markdown(f"**Coordinates:** Latitude `{overview.get('coordinates', {}).get('latitude')}`, Longitude `{overview.get('coordinates', {}).get('longitude')}`")
            st.markdown(f"**Threat Classification:** `{overview.get('threat_classification')}`")
            st.markdown(f"**Analysis Status:** `{overview.get('analysis_status')}`")

        with ov_col2:
            st.markdown("#### Threat Score Rationale")
            st.info(overview.get("calculation_explanation"))
            st.caption(
                "Note: AEGIS Threat Index (ATI) represents an analytical risk estimate calculated from multimodal sensors "
                "and infrastructure proximity, not a calibrated real-world probability. Human verification required."
            )

    # TAB 2: PERSON & CROWD ANALYSIS
    with tabs[1]:
        st.subheader("Person Detection and Crowd Analysis")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Unique Individuals Tracked", p_analysis.get("unique_people_count", 0))
        p_col2.metric("Total Person Detections", p_analysis.get("total_person_detections", 0))
        p_col3.metric("Peak Concurrent Persons", p_analysis.get("peak_concurrent_persons", 0))
        p_col4.metric("Venue Capacity", p_analysis.get("venue_capacity", 15))

        st.info(f"💡 **Association Clarity:** {p_analysis.get('detections_vs_unique_explanation')}")

        # Frame-wise Person Activity Timeline Chart
        timeline_data = res.get("heatmaps_and_visualizations", {}).get("frame_timeline", [])
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            df_chart = pd.DataFrame({
                "Timestamp (sec)": df_timeline["timestamp_sec"],
                "Concurrent People Count": df_timeline["person_count"],
                "Venue Capacity Limit": p_analysis.get("venue_capacity", 15),
            })
            chart = alt.Chart(df_chart).transform_fold(
                ["Concurrent People Count", "Venue Capacity Limit"],
                as_=["Metric", "Value"],
            ).mark_line().encode(
                x="Timestamp (sec):Q",
                y="Value:Q",
                color=alt.Color("Metric:N", scale=alt.Scale(range=["#38bdf8", "#ef4444"])),
            ).properties(
                title="Frame-by-Frame Person Density & Capacity Tracking",
                width="container",
                height=300,
            )
            st.altair_chart(chart, use_container_width=True)

    # TAB 3: THREAT DETECTION
    with tabs[2]:
        st.subheader("Threat Detection Breakdown")
        st.markdown(f"**Dominant Threat Category:** `{dominant.upper()}`")

        threat_cols = st.columns(len(ranked_threats))
        for idx, threat in enumerate(ranked_threats):
            with threat_cols[idx]:
                status_color = "🔴" if threat["status"] == "DETECTED" else "🟢"
                st.markdown(f"### {status_color} {threat['category'].title()}")
                st.metric("Status", threat["status"])
                st.metric("Confidence", f"{threat['confidence']:.2f}")
                st.metric("Detections", f"{threat['detection_count']} ({threat['frame_percentage']}%)")
                st.caption(threat["explanation"])

        st.markdown("#### Ranked Threat Categories by Severity")
        df_ranked = pd.DataFrame([
            {
                "Rank": t["rank"],
                "Category": t["category"].upper(),
                "Status": t["status"],
                "Max Confidence": t["confidence"],
                "Detections": t["detection_count"],
                "Frame %": f"{t['frame_percentage']}%",
                "Severity": t["severity"],
            }
            for t in ranked_threats
        ])
        st.dataframe(df_ranked, use_container_width=True, hide_index=True)

    # TAB 4: HEATMAPS & VISUALIZATIONS
    with tabs[3]:
        st.subheader("Data-Backed Heatmaps & Visualizations")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("#### 🗺️ Spatial Threat Activation Heatmap (EigenCAM)")
            highlight_path = DEFAULT_OUTPUT_DIR / "threat_highlight.jpg"
            if highlight_path.is_file():
                st.image(
                    str(highlight_path),
                    caption="Class-agnostic EigenCAM feature activation overlay on maximum-confidence threat frame.",
                    use_container_width=True,
                )
                st.caption("Visualizes internal layer activations corresponding to detected threat signatures.")
            else:
                st.info("Visual activation map not generated or unavailable.")

        with viz_col2:
            st.markdown("#### 🎥 Annotated CCTV Video Preview")
            vid_artifact = res.get("heatmaps_and_visualizations", {}).get("annotated_video_artifact")
            vid_path = DEFAULT_OUTPUT_DIR / vid_artifact if vid_artifact else None
            if vid_path and vid_path.is_file():
                st.video(str(vid_path))
                st.caption("Bounding boxes with confidence metrics and ByteTrack tracking IDs.")
            else:
                st.info("Annotated video stream preview not available.")

        # Temporal Threat Activity Chart
        st.markdown("#### ⏱️ Frame-wise Threat Activity Over Time")
        if timeline_data:
            threat_records = []
            for item in timeline_data:
                sec = item["timestamp_sec"]
                t_counts = item.get("threat_counts", {})
                for c_name, c_val in t_counts.items():
                    threat_records.append({
                        "Time (s)": sec,
                        "Threat Category": c_name.title(),
                        "Detection Count": c_val,
                    })
            df_threat_timeline = pd.DataFrame(threat_records)
            threat_chart = alt.Chart(df_threat_timeline).mark_area(opacity=0.6).encode(
                x="Time (s):Q",
                y="Detection Count:Q",
                color=alt.Color("Threat Category:N", scale=alt.Scale(range=["#ef4444", "#f97316", "#a855f7"])),
            ).properties(
                title="Temporal Threat Occurrence across Surveillance Duration",
                width="container",
                height=260,
            )
            st.altair_chart(threat_chart, use_container_width=True)

    # TAB 5: EVIDENCE FUSION & XAI
    with tabs[4]:
        st.subheader("Multimodal Evidence Fusion & Explainable AI (SHAP-Style)")
        fa_list = xai.get("feature_attribution", [])

        if fa_list:
            df_fa = pd.DataFrame(fa_list)
            fa_chart = alt.Chart(df_fa).mark_bar().encode(
                x=alt.X("percentage_importance:Q", title="Percentage Contribution to ATI Score (%)"),
                y=alt.Y("factor:N", sort="-x", title="Modality / Threat Factor"),
                color=alt.Color("modality:N"),
                tooltip=["factor", "modality", "percentage_importance", "evidence_reason"],
            ).properties(height=240)
            st.altair_chart(fa_chart, use_container_width=True)

        st.markdown("#### 📋 Actionable Operational Directives")
        rec = report_data.get("recommendations", {})
        dir_col1, dir_col2, dir_col3 = st.columns(3)
        with dir_col1:
            st.markdown("**⏱️ 0 - 5 Minutes (Immediate)**")
            for act in rec.get("immediate_directives_0_to_5_min", []):
                st.markdown(f"- {act}")
        with dir_col2:
            st.markdown("**⏱️ 5 - 30 Minutes (Containment)**")
            for act in rec.get("area_management_5_to_30_min", []):
                st.markdown(f"- {act}")
        with dir_col3:
            st.markdown("**🛡️ Investigation Protocols**")
            for act in rec.get("intelligence_and_investigation", []):
                st.markdown(f"- {act}")

        st.markdown("#### 🗺️ Nearby Critical Infrastructure Assets")
        infra_list = xai.get("nearby_critical_infrastructure", [])
        if infra_list:
            st.dataframe(pd.DataFrame(infra_list)[["name", "category", "distance_km", "bearing"]], hide_index=True)
        else:
            st.caption("No critical infrastructure within 5km radius.")

    # TAB 6: FINAL INTELLIGENCE REPORT
    with tabs[5]:
        st.subheader("Executive Intelligence Briefing")

        st.markdown("### 📋 7 Key Intelligence Answers")
        st.markdown(f"**1. What is happening in the video?**  \n{seven_q.get('q1_what_is_happening')}")
        st.markdown(f"**2. How many people are visible or reliably tracked?**  \n{seven_q.get('q2_how_many_people')}")
        st.markdown(f"**3. Which threats were detected?**  \n{seven_q.get('q3_which_threats_detected')}")
        st.markdown(f"**4. Which threat category is dominant?**  \n{seven_q.get('q4_dominant_threat')}")
        st.markdown(f"**5. What is the overall threat severity and why?**  \n{seven_q.get('q5_overall_threat_severity_and_why')}")
        st.markdown(f"**6. What evidence supports the conclusion?**  \n{seven_q.get('q6_evidence_supporting_conclusion')}")
        st.markdown(f"**7. What additional information or limitations affect the result?**  \n{seven_q.get('q7_limitations_and_additional_info')}")

        st.markdown("---")
        with st.expander("📄 View Full Markdown Report", expanded=False):
            st.markdown(report_data.get("executive_summary_markdown", ""))

        # Download Buttons
        md_content = report_data.get("executive_summary_markdown", "")
        st.download_button(
            "⬇️ Download Intelligence Report (.md)",
            data=md_content,
            file_name="aegis_intelligence_report.md",
            mime="text/markdown",
        )
else:
    st.info("👈 Select a video feed and location in the sidebar, then click **🚨 Analyze Video & Threats** to begin.")
