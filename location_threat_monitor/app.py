import streamlit as st

from incident_geocoder import geocode_location
from analysis_engine import analyze_threats


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

                    st.divider()