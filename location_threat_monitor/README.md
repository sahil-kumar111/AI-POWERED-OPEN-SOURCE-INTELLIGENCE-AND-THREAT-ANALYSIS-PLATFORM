# 🌍 Location-Based Threat Monitor

## Project Overview

Location-Based Threat Monitor is a Python-based application that provides
location-specific threat-related news.

The user manually enters a location such as Kolkata, Chennai, Mumbai,
Gurgaon, London, New York, etc.

The system automatically:

1. Converts the entered location into coordinates.
2. Collects recent threat-related news.
3. Detects potential threats.
4. Identifies the incident location.
5. Converts the incident location into coordinates.
6. Calculates the distance between the selected location and incident.
7. Classifies the incident as Local, Nearby, or Outside.
8. Displays relevant threats to the user.

The application does not depend on the user's physical location.

---

## Technologies Used

- Python
- Streamlit
- NewsAPI
- OpenStreetMap Nominatim
- Requests
- Python-dotenv

---

## Project Structure

```text
LocationThreatMonitor/
│
├── app.py
├── config.py
├── news_collector.py
├── threat_detector.py
├── location_filter.py
├── incident_location.py
├── incident_geocoder.py
├── distance_calculator.py
├── geographic_relevance.py
├── duplicate_filter.py
├── analysis_engine.py
├── geolocation.py
│
├── test_api_key.py
├── test_news.py
├── test_threat_detector.py
├── test_location_filter.py
├── test_incident_location.py
├── test_incident_geocoder.py
├── test_distance.py
├── test_geographic_relevance.py
├── test_analysis_engine.py
│
├── requirements.txt
└── README.md