from news_collector import collect_news
from threat_detector import detect_threat
from incident_location import extract_incident_location
from incident_geocoder import geocode_location
from geographic_relevance import check_geographic_relevance
from duplicate_filter import remove_duplicate_articles


def analyze_threats(location, user_latitude, user_longitude):
    """
    Analyze threat-related news for the user-selected location.

    Only incidents that are Local or Nearby are returned.
    Incidents classified as Outside are discarded.
    """

    # Step 1: Collect news for the selected location
    articles = collect_news(location)

    # Step 2: Remove duplicate articles
    articles = remove_duplicate_articles(articles)

    results = []

    # Step 3: Analyze every article
    for article in articles:

        # Detect whether the article actually describes a threat
        threat = detect_threat(article)

        if not threat["is_threat"]:
            continue

        # Find the actual incident location mentioned in the article
        incident_locations = extract_incident_location(article)

        if not incident_locations:
            continue

        # Step 4: Check each possible incident location
        for incident_location in incident_locations:

            # Convert incident location into coordinates
            geo = geocode_location(incident_location)

            if not geo:
                continue

            # Calculate geographic relevance
            geographic_result = check_geographic_relevance(
                user_latitude,
                user_longitude,
                geo["latitude"],
                geo["longitude"]
            )

            # IMPORTANT:
            # Ignore incidents that are outside the monitoring area
            if geographic_result["classification"] == "Outside":
                continue

            # Step 5: Create final result
            result = {
                "title": article["title"],
                "description": article["description"],
                "source": article["source"],
                "publishedAt": article["publishedAt"],
                "url": article["url"],
                "category": threat["category"],
                "severity": threat["severity"],
                "incident_location": incident_location,
                "incident_latitude": geo["latitude"],
                "incident_longitude": geo["longitude"],
                "distance_km": geographic_result["distance_km"],
                "geographic_classification": geographic_result["classification"]
            }

            results.append(result)

    return results