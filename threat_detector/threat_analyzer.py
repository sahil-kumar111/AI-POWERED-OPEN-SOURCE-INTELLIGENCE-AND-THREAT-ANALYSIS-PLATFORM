from news_collector import collect_news
from threat_detector import detect_threat
from location_filter import check_location_relevance


def analyze_location(location):

    print("\n===================================")
    print("🌍 LOCATION THREAT ANALYZER")
    print("===================================")

    print(f"\n📍 Location: {location}")
    print("📰 Collecting news from the last 48 hours...")

    articles = collect_news(location)

    print(f"📰 Articles collected: {len(articles)}")

    results = []

    for article in articles:

        # Step 1: Check location relevance
        location_relevant = check_location_relevance(
            article,
            location
        )

        # Step 2: Detect threat
        threat_result = detect_threat(article)

        # Only keep articles that are both
        # location relevant and threats
        if location_relevant and threat_result["is_threat"]:

            result = {
                "title": article["title"],
                "description": article["description"],
                "source": article["source"],
                "publishedAt": article["publishedAt"],
                "url": article["url"],
                "category": threat_result["category"],
                "severity": threat_result["severity"],
                "location": location
            }

            results.append(result)

    return results