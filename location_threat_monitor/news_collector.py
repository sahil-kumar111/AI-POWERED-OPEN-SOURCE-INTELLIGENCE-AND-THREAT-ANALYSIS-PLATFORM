import requests
from datetime import datetime, timedelta, timezone

from config import NEWS_API_KEY


NEWS_API_URL = "https://newsapi.org/v2/everything"


def collect_news(location):
    """
    Collect threat-related news for ANY user-entered location
    from the last 48 hours.
    """

    location = location.strip()

    # ------------------------------------------
    # Last 48 hours
    # ------------------------------------------

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=48)

    # ------------------------------------------
    # Threat keywords
    # ------------------------------------------

    threat_keywords = (
        "attack OR violence OR terrorism OR terrorist OR "
        "bomb OR explosion OR fire OR flood OR earthquake OR "
        "cyclone OR disaster OR emergency OR riot OR "
        "shooting OR accident OR cyberattack OR "
        "landslide OR tsunami OR outbreak"
    )

    # ------------------------------------------
    # Search query
    # ------------------------------------------

    query = (
        f'"{location}" AND ({threat_keywords})'
    )

    # ------------------------------------------
    # API parameters
    # ------------------------------------------

    params = {
        "q": query,
        "from": start_time.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "to": end_time.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100
    }

    headers = {
        "X-Api-Key": NEWS_API_KEY
    }

    # ------------------------------------------
    # Send request
    # ------------------------------------------

    try:

        response = requests.get(
            NEWS_API_URL,
            params=params,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        articles = data.get(
            "articles",
            []
        )

        # --------------------------------------
        # Format results
        # --------------------------------------

        results = []

        seen_urls = set()

        for article in articles:

            url = article.get("url")

            # Remove duplicate articles
            if url and url in seen_urls:
                continue

            if url:
                seen_urls.add(url)

            results.append({
                "title": article.get("title"),
                "description": article.get(
                    "description"
                ),
                "source": article.get(
                    "source",
                    {}
                ).get("name"),
                "publishedAt": article.get(
                    "publishedAt"
                ),
                "url": url
            })

        return results

    except requests.RequestException as error:

        print(
            "❌ NewsAPI Error:",
            error
        )

        return []