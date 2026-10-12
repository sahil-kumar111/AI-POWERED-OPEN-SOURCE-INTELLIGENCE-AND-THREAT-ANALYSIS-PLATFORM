import requests


def geocode_location(location):
    """
    Convert a location name into latitude and longitude
    using OpenStreetMap Nominatim.
    """

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "LocationThreatMonitor/1.0"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            return None

        result = data[0]

        return {
            "location": location,
            "latitude": float(result["lat"]),
            "longitude": float(result["lon"]),
            "display_name": result.get("display_name")
        }

    except requests.RequestException as error:

        print("❌ Geocoding error:", error)

        return None