import requests


def reverse_geocode(latitude, longitude):
    """
    Convert latitude and longitude into
    city, state and country using OpenStreetMap.
    """

    url = "https://nominatim.openstreetmap.org/reverse"

    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "json",
        "zoom": 18,
        "addressdetails": 1
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

        address = data.get("address", {})

        # Try several possible fields because
        # different locations use different names.

        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("suburb")
            or address.get("city_district")
            or address.get("county")
        )

        state = address.get("state")

        country = address.get("country")

        return {
            "latitude": latitude,
            "longitude": longitude,
            "city": city,
            "state": state,
            "country": country
        }

    except requests.RequestException as e:

        print("Geocoding error:", e)

        return None