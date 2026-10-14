# ==========================================
# INCIDENT LOCATION DETECTOR
# ==========================================

import re


INDIAN_LOCATIONS = [
    "Chennai",
    "Mumbai",
    "Delhi",
    "New Delhi",
    "Bengaluru",
    "Bangalore",
    "Hyderabad",
    "Kolkata",
    "Pune",
    "Ahmedabad",
    "Jaipur",
    "Lucknow",
    "Kanpur",
    "Nagpur",
    "Indore",
    "Bhopal",
    "Patna",
    "Surat",
    "Vadodara",
    "Coimbatore",
    "Madurai",
    "Tiruchirappalli",
    "Trichy",
    "Salem",
    "Tirunelveli",
    "Vellore",
    "Kanchipuram",
    "Jalandhar",
    "Mohali",
    "Amritsar",
    "Gurugram",
    "Gurgaon",
    "Noida",
    "Ghaziabad",
    "Visakhapatnam",
    "Vijayawada",
    "Mysuru",
    "Mysore",
    "Kochi",
    "Thiruvananthapuram",
    "Kozhikode",
    "Nashik",
    "Rajkot",
    "Ranchi",
    "Bhubaneswar",
    "Guwahati",
    "Chandigarh",
    "Kodangal",
    "Janampally"
]


def extract_incident_location(article):
    """
    Find the most likely incident location.

    Priority:
    1. Location after 'near'
    2. Location after 'at'
    3. Location after 'in'
    4. Location after 'outside'
    5. Other valid location mentions

    Destination phrases such as 'to Chennai'
    are ignored.
    """

    title = article.get("title") or ""
    description = article.get("description") or ""

    text = f"{title} {description}"
    text_lower = text.lower()

    # ------------------------------------------
    # STEP 1
    # Look for locations after strong incident
    # indicators.
    # ------------------------------------------

    strong_indicators = [
        "near",
        "at",
        "outside",
        "around",
        "in"
    ]

    strong_matches = []

    for indicator in strong_indicators:

        for location in INDIAN_LOCATIONS:

            pattern = (
                rf"\b{re.escape(indicator)}\s+"
                rf"{re.escape(location.lower())}\b"
            )

            if re.search(pattern, text_lower):

                strong_matches.append(location)

    if strong_matches:
        return [strong_matches[0]]

    # ------------------------------------------
    # STEP 2
    # Find locations mentioned in the article
    # ------------------------------------------

    candidates = []

    for location in INDIAN_LOCATIONS:

        pattern = rf"\b{re.escape(location.lower())}\b"

        match = re.search(pattern, text_lower)

        if not match:
            continue

        position = match.start()

        # Text immediately before the location
        before = text_lower[
            max(0, position - 40):position
        ]

        # --------------------------------------
        # Ignore destination references
        # --------------------------------------

        destination_patterns = [
            rf"\bto\s+{re.escape(location.lower())}\b",
            rf"\btowards\s+{re.escape(location.lower())}\b",
            rf"\bheading\s+to\s+{re.escape(location.lower())}\b",
            rf"\bbound\s+for\s+{re.escape(location.lower())}\b",
            rf"\btravelling\s+to\s+{re.escape(location.lower())}\b",
            rf"\btraveling\s+to\s+{re.escape(location.lower())}\b"
        ]

        is_destination = False

        for pattern in destination_patterns:

            if re.search(pattern, before + location.lower()):

                is_destination = True
                break

        if not is_destination:

            candidates.append({
                "location": location,
                "position": position
            })

    # ------------------------------------------
    # STEP 3
    # Return first non-destination location
    # ------------------------------------------

    if candidates:

        return [candidates[0]["location"]]

    return []