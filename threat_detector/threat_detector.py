# ==========================================
# THREAT DETECTOR
# ==========================================


THREAT_KEYWORDS = {

    "Security": [
        "terrorism",
        "terrorist",
        "attack",
        "bomb",
        "bomb threat",
        "shooting",
        "gunfire",
        "riot",
        "violence",
        "terror",
        "explosion"
    ],

    "Natural Disaster": [
        "earthquake",
        "flood",
        "flash flood",
        "cyclone",
        "storm",
        "landslide",
        "tsunami",
        "tornado",
        "heavy rainfall",
        "extreme weather"
    ],

    "Fire / Industrial": [
        "fire",
        "factory fire",
        "industrial accident",
        "chemical leak",
        "gas leak",
        "explosion",
        "building collapse",
        "industrial disaster"
    ],

    "Public Health": [
        "outbreak",
        "epidemic",
        "pandemic",
        "disease",
        "contamination",
        "food poisoning",
        "health emergency"
    ],

    "Cyber / Technology": [
        "cyberattack",
        "cyber attack",
        "ransomware",
        "hacking",
        "data breach",
        "cyber threat",
        "critical infrastructure attack"
    ],

    "Infrastructure / Transport": [
        "bridge collapse",
        "road collapse",
        "train accident",
        "train derailment",
        "bus accident",
        "truck accident",
        "traffic disruption",
        "power outage",
        "infrastructure failure"
    ]
}


HIGH_SEVERITY_KEYWORDS = [
    "terrorist attack",
    "terror attack",
    "bomb blast",
    "bomb threat",
    "mass shooting",
    "major earthquake",
    "major flood",
    "cyclone",
    "tsunami",
    "chemical leak",
    "gas leak",
    "building collapse",
    "critical infrastructure attack",
    "major explosion"
]


MEDIUM_SEVERITY_KEYWORDS = [
    "fire",
    "flood",
    "accident",
    "outbreak",
    "riot",
    "cyberattack",
    "cyber attack",
    "data breach",
    "landslide",
    "heavy rainfall",
    "traffic disruption",
    "power outage"
]


def detect_threat(article):
    """
    Analyze an article and determine:
    - whether it represents a threat
    - threat category
    - severity
    """

    title = article.get("title") or ""
    description = article.get("description") or ""

    text = (title + " " + description).lower()

    detected_categories = []

    # ------------------------------------------
    # Detect threat category
    # ------------------------------------------

    for category, keywords in THREAT_KEYWORDS.items():

        for keyword in keywords:

            if keyword.lower() in text:
                detected_categories.append(category)
                break

    # Remove duplicate categories
    detected_categories = list(set(detected_categories))

    # ------------------------------------------
    # If no threat keyword found
    # ------------------------------------------

    if not detected_categories:

        return {
            "is_threat": False,
            "category": "None",
            "severity": "None"
        }

    # ------------------------------------------
    # Determine severity
    # ------------------------------------------

    severity = "Low"

    for keyword in HIGH_SEVERITY_KEYWORDS:

        if keyword.lower() in text:
            severity = "High"
            break

    else:

        for keyword in MEDIUM_SEVERITY_KEYWORDS:

            if keyword.lower() in text:
                severity = "Medium"
                break

    # ------------------------------------------
    # Return result
    # ------------------------------------------

    return {
        "is_threat": True,
        "category": ", ".join(detected_categories),
        "severity": severity
    }