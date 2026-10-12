# ==========================================
# THREAT DETECTOR
# ==========================================

# Only detects the specific physical/public-safety
# threats required for this project.
#
# NOT included:
# - Natural disasters
# - Road accidents
# - Traffic accidents
# - Cyber incidents
# - General political/news stories
# - Disease outbreaks


THREAT_KEYWORDS = {

    # ------------------------------------------
    # Weapons and Explosives
    # ------------------------------------------

    "Weapons / Explosives": [
        "weapon found",
        "weapons found",
        "weapon recovered",
        "weapons recovered",
        "weapon seized",
        "weapons seized",
        "firearm found",
        "firearms found",
        "firearm recovered",
        "firearms recovered",
        "gun found",
        "guns found",
        "gun recovered",
        "pistol found",
        "rifle found",
        "ammunition found",
        "ammunition seized",

        "bomb found",
        "bomb recovered",
        "bomb seized",
        "bomb threat",
        "bomb scare",
        "explosive found",
        "explosives found",
        "explosive recovered",
        "explosives recovered",
        "explosive device",
        "explosive material",
        "suspicious package",
        "suspicious object",
        "unattended bag",
        "unattended package"
    ],


    # ------------------------------------------
    # Fire and Smoke
    # ------------------------------------------

    "Fire / Smoke": [
        "fire broke out",
        "fire breaks out",
        "fire erupted",
        "fire reported",
        "fire engulfed",
        "massive fire",
        "major fire",
        "building fire",
        "house fire",
        "shop fire",
        "market fire",
        "factory fire",
        "warehouse fire",
        "industrial fire",
        "vehicle fire",

        "heavy smoke",
        "thick smoke",
        "smoke billowing",
        "smoke seen",
        "smoke detected",
        "large plume of smoke",
        "smoke fills"
    ],


    # ------------------------------------------
    # Violence / Security
    # ------------------------------------------

    "Violence / Security": [
        "terrorist attack",
        "terror attack",
        "terrorist assault",
        "armed attack",
        "shooting",
        "gunfire",
        "firing",
        "stabbing",
        "knife attack",
        "violent attack",
        "violent clash",
        "mob violence",
        "riot",
        "rioting",
        "violent riot",
        "serious violence",
        "hostage situation"
    ],


    # ------------------------------------------
    # Crowd / Public Safety
    # ------------------------------------------

    "Crowd / Public Safety": [
        "crowd surge",
        "crowd crush",
        "crowd stampede",
        "stampede",
        "mass stampede",
        "crowd panic",
        "panic among crowd",
        "overcrowding",
        "overcrowded",
        "dangerously crowded",
        "massive crowd",
        "huge crowd",
        "large crowd",
        "crowd gathered",
        "crowds gathered",
        "mass gathering"
    ],


    # ------------------------------------------
    # Chemical / Gas
    # ------------------------------------------

    "Chemical / Gas": [
        "chemical leak",
        "chemical leakage",
        "chemical spill",
        "toxic leak",
        "toxic gas",
        "gas leak",
        "gas leakage",
        "gas escape",
        "hazardous chemical",
        "hazardous material"
    ],


    # ------------------------------------------
    # Explosion
    # ------------------------------------------

    "Explosion": [
        "explosion occurred",
        "explosion reported",
        "explosion rocked",
        "explosion ripped",
        "major explosion",
        "massive explosion",
        "industrial explosion",
        "factory explosion",
        "blast occurred",
        "major blast",
        "massive blast"
    ],


    # ------------------------------------------
    # Industrial / Structural
    # ------------------------------------------

    "Industrial / Structural": [
        "industrial accident",
        "industrial incident",
        "factory accident",
        "factory incident",
        "building collapse",
        "building collapsed",
        "structure collapse",
        "structure collapsed",
        "wall collapse",
        "wall collapsed",
        "bridge collapse",
        "bridge collapsed"
    ]
}


# ==========================================
# HIGH SEVERITY
# ==========================================

HIGH_SEVERITY_KEYWORDS = [

    "bomb blast",
    "bomb attack",
    "bomb threat",
    "explosive device",
    "explosive found",
    "explosives found",

    "terrorist attack",
    "terror attack",
    "armed attack",

    "mass shooting",
    "hostage situation",

    "major riot",
    "violent riot",
    "mob violence",

    "crowd crush",
    "stampede",
    "mass stampede",

    "massive fire",
    "major fire",

    "major explosion",
    "massive explosion",
    "major blast",
    "massive blast",

    "toxic gas",
    "hazardous chemical",

    "building collapse",
    "bridge collapse"
]


# ==========================================
# MEDIUM SEVERITY
# ==========================================

MEDIUM_SEVERITY_KEYWORDS = [

    # Weapons
    "weapon found",
    "weapons found",
    "weapon recovered",
    "weapons recovered",
    "weapon seized",
    "weapons seized",
    "firearm found",
    "firearms found",
    "gun found",
    "guns found",
    "bomb found",
    "bomb recovered",
    "bomb seized",
    "bomb scare",
    "explosive found",
    "explosives found",
    "suspicious package",
    "suspicious object",

    # Fire / smoke
    "fire broke out",
    "fire breaks out",
    "fire erupted",
    "fire reported",
    "building fire",
    "factory fire",
    "warehouse fire",
    "industrial fire",
    "vehicle fire",
    "heavy smoke",
    "thick smoke",

    # Violence
    "shooting",
    "gunfire",
    "firing",
    "stabbing",
    "knife attack",
    "violent attack",
    "violent clash",
    "riot",
    "rioting",
    "serious violence",

    # Crowd
    "crowd surge",
    "large crowd",
    "huge crowd",
    "massive crowd",
    "overcrowding",
    "overcrowded",
    "crowd gathered",
    "mass gathering",

    # Chemical / gas
    "chemical leak",
    "chemical leakage",
    "chemical spill",
    "gas leak",
    "gas leakage",
    "gas escape",

    # Explosion
    "explosion occurred",
    "explosion reported",
    "industrial explosion",
    "factory explosion",
    "blast occurred",

    # Industrial / structural
    "industrial accident",
    "industrial incident",
    "factory accident",
    "factory incident",
    "building collapsed",
    "structure collapsed",
    "wall collapsed",
    "bridge collapsed"
]


# ==========================================
# DETECT THREAT
# ==========================================

def detect_threat(article):
    """
    Analyze an article and determine:

    - whether it represents a relevant threat
    - threat category
    - severity

    The detector intentionally excludes:
    - natural disasters
    - road accidents
    - traffic accidents
    - cyber incidents
    - public-health outbreaks
    """

    title = article.get("title") or ""
    description = article.get("description") or ""

    text = (title + " " + description).lower()

    detected_categories = []

    # ------------------------------------------
    # Detect threat categories
    # ------------------------------------------

    for category, keywords in THREAT_KEYWORDS.items():

        for keyword in keywords:

            if keyword.lower() in text:
                detected_categories.append(category)
                break

    # ------------------------------------------
    # No relevant threat
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

    # High severity
    for keyword in HIGH_SEVERITY_KEYWORDS:

        if keyword.lower() in text:
            severity = "High"
            break

    # Medium severity
    else:

        for keyword in MEDIUM_SEVERITY_KEYWORDS:

            if keyword.lower() in text:
                severity = "Medium"
                break

    # ------------------------------------------
    # Remove duplicate categories
    # ------------------------------------------

    detected_categories = list(dict.fromkeys(detected_categories))

    # ------------------------------------------
    # Return result
    # ------------------------------------------

    return {
        "is_threat": True,
        "category": ", ".join(detected_categories),
        "severity": severity
    }