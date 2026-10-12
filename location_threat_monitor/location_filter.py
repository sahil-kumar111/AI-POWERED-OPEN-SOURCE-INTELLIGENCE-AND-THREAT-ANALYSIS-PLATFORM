# ==========================================
# LOCATION RELEVANCE FILTER
# ==========================================

def check_location_relevance(article, location):
    """
    Determine whether an article is relevant to
    the requested location.

    This is a basic rule-based filter.
    """

    title = article.get("title") or ""
    description = article.get("description") or ""

    text = (title + " " + description).lower()
    location = location.lower().strip()

    # Location does not appear at all
    if location not in text:
        return False

    # Phrases where the location is usually just
    # a destination/reference rather than incident location
    destination_phrases = [
        f"to {location}",
        f"towards {location}",
        f"heading to {location}",
        f"bound for {location}",
        f"en route to {location}",
        f"travelling to {location}",
        f"traveling to {location}",
    ]

    # If location appears only in a destination context,
    # reject it for now.
    for phrase in destination_phrases:

        if phrase in text:

            # Look for evidence that an incident actually
            # occurred at/in the requested location.
            incident_phrases = [
                f"fire in {location}",
                f"fire at {location}",
                f"accident in {location}",
                f"accident at {location}",
                f"attack in {location}",
                f"attack at {location}",
                f"explosion in {location}",
                f"explosion at {location}",
                f"flood in {location}",
                f"flooding in {location}",
                f"earthquake in {location}",
                f"riot in {location}",
                f"violence in {location}",
                f"outbreak in {location}",
                f"cyclone in {location}",
                f"storm in {location}",
            ]

            for incident_phrase in incident_phrases:
                if incident_phrase in text:
                    return True

            return False

    # If location appears without an obvious
    # destination-only context, keep it for
    # further analysis.
    return True