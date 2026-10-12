# ==========================================
# GEOGRAPHIC RELEVANCE
# ==========================================

from distance_calculator import calculate_distance


def check_geographic_relevance(
    user_lat,
    user_lon,
    incident_lat,
    incident_lon,
    local_radius=50,
    nearby_radius=100
):
    """
    Determine whether an incident is geographically
    relevant to the user's location.

    Returns:
        Local
        Nearby
        Outside
    """

    distance = calculate_distance(
        user_lat,
        user_lon,
        incident_lat,
        incident_lon
    )

    if distance <= local_radius:
        classification = "Local"

    elif distance <= nearby_radius:
        classification = "Nearby"

    else:
        classification = "Outside"

    return {
        "distance_km": distance,
        "classification": classification
    }