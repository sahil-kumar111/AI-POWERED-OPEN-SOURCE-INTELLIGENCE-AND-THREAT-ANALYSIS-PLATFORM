from geographic_relevance import check_geographic_relevance


# User's location
user_lat = 12.8423
user_lon = 80.1580


# Example incident location: Kodangal
incident_lat = 17.1120
incident_lon = 77.6170


result = check_geographic_relevance(
    user_lat,
    user_lon,
    incident_lat,
    incident_lon
)


print("\n===================================")
print("🌍 GEOGRAPHIC RELEVANCE TEST")
print("===================================")

print(f"Distance: {result['distance_km']:.2f} km")
print("Classification:", result["classification"])