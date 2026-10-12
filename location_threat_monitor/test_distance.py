from distance_calculator import calculate_distance


# Chennai coordinates
chennai_lat = 12.8423
chennai_lon = 80.1580


# Approximate Kodangal coordinates
kodangal_lat = 17.1120
kodangal_lon = 77.6170


distance = calculate_distance(
    chennai_lat,
    chennai_lon,
    kodangal_lat,
    kodangal_lon
)


print("\n===================================")
print("📍 DISTANCE CALCULATION TEST")
print("===================================")

print(f"Chennai → Kodangal: {distance:.2f} km")