from incident_geocoder import geocode_location


location = "Kodangal, Telangana, India"


print("\n===================================")
print("📍 INCIDENT GEOCODER TEST")
print("===================================")

print("\nSearching for:", location)


result = geocode_location(location)


if result:

    print("\n✅ Location found!")

    print("Location:", result["location"])
    print("Latitude:", result["latitude"])
    print("Longitude:", result["longitude"])
    print("Full location:", result["display_name"])

else:

    print("\n❌ Location could not be found.")