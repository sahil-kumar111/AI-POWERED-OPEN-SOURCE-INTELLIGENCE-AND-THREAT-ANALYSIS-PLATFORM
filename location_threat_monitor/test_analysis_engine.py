from analysis_engine import analyze_threats


# ------------------------------------------
# User location
# ------------------------------------------

location = "Chennai"

user_latitude = 12.8423
user_longitude = 80.1580


print("\n===================================")
print("🌍 LOCATION-BASED THREAT MONITOR")
print("===================================")

print("\n📍 Monitoring location:", location)

print("Latitude:", user_latitude)
print("Longitude:", user_longitude)

print("\n📰 Analyzing recent news...")
print("Please wait...\n")


results = analyze_threats(
    location,
    user_latitude,
    user_longitude
)


print("===================================")
print("🚨 ANALYSIS RESULTS")
print("===================================")


print("\nTotal threats found:", len(results))


for number, result in enumerate(
    results,
    start=1
):

    print("\n-----------------------------------")
    print(f"THREAT {number}")
    print("-----------------------------------")

    print("Title:", result["title"])

    print("Source:", result["source"])

    print("Published:", result["publishedAt"])

    print("Category:", result["category"])

    print("Severity:", result["severity"])

    print(
        "Incident location:",
        result["incident_location"]
    )

    print(
        "Distance:",
        f"{result['distance_km']:.2f} km"
    )

    print(
        "Geographic status:",
        result["geographic_classification"]
    )

    print("URL:", result["url"])