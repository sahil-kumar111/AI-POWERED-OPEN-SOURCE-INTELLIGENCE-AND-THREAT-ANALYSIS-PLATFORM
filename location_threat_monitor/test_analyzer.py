from threat_analyzer import analyze_location


location = "Chennai"


results = analyze_location(location)


print("\n===================================")
print("🚨 FINAL THREAT RESULTS")
print("===================================")

print(f"\nTotal threats detected: {len(results)}")


for number, result in enumerate(results, start=1):

    print("\n-----------------------------------")
    print(f"THREAT {number}")
    print("-----------------------------------")

    print("Title:", result["title"])
    print("Source:", result["source"])
    print("Published:", result["publishedAt"])
    print("Category:", result["category"])
    print("Severity:", result["severity"])
    print("Location:", result["location"])
    print("URL:", result["url"])