from news_collector import collect_news
from incident_location import extract_incident_location


location = "Chennai"

articles = collect_news(location)


print("\n===================================")
print("📍 INCIDENT LOCATION TEST")
print("===================================")

print("Requested location:", location)
print("Articles:", len(articles))


for number, article in enumerate(articles, start=1):

    locations = extract_incident_location(article)

    print("\n-----------------------------------")
    print(f"ARTICLE {number}")
    print("-----------------------------------")

    print("Title:", article["title"])

    if locations:
        print("Locations found:", ", ".join(locations))
    else:
        print("Locations found: None")