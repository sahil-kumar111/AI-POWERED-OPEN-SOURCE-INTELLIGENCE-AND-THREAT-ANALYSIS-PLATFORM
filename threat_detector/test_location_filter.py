from news_collector import collect_news
from location_filter import check_location_relevance


location = "Chennai"


print("\n===================================")
print("📍 LOCATION RELEVANCE TEST")
print("===================================")

print("\nSearching news for:", location)


articles = collect_news(location)


print("\nTotal articles collected:", len(articles))

print("\n===================================")
print("📊 FILTER RESULTS")
print("===================================")


relevant_count = 0


for number, article in enumerate(articles, start=1):

    relevant = check_location_relevance(
        article,
        location
    )

    if relevant:
        relevant_count += 1

    print("\n-----------------------------------")
    print(f"ARTICLE {number}")
    print("-----------------------------------")

    print("Title:", article["title"])
    print("Location Relevant:", relevant)


print("\n===================================")
print("SUMMARY")
print("===================================")

print("Total articles:", len(articles))
print("Location-relevant articles:", relevant_count)