from news_collector import collect_news


location = "Chennai"


print("\n===================================")
print("🌍 LOCATION-BASED THREAT MONITOR")
print("===================================")

print("\n📍 Location:", location)
print("📰 Searching news from the last 48 hours...")
print("Please wait...\n")


articles = collect_news(location)


print("===================================")
print("📊 NEWS SEARCH RESULT")
print("===================================")

print("Total articles found:", len(articles))


for number, article in enumerate(articles, start=1):

    print("\n-----------------------------------")
    print(f"ARTICLE {number}")
    print("-----------------------------------")

    print("Title:", article["title"])
    print("Source:", article["source"])
    print("Published:", article["publishedAt"])
    print("URL:", article["url"])
    print("Description:", article["description"])