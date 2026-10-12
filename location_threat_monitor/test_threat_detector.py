from news_collector import collect_news
from threat_detector import detect_threat


location = "Chennai"


print("\n===================================")
print("🌍 THREAT DETECTION TEST")
print("===================================")

print("\n📍 Location:", location)
print("📰 Collecting news...\n")


articles = collect_news(location)


print("Total articles collected:", len(articles))


print("\n===================================")
print("🚨 THREAT ANALYSIS")
print("===================================")


for number, article in enumerate(articles, start=1):

    result = detect_threat(article)

    print("\n-----------------------------------")
    print(f"ARTICLE {number}")
    print("-----------------------------------")

    print("Title:", article["title"])
    print("Source:", article["source"])

    print("Threat:", result["is_threat"])
    print("Category:", result["category"])
    print("Severity:", result["severity"])