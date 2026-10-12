# ==========================================
# DUPLICATE ARTICLE FILTER
# ==========================================


def remove_duplicate_articles(articles):
    """
    Remove duplicate news articles based on URL.
    """

    unique_articles = []
    seen_urls = set()

    for article in articles:

        url = article.get("url")

        # If URL has already been seen,
        # skip the article.
        if url and url in seen_urls:
            continue

        if url:
            seen_urls.add(url)

        unique_articles.append(article)

    return unique_articles