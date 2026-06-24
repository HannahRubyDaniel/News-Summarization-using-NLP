import requests
import wikipedia
from datetime import datetime, timedelta
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------
# API KEYS
# --------------------------------------------------------
GNEWS_API_KEY = "YOUR ID ITS FREE"
NEWSAPI_KEY = "YOUR ID ITS FREE"
NEWSDATA_KEY = "YOUR ID ITS FREE"


# --------------------------------------------------------
# Extract FULL article text safely
# --------------------------------------------------------


def all_words_in_text(user_input, article_text):
    """
    Returns True if every word in the user_input is present in article_text.
    Both are case-insensitive, and matches only whole words.
    """
    if not user_input or not article_text:
        return False

    # Lowercase everything
    user_input = user_input.lower()
    article_text = article_text.lower()

    # Split input into words
    input_words = re.findall(r'\b\w+\b', user_input)

    # Check each word
    for word in input_words:
        # Use regex to match whole word
        if not re.search(r'\b' + re.escape(word) + r'\b', article_text):
            return False

    return True

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'\[\+\d+ chars\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Ensure sentence ends properly
    if text and not text.endswith((".", "!", "?")):
        text += "."

    return text
def extract_full_text(article_url):
    try:
        if not article_url or not article_url.startswith("http"):
            return ""

        art = Article(article_url)
        art.download()
        art.parse()
        return art.text.strip()

    except Exception:
        return ""


# --------------------------------------------------------
# Parse date safely
# --------------------------------------------------------
def parse_date(date_str):
    try:
        if not date_str:
            return datetime.min

        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        return dt

    except Exception:
        return datetime.min


# --------------------------------------------------------
# Validate Content Quality
# --------------------------------------------------------
# --------------------------------------------------------
# Validate Content Quality  — IMPROVED
# --------------------------------------------------------
def is_valid_content(text):
    if not text:
        return False

    text = text.strip()

    # Too short
    if len(text) < 150:
        return False

    # Hard blocked phrases
    blocked_phrases = [
        "AVAILABLE IN PAID PLANS", "SUBSCRIBE", "SIGN IN",
        "REGISTER TO CONTINUE", "ACCESS DENIED"
    ]
    for phrase in blocked_phrases:
        if phrase.lower() in text.lower():
            return False

    # All uppercase = junk
    if text.isupper():
        return False

    # ---- NEW: Detect country-list / enumeration junk ----
    # If text has too many commas relative to sentences it's a list not an article
    comma_count = text.count(",")
    sentence_count = len(re.findall(r'[.!?]', text))
    if sentence_count > 0 and comma_count / sentence_count > 8:
        return False

    # If text contains many known "Republic of / Kingdom of" patterns it's a country list
    republic_matches = len(re.findall(r'\b(Republic|Kingdom|Islands|Democratic)\b', text, re.IGNORECASE))
    words = len(text.split())
    if words > 0 and republic_matches / words > 0.03:  # >3% of words are geo-political titles
        return False

    # ---- NEW: Detect podcast/radio transcript junk ----
    podcast_phrases = [
        "write to them at", "visit our youtube", "click here to listen",
        "subscribe to our podcast", "listen on spotify", "apple podcasts",
        "bonus episode", "youtube channel", "0800", "contact us on"
    ]
    lower_text = text.lower()
    podcast_hits = sum(1 for phrase in podcast_phrases if phrase in lower_text)
    if podcast_hits >= 2:  # 2+ podcast phrases = it's a transcript/promo, not news
        return False

    # ---- NEW: Minimum unique sentence ratio ----
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) < 3:
        return False

    return True

# --------------------------------------------------------
# Process Single Article (Parallel Worker)
# --------------------------------------------------------
def process_article(item, source_name, url_key, date_key, keyword):
    """
    Process a single article:
    - Extracts full text from URL if available
    - Cleans and validates content
    - Prefers articles containing all keyword words,
      but won't reject everything if none match fully
    """

    article_url = item.get(url_key)
    full_text = extract_full_text(article_url)
    raw_content = full_text if full_text else item.get("content")
    content = clean_text(raw_content)

    if not is_valid_content(content):
        return None

    # 1️⃣ Check if article contains all keyword words
    if all_words_in_text(keyword, content):
        return {
            "title": item.get("title"),
            "content": content,
            "publishedAt": item.get(date_key),
            "source": source_name
        }

    # 2️⃣ Fallback: accept partially matching article (at least 50% words)
    input_words = re.findall(r'\b\w+\b', keyword.lower())
    article_words = set(re.findall(r'\b\w+\b', content.lower()))
    match_count = sum(1 for w in input_words if w in article_words)
    if match_count / len(input_words) >= 0.5:
        return {
            "title": item.get("title"),
            "content": content,
            "publishedAt": item.get(date_key),
            "source": source_name
        }

    # 3️⃣ Otherwise reject
    return None

# --------------------------------------------------------
# TF-IDF + Cosine Similarity Deduplication
# --------------------------------------------------------
def remove_similar_articles(articles, similarity_threshold=0.75):

    if len(articles) <= 1:
        return articles

    texts = [a["content"] for a in articles]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    selected = []
    used_indices = set()

    for i in range(len(articles)):
        if i in used_indices:
            continue

        selected.append(articles[i])

        for j in range(i + 1, len(articles)):
            if similarity_matrix[i][j] > similarity_threshold:
                used_indices.add(j)

    return selected


# --------------------------------------------------------
# FETCHERS
# --------------------------------------------------------
def fetch_gnews(keyword, max_articles=10):
    api_url = (
        "https://gnews.io/api/v4/search?"
        f"q={keyword}&"
        "lang=en&"
        f"max={max_articles}&"
        f"from={(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}&"
        f"token={GNEWS_API_KEY}"
    )

    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
    except Exception:
        return []

    articles = data.get("articles", [])
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_article, item, "GNews", "url", "publishedAt", keyword)
            for item in articles
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results


def fetch_newsapi(keyword, max_articles=6):
    api_url = (
        "https://newsapi.org/v2/everything?"
        f"q={keyword}&"
        "language=en&"
        f"from={(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}&"
        f"apiKey={NEWSAPI_KEY}"
    )

    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
    except Exception:
        return []

    articles = data.get("articles", [])[:max_articles]
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_article, item, "NewsAPI", "url", "publishedAt", keyword)
            for item in articles
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results


def fetch_newsdata(keyword, max_articles=6):
    api_url = (
        "https://newsdata.io/api/1/news?"
        f"apikey={NEWSDATA_KEY}&"
        f"q={keyword}&"
        "language=en"
    )

    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
    except Exception:
        return []

    articles = data.get("results", [])[:max_articles]
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_article, item, "NewsData.io", "link", "pubDate", keyword)
            for item in articles
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results


# --------------------------------------------------------
# Wikipedia fallback
# --------------------------------------------------------
def fetch_wikipedia(keyword):
    try:
        summary = wikipedia.summary(keyword, sentences=5)
        return [{
            "title": f"Wikipedia: {keyword}",
            "content": summary,
            "publishedAt": None,
            "source": "Wikipedia"
        }]
    except Exception:
        return []


# --------------------------------------------------------
# MAIN FETCH FUNCTION
# --------------------------------------------------------

def fetch_all(keyword, max_articles=6):

    combined = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(fetch_gnews, keyword, max_articles),
            executor.submit(fetch_newsapi, keyword, max_articles),
            executor.submit(fetch_newsdata, keyword, max_articles),
        ]

        for future in as_completed(futures):
            combined.extend(future.result())

    # Remove title duplicates
    seen_titles = set()
    unique = []

    for art in combined:
        title = art.get("title")
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique.append(art)

    if not unique:
        print("No news found. Using Wikipedia fallback.")
        return fetch_wikipedia(keyword)

    # Sort chronologically (OLDEST → NEWEST)
    unique.sort(key=lambda x: parse_date(x.get("publishedAt")))

    # Remove near duplicates using cosine similarity
    unique = remove_similar_articles(unique)

    return unique[:max_articles]


import streamlit as st
from concurrent.futures import ThreadPoolExecutor


# --------------------------------------------------------
# Example Usage
# --------------------------------------------------------
if __name__ == "__main__":
    keyword = "Kerala flood"
    articles = fetch_all(keyword, max_articles=6)

    print(f"\nFetched {len(articles)} timeline articles:\n")

    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['title']} ({article['source']})")
        print("Published:", article["publishedAt"])
        print("Preview:", (article["content"] or "")[:250])
        print("-" * 80)
