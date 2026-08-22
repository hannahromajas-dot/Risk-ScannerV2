import pandas as pd
import feedparser
import urllib.parse
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Define mappings (matching app.py)
REGIONS_LIST = [
    "Global",
    "Americas (including U.S.)",
    "Europe (EMEA)",
    "Asia (APAC)",
]

INDUSTRY_MAP = {
    "Industrials / Manufacturing": [
        "industrial", "factory", "manufacturing", "BMW", "Siemens",
        "automotive", "assembly", "supply chain", "plant",
    ],
    "Technology": [
        "technology", "software", "semiconductor", "microchip", "Nvidia",
        "Apple", "AI", "cloud", "cybersecurity", "chip", "data center",
    ],
    "Financials": [
        "banking", "credit", "financial", "JPMorgan", "interest rate",
        "liquidity", "wall street", "bank", "loan", "capital",
    ],
    "Consumer & Retail": [
        "retail", "consumer", "Walmart", "spending", "e-commerce",
        "Nike", "store", "shopping", "consumer demand",
    ],
    "Energy & Raw Materials": [
        "energy", "oil", "gas", "BASF", "commodity", "carbon",
        "power grid", "renewable", "mining", "raw materials",
    ],
}

REGION_TERM_MAP = {
    "Global": "global OR worldwide OR international",
    "Americas (including U.S.)": "US OR USA OR America OR Americas OR \"United States\" OR Canada OR Mexico",
    "Europe (EMEA)": "Europe OR EU OR EMEA OR Germany OR France OR UK OR Britain",
    "Asia (APAC)": "Asia OR APAC OR China OR Japan OR India OR \"South Korea\" OR Singapore",
}

REGION_GEO_TERMS = {
    "Global": "",
    "Americas (including U.S.)": "US OR America OR Canada",
    "Europe (EMEA)": "Europe OR Germany OR France OR UK",
    "Asia (APAC)": "Asia OR China OR Japan OR India OR Taiwan",
}

HISTORICAL_CSV = Path("historical_news.csv")

def clean_headline(raw_title: str) -> str:
    """Remove source tags and trailing site names from Google News titles."""
    if not raw_title:
        return ""
    cleaned = re.sub(r"^\[.*?\]\s*", "", raw_title).strip()
    cleaned = cleaned.lstrip("[").strip()

    for separator in [" - ", " | ", " — "]:
        if separator in cleaned:
            parts = cleaned.rsplit(separator, 1)
            if len(parts[0]) > 20:
                cleaned = parts[0]
    return cleaned.strip()

def train_erm_classifier():
    """Train the TF-IDF + Logistic Regression risk classifier."""
    training_corpus = [
        ("Quarterly profit loss recorded due to debt liquidity crunch", "Financial"),
        ("Credit downgrade risk increases as revenue misses estimates", "Financial"),
        ("Factory shutdown imminent as supplier microchip delivery halts", "Operational"),
        ("Transit port congestion causes shipping bottleneck delays", "Operational"),
        ("Rival release causes sudden loss of market share dominance", "Strategic"),
        ("Failed merger strategy leaves corporate growth outlook uncertain", "Strategic"),
        ("EU emission fine increases cross-border export tariff burden", "Regulatory"),
        ("Antitrust inquiry launched over non-compliance trade practices", "Regulatory"),
        ("Company beats earnings estimates with record quarterly output", "General (Neutral / Positive)"),
    ]
    train_df = pd.DataFrame(training_corpus, columns=["Headline", "Label"])
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X_train = vectorizer.fit_transform(train_df["Headline"])
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    model.fit(X_train, train_df["Label"])
    return vectorizer, model

def run_daily_archiver():
    print("Starting daily RSS news archival...")
    vectorizer, erm_model = train_erm_classifier()
    
    # 1. Load existing historical CSV if it exists
    if HISTORICAL_CSV.exists():
        try:
            hist_df = pd.read_csv(HISTORICAL_CSV)
        except Exception:
            hist_df = pd.DataFrame(columns=["Date", "Region", "Industry", "Risk_Vector", "Keyword", "Headline", "Link"])
    else:
        hist_df = pd.DataFrame(columns=["Date", "Region", "Industry", "Risk_Vector", "Keyword", "Headline", "Link"])

    new_records = []

    # 2. Loop through all regions and industries to fetch fresh RSS articles
    for region in REGIONS_LIST:
        for industry, keywords in INDUSTRY_MAP.items():
            keyword_part = " OR ".join(f'"{k}"' if " " in k else k for k in keywords[:4])
            geo_term = REGION_GEO_TERMS.get(region, "")
            geo_part = f"({geo_term})" if geo_term else ""
            query = f"({keyword_part}) {geo_part}".strip()
            
            encoded_query = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            try:
                feed = feedparser.parse(rss_url)
                entries = getattr(feed, "entries", [])[:15] # Grab top 15 per combo

                for entry in entries:
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        parsed_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
                    else:
                        parsed_date = datetime.now().strftime("%Y-%m-%d")

                    raw_headline = getattr(entry, "title", "")
                    link = getattr(entry, "link", "")
                    cleaned = clean_headline(raw_headline)

                    if len(cleaned) < 18:
                        continue

                    # Predict risk vector using model
                    X_test = vectorizer.transform([cleaned])
                    pred_vector = erm_model.predict(X_test)[0]

                    new_records.append({
                        "Date": parsed_date,
                        "Region": region,
                        "Industry": industry,
                        "Risk_Vector": pred_vector,
                        "Keyword": keywords[0].capitalize(),
                        "Headline": cleaned,
                        "Link": link if link.startswith("http") else "",
                    })
            except Exception as e:
                print(f"Skipping {region} - {industry} due to error: {e}")

    new_df = pd.DataFrame(new_records)

    if not new_df.empty:
        # 3. Concatenate old historical data with new records
        combined = pd.concat([hist_df, new_df], ignore_index=True)
        
        # 4. Drop duplicates based on Headline and Date to ensure integrity
        combined = combined.drop_duplicates(subset=["Headline", "Date"], keep="last")
        
        # 5. Save back out to historical_news.csv
        combined.to_csv(HISTORICAL_CSV, index=False)
        print(f"Successfully archived daily news! Total records now: {len(combined)}")
    else:
        print("No new records fetched today.")

if __name__ == "__main__":
    run_daily_archiver()
