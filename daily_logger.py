import pandas as pd
import feedparser
import urllib.parse
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Sector Mappings & Classifier Pipeline
REGIONS = ["Global", "Americas (including U.S.)", "Europe (EMEA)", "Asia (APAC)"]
INDUSTRY_MAP = {
    "Industrials / Manufacturing": ["industrial", "factory", "manufacturing", "BMW", "Siemens"],
    "Technology": ["technology", "software", "semiconductor", "microchip", "Nvidia", "Apple"],
    "Financials": ["banking", "credit", "financial", "JPMorgan", "interest rate"],
    "Consumer & Retail": ["retail", "consumer", "Walmart", "spending", "e-commerce"],
    "Energy & Raw Materials": ["energy", "oil", "gas", "BASF", "commodity"]
}

# Training corpus for daily classifier
training_corpus = [
    ("Quarterly profit loss recorded due to debt liquidity crunch", "Financial"),
    ("Credit downgrade risk increases as revenue misses estimates", "Financial"),
    ("Factory shutdown imminent as supplier microchip delivery halts", "Operational"),
    ("Labor strike stops manufacturing plant production line", "Operational"),
    ("Rival release causes sudden loss of market share dominance", "Strategic"),
    ("EU emission fine increases cross-border export tariff burden", "Regulatory"),
    ("Company beats earnings estimates with record quarterly output", "General (Neutral / Positive)")
]

train_df = pd.DataFrame(training_corpus, columns=["Headline", "Label"])
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(train_df["Headline"])
model = LogisticRegression(random_state=42)
model.fit(X_train, train_df["Label"])

def run_daily_archiver():
    # 2. Load Existing Historical CSV
    try:
        hist_df = pd.read_csv("historical_news.csv")
    except FileNotFoundError:
        hist_df = pd.DataFrame(columns=["Date", "Region", "Industry", "Risk_Vector", "Keyword", "Headline", "Link"])

    new_records = []
   
    # 3. Fetch current RSS headlines across all Regions & Industries
    for ind, keywords in INDUSTRY_MAP.items():
        query = f"{keywords[0]} OR {keywords[1]} business"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
       
        feed = feedparser.parse(rss_url)
       
        for entry in feed.entries[:10]: # Fetch top 10 articles per industry
            headline = entry.title
            link = entry.link
            pub_date = datetime.now().strftime("%Y-%m-%d")
           
            # Run AI classification
            X_test = vectorizer.transform([headline])
            pred_vector = model.predict(X_test)[0]
           
            for reg in REGIONS:
                new_records.append({
                    "Date": pub_date,
                    "Region": reg,
                    "Industry": ind,
                    "Risk_Vector": pred_vector,
                    "Keyword": keywords[0].capitalize(),
                    "Headline": headline,
                    "Link": link
                })

    # 4. Merge, Deduplicate on Headline Text, and Save
    new_df = pd.DataFrame(new_records)
    combined_df = pd.concat([hist_df, new_df], ignore_index=True)
   
    # Remove duplicate headlines while preserving historical order
    combined_df.drop_duplicates(subset=["Headline", "Industry", "Region"], keep="first", inplace=True)
   
    combined_df.to_csv("historical_news.csv", index=False)
    print(f" Daily archive completed. Full dataset now has {len(combined_df)} records.")

if __name__ == "__main__":
    run_daily_archiver()

