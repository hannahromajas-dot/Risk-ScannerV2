import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# SECTION 1: PAGE CONFIGURATION & LAYOUT
# Functionality: Configures Streamlit page structure and header banners.
# How it works: Expands viewport layout to wide mode for multi-column dashboards.
# Data Sources: None (UI configuration).
# Tools Used: Streamlit (st.set_page_config, st.title, st.markdown)
# ==============================================================================
st.set_page_config(page_title="Enterprise Risk Scanner", page_icon="🏛️", layout="wide")

st.title("🏛️ Enterprise Market Risk Management Scanner")
st.markdown("**Real-Time & 12-Month Longitudinal Media Threat Intelligence**")
st.divider()


# ==============================================================================
# SECTION 2: INDUSTRY & REGIONAL MAPPINGS & AI CLASSIFIER TRAINING
# Functionality: Trains an in-memory Scikit-Learn text classification engine.
# How it works: Vectorizes headlines using TF-IDF and classifies text into
#               Financial, Operational, Strategic, Regulatory, or General.
# Data Sources: Pre-labeled corporate ERM training corpus.
# Tools Used: Scikit-Learn (TfidfVectorizer, LogisticRegression), Pandas
# ==============================================================================
REGIONS_LIST = ["Global", "Americas (including U.S.)", "Europe (EMEA)", "Asia (APAC)"]

INDUSTRY_MAP = {
    "Industrials / Manufacturing": ["industrial", "factory", "manufacturing", "BMW", "Siemens", "automotive", "assembly"],
    "Technology": ["technology", "software", "semiconductor", "microchip", "Nvidia", "Apple", "AI", "cloud"],
    "Financials": ["banking", "credit", "financial", "JPMorgan", "interest rate", "liquidity", "wall street"],
    "Consumer & Retail": ["retail", "consumer", "Walmart", "spending", "e-commerce", "Nike", "store"],
    "Energy & Raw Materials": ["energy", "oil", "gas", "BASF", "commodity", "carbon", "power grid"]
}

@st.cache_resource
def train_erm_classifier():
    training_corpus = [
        # Financial
        ("Quarterly profit loss recorded due to debt liquidity crunch", "Financial"),
        ("Credit downgrade risk increases as revenue misses estimates", "Financial"),
        ("Rising interest rate expense squeezes corporate margin balance", "Financial"),
       
        # Operational
        ("Factory shutdown imminent as supplier microchip delivery halts", "Operational"),
        ("Transit port congestion causes shipping bottleneck delays", "Operational"),
        ("Labor strike stops manufacturing plant production line", "Operational"),
       
        # Strategic
        ("Rival release causes sudden loss of market share dominance", "Strategic"),
        ("Delayed EV transition compromises multi-year market position", "Strategic"),
        ("Failed merger strategy leaves corporate growth outlook uncertain", "Strategic"),
       
        # Regulatory
        ("EU emission fine increases cross-border export tariff burden", "Regulatory"),
        ("Antitrust inquiry launched over non-compliance trade practices", "Regulatory"),
        ("Bilateral export sanctions restrict international market access", "Regulatory"),
       
        # General
        ("Company beats earnings estimates with record quarterly output", "General (Neutral / Positive)"),
        ("New automated facility opens boosting operational efficiency", "General (Neutral / Positive)"),
        ("Strategic partnership established to accelerate clean technology", "General (Neutral / Positive)")
    ]
   
    train_df = pd.DataFrame(training_corpus, columns=["Headline", "Label"])
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(train_df["Headline"])
    model = LogisticRegression(random_state=42)
    model.fit(X_train, train_df["Label"])
    return vectorizer, model

vectorizer, erm_model = train_erm_classifier()


# ==============================================================================
# SECTION 3: DATA INGESTION PIPELINE (HISTORICAL CSV + LIVE RSS FEED)
# Functionality: Loads historical records and merges them with live RSS feeds.
# How it works: 1. Ingests historical_news.csv.
#               2. Fetches live RSS Google News matching sector parameters.
#               3. Runs AI classifier on RSS rows to standardize schema.
# Data Sources: historical_news.csv & Google News RSS endpoint.
# Tools Used: Pandas, feedparser, urllib.parse, Scikit-Learn
# ==============================================================================
@st.cache_data(ttl=600)
def load_combined_dataset(selected_region, selected_industry):
    # 1. Load Historical Baseline CSV
    try:
        hist_df = pd.read_csv("historical_news.csv")
    except FileNotFoundError:
        hist_df = pd.DataFrame(columns=["Date", "Region", "Industry", "Risk_Vector", "Keyword", "Headline", "Link"])

    # 2. Fetch Live Google News RSS Feed
    keywords = INDUSTRY_MAP[selected_industry]
    query = f"{keywords[0]} OR {keywords[1]} business"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
   
    feed = feedparser.parse(rss_url)
    live_records = []
   
    for entry in feed.entries[:30]:  # Rate-sample live feed
        pub_date = getattr(entry, "published", datetime.now().strftime("%Y-%m-%d"))
        try:
            parsed_date = datetime.strptime(pub_date[:16], "%a, %d %b %Y").strftime("%Y-%m-%d")
        except:
            parsed_date = datetime.now().strftime("%Y-%m-%d")
           
        headline = entry.title
        link = entry.link
       
        # AI ML Prediction for Live RSS headline
        X_test = vectorizer.transform([headline])
        pred_vector = erm_model.predict(X_test)[0]
       
        live_records.append({
            "Date": parsed_date,
            "Region": selected_region, # Applicable to user selected region
            "Industry": selected_industry,
            "Risk_Vector": pred_vector,
            "Keyword": keywords[0].capitalize(),
            "Headline": headline,
            "Link": link
        })
       
    live_df = pd.DataFrame(live_records)
   
    # Merge datasets into unified full corpus
    full_df = pd.concat([hist_df, live_df], ignore_index=True)
    full_df["Date"] = pd.to_datetime(full_df["Date"])
    return full_df

# ==============================================================================
# SECTION 4: LEFT-HAND SIDE DROPDOWN CONTROLS
# Functionality: Captures Region and Industry user selections.
# How it works: Sidebar widgets pass selected state variables to data pipeline.
# Data Sources: User click state.
# Tools Used: Streamlit (st.sidebar.selectbox)
# ==============================================================================
st.sidebar.header("⚙️ Filter Controls")

selected_region = st.sidebar.selectbox("1) Select Region:", options=REGIONS_LIST)
selected_industry = st.sidebar.selectbox("2) Select Primary Industry:", options=list(INDUSTRY_MAP.keys()))

# Load unified dataset
full_dataset = load_combined_dataset(selected_region, selected_industry)

# Filter dataset for selected Region & Industry
sector_df = full_dataset[
    (full_dataset["Industry"] == selected_industry) &
    (full_dataset["Region"].str.contains(selected_region.split()[0], case=False, na=False))
].copy()


# ==============================================================================
# SECTION 5: RIGHT-HAND SIDE - RISK TREND ANALYSIS (PAST 12 MONTHS)
# Functionality: Dual Y-Axis Line Chart & Formulaic 3-Month vs 12-Month Moving Trends.
# How it works: Calculates % Share of Neutral/Positive vs Risk news per month.
#               Computes moving average thresholds (+/- 10%) for formulaic arrows.
# Data Sources: Filtered sector_df.
# Tools Used: Plotly Graph Objects (go.Figure), Pandas GroupBy
# ==============================================================================
# Draw Dual Y-Axis Plotly Chart (Updated for Plotly compatibility)
with col_graph:
    fig = go.Figure()

    # Primary Y-Axis: General News (% Share) - BLUE
    fig.add_trace(go.Scatter(
        x=monthly_counts.index, y=monthly_counts["General_Pct"],
        name="Neutral / Positive News (%)", line=dict(color="blue", width=3),
        yaxis="y"
    ))

    # Secondary Y-Axis: Risk Vectors (% Share) - RED
    fig.add_trace(go.Scatter(
        x=monthly_counts.index, y=monthly_counts["Risk_Pct"],
        name="Risk Vector News (%)", line=dict(color="red", width=3),
        yaxis="y2"
    ))

    # Explicit layout configuration for dual axes without nested dictionary updating errors
    fig.update_axes(title_text="Last 12 Months", xaxis=True)
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="General News Share (%)", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(title="Risk Vector Share (%)", titlefont=dict(color="red"), tickfont=dict(color="red"), overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# SECTION 6: RIGHT-HAND SIDE - RECENT THREATS (LAST 7 DAYS)
# Functionality: Vertical Bar Chart & Top 3 Threat Headlines with Search Links.
# How it works: Filters sector_df for the past 7 days, groups counts by Risk Vector,
#               and displays Top 3 headlines linking to source or Google Search query.
# Data Sources: Filtered sector_df.
# Tools Used: Streamlit, Plotly Bar, Pandas Markdown HTML formatting
# ==============================================================================
st.subheader("⚠️ Recent Threats - Last 7 Days")

seven_days_ago = pd.to_datetime(datetime.now() - timedelta(days=7))
recent_df = sector_df[sector_df["Date"] >= seven_days_ago].copy()

col_bar, col_table = st.columns([1, 1.5])

with col_bar:
    st.markdown("#### Threat Count by Vector")
    risk_only_recent = recent_df[recent_df["Risk_Vector"] != "General (Neutral / Positive)"]
   
    if not risk_only_recent.empty:
        bar_counts = risk_only_recent["Risk_Vector"].value_counts().reset_index()
        bar_counts.columns = ["Risk Vector", "Count"]
       
        fig_bar = go.Figure(go.Bar(
            x=bar_counts["Risk Vector"], y=bar_counts["Count"],
            marker_color="crimson"
        ))
        fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No elevated risk threats detected in the last 7 days.")

with col_table:
    st.markdown("#### Top 3 Headlines per Risk Vector")
   
    top_headlines = []
    for rv in ["Financial", "Operational", "Strategic", "Regulatory"]:
        sub_rv = recent_df[recent_df["Risk_Vector"] == rv].head(3)
        for _, row in sub_rv.iterrows():
            # Format link to source or fallback to Google Search query if link is empty
            if row["Link"] and str(row["Link"]).startswith("http"):
                target_url = row["Link"]
            else:
                target_url = f"https://www.google.com/search?q={urllib.parse.quote(row['Headline'])}"
               
            formatted_link = f"[{row['Headline']}]({target_url})"
            top_headlines.append({
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Vector": rv,
                "Keyword": row["Keyword"],
                "Headline": formatted_link
            })
           
    if top_headlines:
        top_df = pd.DataFrame(top_headlines)
        st.write(top_df.to_markdown(index=False), unsafe_allow_html=True)
    else:
        st.caption("No recent threat headlines available for display.")

st.divider()


# ==============================================================================
# SECTION 7: IN-MEMORY PANDAS NATURAL-LANGUAGE CHATBOT ENGINE
# Functionality: Answers user questions over FULL corpus agnostic to UI filters.
# How it works: Queries the entire full_dataset using keyword matching and Pandas SQL logic.
# Data Sources: Full merged corpus (historical_news.csv + Live RSS).
# Tools Used: Streamlit Chat (st.chat_message, st.chat_input, st.button), Pandas
# ==============================================================================
st.subheader("💬 Executive Risk Assistant Chatbot")
st.caption("Queries the **full global dataset** across all regions and industries in real time.")

# 3 Standard Contextual Click Prompt Buttons
btn_c1, btn_c2, btn_c3 = st.columns(3)
selected_prompt = None

with btn_c1:
    if st.button("🚨 Top Daily High-Risk Threat"):
        selected_prompt = "What is the single biggest high-risk headline detected in the news feed today, and why was it flagged?"

with btn_c2:
    if st.button("📊 Risk vs Growth Ratio"):
        selected_prompt = f"How does the volume of Risk Alerts compare to positive Stable Growth Signals for {selected_industry} in {selected_region}?"

with btn_c3:
    if st.button("⚙️ 30-Day Operational Threats"):
        selected_prompt = f"Show me all headlines in the past month flagged under Operational Risk affecting {selected_industry} in {selected_region}."

user_query = st.chat_input("Ask a question about the full dataset...") or selected_prompt

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
       
    with st.chat_message("assistant"):
        q_lower = user_query.lower()
       
        # 1. Answer Prompt 1: Single biggest high-risk headline today
        if "single biggest high-risk headline" in q_lower or "today" in q_lower:
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_risk = full_dataset[
                (full_dataset["Date"] >= today_str) &
                (full_dataset["Risk_Vector"] != "General (Neutral / Positive)")
            ]
            if not today_risk.empty:
                top_row = today_risk.iloc[0]
                st.error(f"**[{top_row['Risk_Vector']} Risk - {top_row['Industry']}]** {top_row['Headline']}")
                st.caption(f"Flagged by AI Engine under {top_row['Risk_Vector']} risk due to keyword pattern indicators.")
            else:
                st.success("No critical high-risk headlines flagged in today's feed snapshot.")

        # 2. Answer Prompt 2: Risk vs Growth Ratio
        elif "compare to positive stable growth" in q_lower or "volume of risk alerts" in q_lower:
            risk_cnt = len(sector_df[sector_df["Risk_Vector"] != "General (Neutral / Positive)"])
            growth_cnt = len(sector_df[sector_df["Risk_Vector"] == "General (Neutral / Positive)"])
            st.write(f"**Sentiment Volume Ratio for {selected_industry} ({selected_region}):**")
            st.write(f"• **Active Risk Vector Alerts 🔴:** {risk_cnt} headlines")
            st.write(f"• **Neutral / Growth Signals 🟢:** {growth_cnt} headlines")

        # 3. Answer Prompt 3: Past Month Operational Risks for selected sector
        elif "operational risk" in q_lower and "past month" in q_lower:
            one_month_ago = pd.to_datetime(datetime.now() - timedelta(days=30))
            ops_df = sector_df[
                (sector_df["Risk_Vector"] == "Operational") &
                (sector_df["Date"] >= one_month_ago)
            ]
            if not ops_df.empty:
                st.write(f"**Operational Risk Headlines (Past 30 Days) for {selected_industry}:**")
                for _, r in ops_df.head(5).iterrows():
                    st.write(f"• **[{r['Date'].strftime('%Y-%m-%d')}]** {r['Headline']}")
            else:
                st.info(f"No Operational Risk alerts recorded in the past 30 days for {selected_industry}.")

        # 4. Fallback Natural-Language Pandas Search Engine
        else:
            search_words = [w for w in q_lower.split() if len(w) > 3 and w not in ["show", "tell", "what", "where", "about"]]
            if search_words:
                pattern = "|".join(search_words)
                results = full_dataset[full_dataset["Headline"].str.contains(pattern, case=False, na=False)].head(5)
                if not results.empty:
                    st.write(f"**Found {len(results)} matching records across full dataset:**")
                    for _, r in results.iterrows():
                        st.write(f"• **[{r['Risk_Vector']}]** ({r['Region']} | {r['Industry']}) {r['Headline']}")
                else:
                    st.warning("No direct matches found in the dataset for that query.")
            else:
                st.info("Please ask a specific risk, region, or industry query.")
