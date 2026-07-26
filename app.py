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
   
    for entry in feed.entries[:30]:
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
            "Region": selected_region,
            "Industry": selected_industry,
            "Risk_Vector": pred_vector,
            "Keyword": keywords[0].capitalize(),
            "Headline": headline,
            "Link": link
        })
       
    live_df = pd.DataFrame(live_records)
   
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

full_dataset = load_combined_dataset(selected_region, selected_industry)

sector_df = full_dataset[
    (full_dataset["Industry"] == selected_industry) &
    (full_dataset["Region"].str.contains(selected_region.split()[0], case=False, na=False))
].copy()


# ==============================================================================
# SECTION 5: RIGHT-HAND SIDE - RISK TREND ANALYSIS (PAST 12 MONTHS)
# Functionality: Dual Y-Axis Line Chart (0-100% scale) & Formulaic Trends.
# How it works: Calculates % Share of Neutral/Positive vs Risk news per month.
#               Computes moving average thresholds (+/- 10%) for colored trend arrows.
# Data Sources: Filtered sector_df.
# Tools Used: Plotly Graph Objects (go.Figure), Pandas GroupBy
# ==============================================================================
st.subheader("📈 Risk Trend Analysis - Past 12 Months")
st.caption(f"Longitudinal Risk Exposure for **{selected_industry}** in **{selected_region}**")

col_graph, col_arrows = st.columns([2, 1])

if not sector_df.empty:
    sector_df["YearMonth"] = sector_df["Date"].dt.to_period("M").dt.to_timestamp()
    monthly_counts = sector_df.groupby(["YearMonth", "Risk_Vector"]).size().unstack(fill_value=0)
   
    for col in ["General (Neutral / Positive)", "Financial", "Operational", "Strategic", "Regulatory"]:
        if col not in monthly_counts.columns:
            monthly_counts[col] = 0
           
    monthly_counts["Total_Risk"] = monthly_counts[["Financial", "Operational", "Strategic", "Regulatory"]].sum(axis=1)
    monthly_counts["Total_News"] = monthly_counts["Total_Risk"] + monthly_counts["General (Neutral / Positive)"]
   
    monthly_counts["General_Pct"] = (monthly_counts["General (Neutral / Positive)"] / monthly_counts["Total_News"].replace(0, 1)) * 100
    monthly_counts["Risk_Pct"] = (monthly_counts["Total_Risk"] / monthly_counts["Total_News"].replace(0, 1)) * 100

    with col_graph:
        fig = go.Figure()

        # Primary Y-Axis trace (Blue)
        fig.add_trace(go.Scatter(
            x=monthly_counts.index, y=monthly_counts["General_Pct"],
            name="Neutral / Positive News (%)", line=dict(color="blue", width=3)
        ))

        # Secondary Y-Axis trace (Red)
        fig.add_trace(go.Scatter(
            x=monthly_counts.index, y=monthly_counts["Risk_Pct"],
            name="Risk Vector News (%)", line=dict(color="red", width=3),
            yaxis="y2"
        ))

        # Layout configuration with 0% to 100% explicit Y-Axis ranges
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(title="Last 12 Months"),
            yaxis=dict(
                title=dict(text="General News Share (%)", font=dict(color="blue")),
                tickfont=dict(color="blue"),
                range=[0, 100]
            ),
            yaxis2=dict(
                title=dict(text="Risk Vector Share (%)", font=dict(color="red")),
                tickfont=dict(color="red"),
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
        )
       
        st.plotly_chart(fig, width="stretch")

    with col_arrows:
        st.markdown("#### 3-Mo vs 12-Mo Trend")
       
        def calculate_trend_arrow(series_3mo, series_12mo, label_name):
            avg_3m = series_3mo.mean()
            avg_12m = series_12mo.mean()
           
            if avg_12m == 0:
                diff_pct = 0.0
            else:
                diff_pct = ((avg_3m - avg_12m) / avg_12m) * 100
               
            if diff_pct <= -10.0:
                # Downward trending and green
                colored_text = f"<span style='color:green; font-weight:bold;'>⬇️ Down {abs(diff_pct):.1f}% vs 12-mo avg</span>"
            elif diff_pct >= 10.0:
                # Upward trending and red
                colored_text = f"<span style='color:red; font-weight:bold;'>⬆️ Up {diff_pct:.1f}% vs 12-mo avg</span>"
            else:
                # Sideways trending and blue
                colored_text = f"<span style='color:blue; font-weight:bold;'>➡️ Flat ({diff_pct:+.1f}% vs 12-mo avg)</span>"
               
            st.markdown(f"**{label_name}:**<br>{colored_text}", unsafe_allow_html=True)

        last_3m = monthly_counts.tail(3)
       
        calculate_trend_arrow(last_3m["Total_Risk"], monthly_counts["Total_Risk"], "Overall")
        calculate_trend_arrow(last_3m["Financial"], monthly_counts["Financial"], "Financial")
        calculate_trend_arrow(last_3m["Operational"], monthly_counts["Operational"], "Operational")
        calculate_trend_arrow(last_3m["Strategic"], monthly_counts["Strategic"], "Strategic")
        calculate_trend_arrow(last_3m["Regulatory"], monthly_counts["Regulatory"], "Regulatory")

st.divider()


# ==============================================================================
# SECTION 6: RIGHT-HAND SIDE - RECENT THREATS (LAST 7 DAYS)
# Functionality: Vertical Bar Chart & Separate Tables per Risk Vector.
# How it works: Filters sector_df for past 7 days, renders individual tables
#               per risk vector without the vector column, with search fallback links.
# Data Sources: Filtered sector_df.
# Tools Used: Streamlit, Plotly Bar, Pandas Markdown formatting
# ==============================================================================
st.subheader("⚠️ Recent Threats - Last 7 Days")

seven_days_ago = pd.to_datetime(datetime.now() - timedelta(days=7))
recent_df = sector_df[sector_df["Date"] >= seven_days_ago].copy()

col_bar, col_tables = st.columns([1, 1.5])

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
        st.plotly_chart(fig_bar, width="stretch")
    else:
        st.info("No elevated risk threats detected in the last 7 days.")

with col_tables:
    st.markdown("#### Top Headlines by Risk Vector")
   
    risk_vectors_list = ["Financial", "Operational", "Strategic", "Regulatory"]
    has_any_threats = False
   
    for rv in risk_vectors_list:
        sub_rv = recent_df[recent_df["Risk_Vector"] == rv].head(3)
        if not sub_rv.empty:
            has_any_threats = True
            st.markdown(f"**{rv} Risk**")
           
            top_headlines = []
            for _, row in sub_rv.iterrows():
                if row["Link"] and str(row["Link"]).startswith("http"):
                    target_url = row["Link"]
                else:
                    target_url = f"https://www.google.com/search?q={urllib.parse.quote(row['Headline'])}"
                   
                formatted_link = f"[{row['Headline']}]({target_url})"
                top_headlines.append({
                    "Date": row["Date"].strftime("%Y-%m-%d"),
                    "Keyword": row["Keyword"],
                    "Headline": formatted_link
                })
               
            top_df = pd.DataFrame(top_headlines)
            st.write(top_df.to_markdown(index=False), unsafe_allow_html=True)
            st.markdown("") # spacing between tables
           
    if not has_any_threats:
        st.caption("No recent threat headlines available for display across risk vectors.")
