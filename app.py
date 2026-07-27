import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
import re
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# SECTION 1: PAGE CONFIGURATION & LAYOUT
# ==============================================================================
st.set_page_config(page_title="Enterprise Risk Scanner", page_icon="🏛️", layout="wide")

# Sidebar Dark Mode Toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Enterprise Market Risk Management Scanner")
st.markdown("**Real-Time & 12-Month Longitudinal Media Threat Intelligence**")
st.divider()


# ==============================================================================
# SECTION 2: INDUSTRY & REGIONAL MAPPINGS & AI CLASSIFIER TRAINING
# ==============================================================================
REGIONS_LIST = ["Global", "Americas (including U.S.)", "Europe (EMEA)", "Asia (APAC)"]

INDUSTRY_MAP = {
    "Industrials / Manufacturing": ["industrial", "factory", "manufacturing", "BMW", "Siemens", "automotive", "assembly"],
    "Technology": ["technology", "software", "semiconductor", "microchip", "Nvidia", "Apple", "AI", "cloud", "cybersecurity"],
    "Financials": ["banking", "credit", "financial", "JPMorgan", "interest rate", "liquidity", "wall street"],
    "Consumer & Retail": ["retail", "consumer", "Walmart", "spending", "e-commerce", "Nike", "store"],
    "Energy & Raw Materials": ["energy", "oil", "gas", "BASF", "commodity", "carbon", "power grid"]
}

REGION_TERM_MAP = {
    "Global": "global market",
    "Americas (including U.S.)": "US business",
    "Europe (EMEA)": "Europe business",
    "Asia (APAC)": "Asia business"
}

# Strict Risk Vector Order: Regulatory, Strategic, Operational, Financial
RISK_VECTORS_ORDER = ["Regulatory", "Strategic", "Operational", "Financial"]

VECTOR_COLORS = {
    "Regulatory": "#FF6B6B",    # Coral-Red
    "Strategic": "#DC143C",     # Crimson
    "Operational": "#B22222",   # Firebrick
    "Financial": "#8B0000"      # Dark Crimson
}

@st.cache_resource
def train_erm_classifier():
    training_corpus = [
        # Financial
        ("Quarterly profit loss recorded due to debt liquidity crunch", "Financial"),
        ("Credit downgrade risk increases as revenue misses estimates", "Financial"),
        ("Rising interest rate expense squeezes corporate margin balance", "Financial"),
        
        # Operational (General + Tech-specific)
        ("Factory shutdown imminent as supplier microchip delivery halts", "Operational"),
        ("Transit port congestion causes shipping bottleneck delays", "Operational"),
        ("Labor strike stops manufacturing plant production line", "Operational"),
        ("Major cloud service outage causes widespread enterprise server downtime", "Operational"),
        ("Critical cybersecurity data breach exposes millions of user credentials", "Operational"),
        ("Ransomware cyber attack paralyzes software infrastructure systems", "Operational"),
        
        # Strategic (General + Tech-specific)
        ("Rival release causes sudden loss of market share dominance", "Strategic"),
        ("Delayed EV transition compromises multi-year market position", "Strategic"),
        ("Failed merger strategy leaves corporate growth outlook uncertain", "Strategic"),
        ("Missed generative AI shift leads to rapid customer attrition", "Strategic"),
        ("Semiconductor supply shortage stalls hardware product roadmap", "Strategic"),
        
        # Regulatory (General + Tech-specific)
        ("EU emission fine increases cross-border export tariff burden", "Regulatory"),
        ("Antitrust inquiry launched over non-compliance trade practices", "Regulatory"),
        ("Bilateral export sanctions restrict international market access", "Regulatory"),
        ("Big tech antitrust investigation launched over anticompetitive app store rules", "Regulatory"),
        ("Severe data privacy compliance fine issued for GDPR violations", "Regulatory"),
        
        # General (Neutral / Positive) & Consumer Banking
        ("Company beats earnings estimates with record quarterly output", "General (Neutral / Positive)"),
        ("New automated facility opens boosting operational efficiency", "General (Neutral / Positive)"),
        ("Strategic partnership established to accelerate clean technology", "General (Neutral / Positive)"),
        ("Checking account bonuses and promotional cash rewards attract depositors", "General (Neutral / Positive)"),
        ("Retail banking promotional rates and high yield incentives announced", "General (Neutral / Positive)"),
        ("Apple unveils breakthrough developer software tools at annual conference", "General (Neutral / Positive)"),
        ("Nvidia reports record quarterly revenue driven by high AI chip demand", "General (Neutral / Positive)")
    ]
    
    train_df = pd.DataFrame(training_corpus, columns=["Headline", "Label"])
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(train_df["Headline"])
    model = LogisticRegression(random_state=42)
    model.fit(X_train, train_df["Label"])
    return vectorizer, model

vectorizer, erm_model = train_erm_classifier()


# ==============================================================================
# SECTION 3: DATA INGESTION PIPELINE (ROBUST RSS QUERY + ADVANCED CLEANING)
# ==============================================================================
def clean_headline(raw_title):
    if not raw_title:
        return ""
    cleaned = re.sub(r'^\[.*?\]\s*', '', raw_title).strip()
    cleaned = cleaned.lstrip('[').strip()
    
    for separator in [' - ', ' | ']:
        if separator in cleaned:
            parts = cleaned.rsplit(separator, 1)
            if len(parts[0]) > 15:
                cleaned = parts[0]
                
    return cleaned.strip()

@st.cache_data(ttl=600)
def load_combined_dataset(selected_region, selected_industry):
    try:
        hist_df = pd.read_csv("historical_news.csv")
    except FileNotFoundError:
        hist_df = pd.DataFrame(columns=["Date", "Region", "Industry", "Risk_Vector", "Keyword", "Headline", "Link"])

    keywords = INDUSTRY_MAP[selected_industry]
    geo_term = REGION_TERM_MAP.get(selected_region, "business")
    
    query = f"{keywords[0]} {geo_term}"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    live_records = []
    
    for entry in feed.entries[:30]:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            parsed_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
        else:
            parsed_date = datetime.now().strftime("%Y-%m-%d")
            
        raw_headline = getattr(entry, "title", "")
        link = getattr(entry, "link", "")
        
        cleaned_headline = clean_headline(raw_headline)
        if len(cleaned_headline) < 15:
            continue
            
        X_test = vectorizer.transform([cleaned_headline])
        pred_vector = erm_model.predict(X_test)[0]
        
        live_records.append({
            "Date": parsed_date,
            "Region": selected_region,
            "Industry": selected_industry,
            "Risk_Vector": pred_vector,
            "Keyword": keywords[0].capitalize(),
            "Headline": cleaned_headline,
            "Link": link
        })
        
    live_df = pd.DataFrame(live_records)
    
    full_df = pd.concat([hist_df, live_df], ignore_index=True)
    full_df["Date"] = pd.to_datetime(full_df["Date"])
    return full_df


# ==============================================================================
# SECTION 4: LEFT-HAND SIDE DROPDOWN CONTROLS
# ==============================================================================
st.sidebar.header("⚙️ Filter Controls")

selected_region = st.sidebar.selectbox("1) Select Region:", options=REGIONS_LIST)
selected_industry = st.sidebar.selectbox("2) Select Primary Industry:", options=list(INDUSTRY_MAP.keys()))

st.markdown(f"### 🌐 Active Scope: **{selected_industry}** | 📍 **{selected_region}**")
st.markdown("---")

full_dataset = load_combined_dataset(selected_region, selected_industry)

sector_df = full_dataset[
    (full_dataset["Industry"] == selected_industry) & 
    (full_dataset["Region"].str.contains(selected_region.split()[0], case=False, na=False))
].copy()


# ==============================================================================
# SECTION 5: RIGHT-HAND SIDE - RISK TREND ANALYSIS (STACKED BAR CHART)
# ==============================================================================
st.subheader("📈 Risk Trend Analysis - Past 12 Months")
st.caption(f"Monthly Risk Vector Volume for **{selected_industry}** in **{selected_region}**")

col_graph, col_arrows = st.columns([2, 1])

if not sector_df.empty:
    sector_df["YearMonth"] = sector_df["Date"].dt.to_period("M").dt.to_timestamp()
    
    risk_only_df = sector_df[sector_df["Risk_Vector"] != "General (Neutral / Positive)"]
    monthly_counts = risk_only_df.groupby(["YearMonth", "Risk_Vector"]).size().unstack(fill_value=0)
    
    for rv in RISK_VECTORS_ORDER:
        if rv not in monthly_counts.columns:
            monthly_counts[rv] = 0

    with col_graph:
        fig = go.Figure()

        for rv in RISK_VECTORS_ORDER:
            fig.add_trace(go.Bar(
                x=monthly_counts.index, y=monthly_counts[rv],
                name=rv, marker_color=VECTOR_COLORS[rv]
            ))

        fig.update_layout(
            barmode='stack',
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(title="Last 12 Months"),
            yaxis=dict(title="Headline Count"),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
        )
        st.plotly_chart(fig, width="stretch")

    with col_arrows:
        st.markdown("#### 3-Mo vs 12-Mo Trend")
        
        monthly_total_risk = monthly_counts.sum(axis=1)
        
        def calculate_trend_arrow(series_3mo, series_12mo, label_name):
            avg_3m = series_3mo.mean()
            avg_12m = series_12mo.mean()
            
            if avg_12m == 0:
                diff_pct = 0.0
            else:
                diff_pct = ((avg_3m - avg_12m) / avg_12m) * 100
                
            if diff_pct <= -10.0:
                colored_text = f"<span style='color:green; font-weight:bold;'>⬇️ Down {abs(diff_pct):.1f}% vs 12-mo avg</span>"
            elif diff_pct >= 10.0:
                colored_text = f"<span style='color:red; font-weight:bold;'>⬆️ Up {diff_pct:.1f}% vs 12-mo avg</span>"
            else:
                colored_text = f"<span style='color:blue; font-weight:bold;'>➡️ Flat ({diff_pct:+.1f}% vs 12-mo avg)</span>"
                
            st.markdown(f"**{label_name}:**<br>{colored_text}", unsafe_allow_html=True)

        last_3m_total = monthly_total_risk.tail(3)
        calculate_trend_arrow(last_3m_total, monthly_total_risk, "Overall Risk")
        
        for rv in RISK_VECTORS_ORDER:
            last_3m_rv = monthly_counts[rv].tail(3)
            calculate_trend_arrow(last_3m_rv, monthly_counts[rv], rv)

st.divider()


# ==============================================================================
# SECTION 6: RECENT THREATS (LAST 14 DAYS) & SEPARATE SORTED TABLES
# ==============================================================================
st.subheader("⚠️ Recent Threats - Last 14 Days")

fourteen_days_ago = pd.to_datetime(datetime.now() - timedelta(days=14))
recent_df = sector_df[
    (sector_df["Date"] >= fourteen_days_ago) & 
    (sector_df["Risk_Vector"] != "General (Neutral / Positive)") &
    (sector_df["Link"].str.startswith("http", na=False))
].copy()

recent_df = recent_df.sort_values(by="Date", ascending=False)

col_bar, col_tables = st.columns([1, 1.5])

with col_bar:
    st.markdown("#### Threat Count by Vector")
    
    if not recent_df.empty:
        bar_counts = recent_df["Risk_Vector"].value_counts()
    else:
        bar_counts = pd.Series(0, index=RISK_VECTORS_ORDER)
        
    bar_counts = bar_counts.reindex(RISK_VECTORS_ORDER, fill_value=0).reset_index()
    bar_counts.columns = ["Risk Vector", "Count"]
    
    bar_colors = [VECTOR_COLORS[rv] for rv in bar_counts["Risk Vector"]]
    
    fig_bar = go.Figure(go.Bar(
        x=bar_counts["Risk Vector"], y=bar_counts["Count"],
        marker_color=bar_colors
    ))
    fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bar, width="stretch")

with col_tables:
    st.markdown("#### Top Headlines by Risk Vector")
    
    has_any_threats = False
    
    for rv in RISK_VECTORS_ORDER:
        sub_rv = recent_df[recent_df["Risk_Vector"] == rv].head(3)
        if not sub_rv.empty:
            has_any_threats = True
            st.markdown(f"**{rv} Risk**")
            
            top_headlines = []
            for _, row in sub_rv.iterrows():
                target_url = row["Link"]
                formatted_link = f"[{row['Headline']}]({target_url})"
                top_headlines.append({
                    "Date": row["Date"].strftime("%Y-%m-%d"),
                    "Keyword": row["Keyword"],
                    "Headline": formatted_link
                })
                
            top_df = pd.DataFrame(top_headlines)
            st.write(top_df.to_markdown(index=False), unsafe_allow_html=True)
            st.markdown("") 
            
    if not has_any_threats:
        st.caption("No direct article threat links available for display across risk vectors.")
