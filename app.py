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
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 1: PAGE CONFIGURATION & SESSION STATE
# ==============================================================================
st.set_page_config(
    page_title="Enterprise Risk Scanner",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page navigation via session state
if "page" not in st.session_state:
    st.session_state.page = "main"

# Dark-mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# Comprehensive light / dark CSS – backgrounds, text, buttons, dropdowns, inputs
if dark_mode:
    st.markdown(
        """
        <style>
        /* ---------- Base app ---------- */
        .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: #0e1117 !important;
            color: #f0f2f6 !important;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #161b22 !important;
            color: #f0f2f6 !important;
        }
        [data-testid="stSidebar"] * {
            color: #f0f2f6 !important;
        }

        /* ---------- Headings & body text ---------- */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        .stCaption, label, .stText, p, span, div {
            color: #f0f2f6 !important;
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background-color: #21262d !important;
            color: #f0f2f6 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
        }
        .stButton > button:hover {
            background-color: #30363d !important;
            border-color: #8b949e !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background-color: #238636 !important;
            border-color: #2ea043 !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #2ea043 !important;
        }

        /* ---------- Selectbox / Dropdown ---------- */
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] div[data-baseweb="select"] {
            color: #f0f2f6 !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #21262d !important;
            color: #f0f2f6 !important;
            border-color: #30363d !important;
        }
        /* Dropdown menu popup */
        div[data-baseweb="popover"] div[data-baseweb="menu"],
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li {
            background-color: #21262d !important;
            color: #f0f2f6 !important;
        }
        div[data-baseweb="popover"] li:hover {
            background-color: #30363d !important;
        }

        /* ---------- Toggle ---------- */
        [data-testid="stToggle"] label {
            color: #f0f2f6 !important;
        }

        /* ---------- Text inputs / other widgets ---------- */
        .stTextInput input, .stNumberInput input,
        [data-baseweb="input"] {
            background-color: #21262d !important;
            color: #f0f2f6 !important;
            border-color: #30363d !important;
        }

        /* ---------- Dataframes / tables ---------- */
        [data-testid="stDataFrame"], .dataframe {
            background-color: #1c2128 !important;
            color: #f0f2f6 !important;
        }

        /* ---------- Dividers ---------- */
        hr, .stDivider {
            border-color: #30363d !important;
        }

        /* ---------- Links ---------- */
        a {
            color: #58a6ff !important;
        }

        /* ---------- Info / success / warning boxes ---------- */
        [data-testid="stAlert"] {
            background-color: #1c2128 !important;
            color: #f0f2f6 !important;
        }

        /* ---------- Metric ---------- */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: #f0f2f6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        /* ---------- Light mode base ---------- */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f0f2f6 !important;
            color: #1a1a1a !important;
        }

        /* ---------- Buttons (light) ---------- */
        .stButton > button {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border: 1px solid #d0d5dd !important;
            border-radius: 6px !important;
        }
        .stButton > button:hover {
            background-color: #f0f2f6 !important;
            border-color: #98a2b3 !important;
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background-color: #ff4b4b !important;
            border-color: #ff4b4b !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #e03c3c !important;
        }

        /* ---------- Selectbox (light) ---------- */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border-color: #d0d5dd !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# SECTION 2: INDUSTRY & REGIONAL MAPPINGS & AI CLASSIFIER
# ==============================================================================
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

RISK_VECTORS_ORDER = ["Regulatory", "Strategic", "Operational", "Financial"]

VECTOR_COLORS = {
    "Regulatory": "#FF6B6B",
    "Strategic": "#DC143C",
    "Operational": "#B22222",
    "Financial": "#8B0000",
}

HISTORICAL_CSV = Path("historical_news.csv")

@st.cache_resource
def train_erm_classifier():
    """Train a simple TF-IDF + Logistic Regression risk classifier.
    Note: Training set is intentionally small for demo purposes.
    For production, expand significantly or replace with a stronger model.
    """
    training_corpus = [
        # Financial
        ("Quarterly profit loss recorded due to debt liquidity crunch", "Financial"),
        ("Credit downgrade risk increases as revenue misses estimates", "Financial"),
        ("Rising interest rate expense squeezes corporate margin balance", "Financial"),
        ("Bank reports sharp increase in non-performing loans", "Financial"),
        ("Currency volatility hits quarterly earnings guidance", "Financial"),
        ("Debt refinancing costs surge amid higher rates", "Financial"),

        # Operational
        ("Factory shutdown imminent as supplier microchip delivery halts", "Operational"),
        ("Transit port congestion causes shipping bottleneck delays", "Operational"),
        ("Labor strike stops manufacturing plant production line", "Operational"),
        ("Major cloud service outage causes widespread enterprise server downtime", "Operational"),
        ("Critical cybersecurity data breach exposes millions of user credentials", "Operational"),
        ("Ransomware cyber attack paralyzes software infrastructure systems", "Operational"),
        ("Supply chain disruption forces temporary plant closure", "Operational"),
        ("Severe weather disrupts logistics and warehouse operations", "Operational"),

        # Strategic
        ("Rival release causes sudden loss of market share dominance", "Strategic"),
        ("Delayed EV transition compromises multi-year market position", "Strategic"),
        ("Failed merger strategy leaves corporate growth outlook uncertain", "Strategic"),
        ("Missed generative AI shift leads to rapid customer attrition", "Strategic"),
        ("Semiconductor supply shortage stalls hardware product roadmap", "Strategic"),
        ("New competitor enters market with aggressive pricing strategy", "Strategic"),
        ("Loss of key customer contract threatens long-term revenue", "Strategic"),

        # Regulatory
        ("EU emission fine increases cross-border export tariff burden", "Regulatory"),
        ("Antitrust inquiry launched over non-compliance trade practices", "Regulatory"),
        ("Bilateral export sanctions restrict international market access", "Regulatory"),
        ("Big tech antitrust investigation launched over anticompetitive app store rules", "Regulatory"),
        ("Severe data privacy compliance fine issued for GDPR violations", "Regulatory"),
        ("New environmental regulations raise compliance costs for manufacturers", "Regulatory"),
        ("Securities regulator opens investigation into accounting practices", "Regulatory"),

        # Neutral / Positive
        ("Company beats earnings estimates with record quarterly output", "General (Neutral / Positive)"),
        ("New automated facility opens boosting operational efficiency", "General (Neutral / Positive)"),
        ("Strategic partnership established to accelerate clean technology", "General (Neutral / Positive)"),
        ("Checking account bonuses and promotional cash rewards attract depositors", "General (Neutral / Positive)"),
        ("Retail banking promotional rates and high yield incentives announced", "General (Neutral / Positive)"),
        ("Apple unveils breakthrough developer software tools at annual conference", "General (Neutral / Positive)"),
        ("Nvidia reports record quarterly revenue driven by high AI chip demand", "General (Neutral / Positive)"),
        ("Successful product launch drives strong consumer demand", "General (Neutral / Positive)"),
        ("Company announces share buyback program and dividend increase", "General (Neutral / Positive)"),
    ]

    train_df = pd.DataFrame(training_corpus, columns=["Headline", "Label"])
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X_train = vectorizer.fit_transform(train_df["Headline"])
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    model.fit(X_train, train_df["Label"])
    return vectorizer, model

vectorizer, erm_model = train_erm_classifier()

# ==============================================================================
# SECTION 3: DATA INGESTION PIPELINE
# ==============================================================================
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

def generate_sample_historical() -> pd.DataFrame:
    """Create realistic sample historical data so the 12-month view works on first run."""
    rng = np.random.default_rng(42)
    rows = []
    base = datetime.now() - timedelta(days=365)

    sample_headlines = {
        "Financial": [
            "Rising rates pressure corporate debt refinancing costs",
            "Bank warns of higher credit loss provisions this quarter",
            "Liquidity concerns emerge after unexpected revenue miss",
        ],
        "Operational": [
            "Port congestion delays critical component deliveries",
            "Cyber incident disrupts manufacturing systems for two days",
            "Labor shortage forces temporary reduction in plant output",
        ],
        "Strategic": [
            "Competitor gains share after delayed product launch",
            "Missed technology shift raises questions on long-term positioning",
            "Failed acquisition attempt leaves growth strategy unclear",
        ],
        "Regulatory": [
            "New emissions rules increase compliance burden for exporters",
            "Antitrust probe opened into industry pricing practices",
            "Data privacy fine issued under updated regional regulations",
        ],
    }

    industries = list(INDUSTRY_MAP.keys())
    regions = REGIONS_LIST

    for month_offset in range(12):
        month_start = base + timedelta(days=30 * month_offset)
        for _ in range(rng.integers(4, 12)):
            risk = rng.choice(RISK_VECTORS_ORDER)
            industry = rng.choice(industries)
            region = rng.choice(regions)
            headline = rng.choice(sample_headlines[risk])
            day = int(rng.integers(1, 28))
            date = (month_start.replace(day=1) + timedelta(days=day - 1)).strftime("%Y-%m-%d")
            rows.append(
                {
                    "Date": date,
                    "Region": region,
                    "Industry": industry,
                    "Risk_Vector": risk,
                    "Keyword": INDUSTRY_MAP[industry][0].capitalize(),
                    "Headline": headline,
                    "Link": "https://news.google.com/",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(HISTORICAL_CSV, index=False)
    return df

@st.cache_data(ttl=600, show_spinner="Fetching latest headlines…")
def load_combined_dataset(selected_region: str, selected_industry: str) -> pd.DataFrame:
    # Load or create historical data
    if HISTORICAL_CSV.exists():
        try:
            hist_df = pd.read_csv(HISTORICAL_CSV)
        except Exception:
            hist_df = generate_sample_historical()
    else:
        hist_df = generate_sample_historical()

    # Build richer Google News RSS query
    keywords = INDUSTRY_MAP[selected_industry]
    keyword_part = " OR ".join(f'"{k}"' if " " in k else k for k in keywords[:6])
    geo_term = REGION_TERM_MAP.get(selected_region, "business")
    query = f"({keyword_part}) ({geo_term})"
    encoded_query = urllib.parse.quote(query)
    rss_url = (
        f"https://news.google.com/rss/search?q={encoded_query}"
        f"&hl=en-US&gl=US&ceid=US:en"
    )

    live_records = []
    try:
        feed = feedparser.parse(rss_url)
        entries = getattr(feed, "entries", [])[:40]

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

            X_test = vectorizer.transform([cleaned])
            pred_vector = erm_model.predict(X_test)[0]

            live_records.append(
                {
                    "Date": parsed_date,
                    "Region": selected_region,
                    "Industry": selected_industry,
                    "Risk_Vector": pred_vector,
                    "Keyword": keywords[0].capitalize(),
                    "Headline": cleaned,
                    "Link": link if link.startswith("http") else "",
                }
            )
    except Exception as e:
        st.warning(f"Live feed temporarily unavailable ({type(e).__name__}). Showing historical data only.")

    live_df = pd.DataFrame(live_records) if live_records else pd.DataFrame(
        columns=["Date", "Region", "Industry", "Risk_Vector", "Keyword", "Headline", "Link"]
    )

    # Persist new live records (deduplicated)
    if not live_df.empty:
        combined = pd.concat([hist_df, live_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Headline", "Date"], keep="last")
        try:
            combined.to_csv(HISTORICAL_CSV, index=False)
        except Exception:
            pass  # non-fatal if write fails
        full_df = combined
    else:
        full_df = hist_df

    full_df["Date"] = pd.to_datetime(full_df["Date"], errors="coerce")
    full_df = full_df.dropna(subset=["Date"])
    return full_df

# ==============================================================================
# SECTION 4: SIDEBAR – ALWAYS VISIBLE CONTROLS + ABOUT BUTTON
# ==============================================================================
st.sidebar.header("⚙️ Filter Controls")

# About button in sidebar (always available)
if st.sidebar.button("ℹ️ About This Project", use_container_width=True):
    st.session_state.page = "about"
    st.rerun()

st.sidebar.markdown("---")

selected_region = st.sidebar.selectbox("1) Select Region:", options=REGIONS_LIST)
selected_industry = st.sidebar.selectbox(
    "2) Select Primary Industry:", options=list(INDUSTRY_MAP.keys())
)

if st.sidebar.button("🔄 Refresh Data", help="Clear cache and re-fetch live headlines"):
    st.cache_data.clear()
    st.rerun()

# ==============================================================================
# SECTION 5: PAGE ROUTING
# ==============================================================================
if st.session_state.page == "about":
    # -------------------- ABOUT PAGE --------------------
    st.title("ℹ️ About This Project")
    st.markdown("**Project Outline: Interactive Enterprise Market Risk Scanner**")
    st.divider()

    # Back button at the top for convenience
    if st.button("⬅️ Back to Risk Scanner", type="primary"):
        st.session_state.page = "main"
        st.rerun()

    st.markdown("---")

    # 1. What This Project Does
    st.header("1. What This Project Does (Business Purpose)")
    st.markdown(
        """
Businesses and investors deal with thousands of news articles every day, making it impossible to spot major threats—like supply chain failures, lawsuits, or financial crashes—before it's too late. This app acts as an AI-powered early warning system that automatically reads, groups, and sorts business news headlines.
        """
    )

    st.subheader("How Managers Can Use It")
    st.markdown(
        """
Operations managers can track recent risk trends in their specific industry and region (like manufacturing delays in Europe) so they can fix problems before they turn into full-blown crises.
        """
    )

    st.subheader("How Investors Can Use It")
    st.markdown(
        """
Investors can look at past 12-month threat charts to see how a company or sector's risk levels have changed over time, helping them make smarter choices with their money.
        """
    )

    st.markdown("---")

    # 2. Tools Used
    st.header("2. Tools Used")
    st.markdown(
        """
* **Streamlit**: Powers the interactive web dashboard where users can click dropdown menus, toggle dark mode, and view live charts.
* **GitHub**: Stores all the project code files and uses automated workflows to run background updates every single night.
* **Google RSS**: Pulls live, real-time news headlines based on the industry and region you select.
* **Kaggle / CSV Data**: Stores the historical news records so the app can show past trends over the full year without relying only on today's news.
* **Python & Scikit-Learn**: Uses machine learning (TF-IDF and Logistic Regression) to read incoming headlines and sort them into different risk categories.
        """
    )

    st.markdown("---")

    # 3. Data Sources Used
    st.header("3. Data Sources Used")
    st.markdown(
        """
* **Live Google News Feeds**: Pulls fresh news articles on the fly.
* **Historical CSV File**: Stores older saved news data to keep past months visible on the charts.
        """
    )

    st.markdown("---")

    # 4. Limitations
    st.header("4. Limitations")
    st.markdown(
        """
* **Smart Guessing (Model Limits)**: The AI was trained on a small starter list of examples, so if it reads brand-new or unusual phrasing (like a strange banking ad), it might occasionally mislabel it.
* **Quiet News Cycles**: If an industry hasn't had much media attention lately, the live feed might look a bit empty.
* **Messy Headlines**: Raw news titles often come with weird tags (like [Opinion]) or publisher names that require extra cleaning to look nice on the screen.
        """
    )

    st.markdown("---")

    # 5. Future Ideas
    st.header("5. Future Ideas")
    st.markdown(
        """
* **Smarter AI Models**: Upgrading to advanced deep learning tools that understand business tone and context even better.
* **Automatic Alerts**: Setting up text or email alerts to notify managers immediately if a certain type of risk suddenly spikes.
* **Multi-Language News**: Expanding the app to scan business news written in other languages, like German financial reports.
        """
    )

    st.markdown("---")
    if st.button("⬅️ Back to Risk Scanner", key="back_bottom", type="primary"):
        st.session_state.page = "main"
        st.rerun()

else:
    # -------------------- MAIN RISK SCANNER PAGE --------------------
    st.title("🏛️ Enterprise Market Risk Management Scanner")
    st.markdown("**Real-Time & 12-Month Longitudinal Media Threat Intelligence**")

    # About button immediately below the main heading
    if st.button("ℹ️ About This Project"):
        st.session_state.page = "about"
        st.rerun()

    st.divider()

    st.markdown(f"### 🌐 Active Scope: **{selected_industry}** | 📍 **{selected_region}**")
    st.markdown("---")

    full_dataset = load_combined_dataset(selected_region, selected_industry)

    # Exact region + industry match
    sector_df = full_dataset[
        (full_dataset["Industry"] == selected_industry)
        & (full_dataset["Region"] == selected_region)
    ].copy()

    # Plotly template based on dark mode
    plotly_template = "plotly_dark" if dark_mode else "plotly_white"

    # ==============================================================================
    # RISK TREND ANALYSIS (STACKED BAR – LAST 12 MONTHS)
    # ==============================================================================
    st.subheader("📈 Risk Trend Analysis – Past 12 Months")
    st.caption(f"Monthly Risk Vector Volume for **{selected_industry}** in **{selected_region}**")

    col_graph, col_arrows = st.columns([2.2, 1])

    if sector_df.empty:
        st.info("No historical data available for this combination yet. Try another region/industry or wait for live data.")
    else:
        # Restrict to last 12 months and force complete month index
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
        sector_df = sector_df[sector_df["Date"] >= cutoff].copy()

        sector_df["YearMonth"] = sector_df["Date"].dt.to_period("M").dt.to_timestamp()

        risk_only_df = sector_df[sector_df["Risk_Vector"] != "General (Neutral / Positive)"]

        end_month = pd.Timestamp.now().to_period("M").to_timestamp()
        start_month = end_month - pd.DateOffset(months=11)
        full_months = pd.date_range(start_month, end_month, freq="MS")

        monthly_counts = (
            risk_only_df.groupby(["YearMonth", "Risk_Vector"])
            .size()
            .unstack(fill_value=0)
            .reindex(full_months, fill_value=0)
        )

        for rv in RISK_VECTORS_ORDER:
            if rv not in monthly_counts.columns:
                monthly_counts[rv] = 0
        monthly_counts = monthly_counts[RISK_VECTORS_ORDER]

        with col_graph:
            fig = go.Figure()
            for rv in RISK_VECTORS_ORDER:
                fig.add_trace(
                    go.Bar(
                        x=monthly_counts.index,
                        y=monthly_counts[rv],
                        name=rv,
                        marker_color=VECTOR_COLORS[rv],
                    )
                )
            fig.update_layout(
                barmode="stack",
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title="Last 12 Months", tickformat="%b %Y"),
                yaxis=dict(title="Headline Count"),
                legend=dict(
                    orientation="h",
                    y=1.08,
                    x=1,
                    xanchor="right",
                    yanchor="bottom",
                ),
                template=plotly_template,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f2f6" if dark_mode else "#1a1a1a"),
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with col_arrows:
            st.markdown("#### 3-Mo vs 12-Mo Trend")

            monthly_total_risk = monthly_counts.sum(axis=1)

            def calculate_trend_arrow(series_3mo, series_12mo, label_name):
                avg_3m = series_3mo.mean() if len(series_3mo) > 0 else 0.0
                avg_12m = series_12mo.mean() if len(series_12mo) > 0 else 0.0

                if avg_12m == 0:
                    diff_pct = 0.0
                else:
                    diff_pct = ((avg_3m - avg_12m) / avg_12m) * 100

                if diff_pct <= -10.0:
                    colored = f"<span style='color:#2ecc71; font-weight:bold;'>⬇️ Down {abs(diff_pct):.1f}% vs 12-mo avg</span>"
                elif diff_pct >= 10.0:
                    colored = f"<span style='color:#e74c3c; font-weight:bold;'>⬆️ Up {diff_pct:.1f}% vs 12-mo avg</span>"
                else:
                    colored = f"<span style='color:#3498db; font-weight:bold;'>➡️ Flat ({diff_pct:+.1f}% vs 12-mo avg)</span>"

                st.markdown(f"**{label_name}:**<br>{colored}", unsafe_allow_html=True)

            last_3m_total = monthly_total_risk.tail(3)
            calculate_trend_arrow(last_3m_total, monthly_total_risk, "Overall Risk")

            for rv in RISK_VECTORS_ORDER:
                last_3m_rv = monthly_counts[rv].tail(3)
                calculate_trend_arrow(last_3m_rv, monthly_counts[rv], rv)

    st.divider()

    # ==============================================================================
    # RECENT THREATS (LAST 14 DAYS) – PIE CHART
    # ==============================================================================
    st.subheader("⚠️ Recent Threats – Last 14 Days")

    fourteen_days_ago = pd.Timestamp.now() - timedelta(days=14)

    recent_df = sector_df[
        (sector_df["Date"] >= fourteen_days_ago)
        & (sector_df["Risk_Vector"] != "General (Neutral / Positive)")
    ].copy()

    recent_df = recent_df.sort_values(by="Date", ascending=False)

    col_pie, col_tables = st.columns([1, 1.6])

    with col_pie:
        st.markdown("#### Threat Distribution by Vector")

        if not recent_df.empty:
            pie_counts = recent_df["Risk_Vector"].value_counts()
        else:
            pie_counts = pd.Series(0, index=RISK_VECTORS_ORDER)

        pie_counts = pie_counts.reindex(RISK_VECTORS_ORDER, fill_value=0)
        pie_colors = [VECTOR_COLORS[rv] for rv in pie_counts.index]

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=pie_counts.index,
                    values=pie_counts.values,
                    marker=dict(colors=pie_colors),
                    textinfo="label+percent+value",
                    hole=0.35,
                    sort=False,
                )
            ]
        )
        fig_pie.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            template=plotly_template,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            font=dict(color="#f0f2f6" if dark_mode else "#1a1a1a"),
        )
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

    with col_tables:
        st.markdown("#### Top Headlines by Risk Vector")

        has_any_threats = False

        for rv in RISK_VECTORS_ORDER:
            sub_rv = recent_df[recent_df["Risk_Vector"] == rv].head(4)
            if sub_rv.empty:
                continue

            has_any_threats = True
            st.markdown(f"**{rv} Risk**")

            display_rows = []
            for _, row in sub_rv.iterrows():
                link = row["Link"] if isinstance(row["Link"], str) and row["Link"].startswith("http") else None
                headline_md = f"[{row['Headline']}]({link})" if link else row["Headline"]
                display_rows.append(
                    {
                        "Date": row["Date"].strftime("%Y-%m-%d"),
                        "Keyword": row["Keyword"],
                        "Headline": headline_md,
                    }
                )

            top_df = pd.DataFrame(display_rows)
            st.markdown(top_df.to_markdown(index=False), unsafe_allow_html=True)
            st.markdown("")

        if not has_any_threats:
            st.caption(
                "No recent risk-classified headlines with usable links found for this scope. "
                "Try refreshing data or selecting a different region/industry."
            )

    # Footer note
    st.divider()
    st.caption(
        "Classifier is a lightweight demo model (TF-IDF + Logistic Regression). "
        "Expand the training corpus or replace with a stronger NLP model for production use. "
        "Data is automatically persisted to historical_news.csv."
    )
