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
    page_title="News Headline Risk Scanner",
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
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f0f2f6 !important;
        }

        /* ---------- Headings & body text (main area) ---------- */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        .stCaption, label, .stText, p {
            color: #f0f2f6 !important;
        }

        /* ---------- Buttons (dark mode – high contrast) ---------- */
        .stButton > button,
        div[data-testid="stSidebar"] .stButton > button {
            background-color: #21262d !important;
            color: #f0f2f6 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
        }
        .stButton > button:hover,
        div[data-testid="stSidebar"] .stButton > button:hover {
            background-color: #30363d !important;
            border-color: #8b949e !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"],
        div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background-color: #238636 !important;
            border-color: #2ea043 !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #2ea043 !important;
        }

        /* ---------- Selectbox / Dropdown – keep light appearance, force dark text ---------- */
        [data-testid="stSelectbox"] label {
            color: #f0f2f6 !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border-color: #d0d5dd !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stSelectbox"] div[data-baseweb="select"] div {
            color: #1a1a1a !important;
        }
        div[data-baseweb="popover"] div[data-baseweb="menu"],
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] li span {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
        }
        div[data-baseweb="popover"] li:hover {
            background-color: #f0f2f6 !important;
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
    "Regulatory": "#FF8A80",  # Light coral / soft red
    "Strategic": "#FF1744",   # Vivid bright red
    "Operational": "#C62828", # Strong medium red
    "Financial": "#6D1B1B",   # Deep burgundy / dark red
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
    st.markdown("**Project Outline: News Headline Risk Scanner**")
    st.divider()

    # Back button at the top for convenience
    if st.button("⬅️ Back to Risk Scanner", type="primary"):
        st.session_state.page = "main"
        st.rerun()

    st.markdown("---")

    # 1. Business Purpose
    st.header("1. Business Purpose")
    st.markdown(
        """
Business managers and investors face many news articles every day, making it difficult to spot threats—like supply chain failures, lawsuits, or financial stress. This app runs as an AI-powered early warning system that automatically reads, groups, and sorts business news headlines based on potential threats.
        """
    )

    st.subheader("How Managers Can Use It")
    st.markdown(
        """
Business managers can identify recent risk trends in their specific industry and region (like manufacturing in Europe) so they can mitigate problems before it is too late.
        """
    )

    st.subheader("How Investors Can Use It")
    st.markdown(
        """
Investors can look at the past 12-month risk trends to see how a company or sector's risk levels are trending, helping them make smarter investment decisions.
        """
    )

    st.markdown("---")

    # 2. How This Project Works
    st.header("2. How This Project Works")
    st.markdown(
        """
* **You choose a focus** — Pick an industry (like Technology or Manufacturing) and a region (like Europe or Asia). The app builds a smart search query from those choices.

* **It pulls live news** — The app reaches out to Google News using RSS feeds and downloads the latest headlines that match your filters.

* **It cleans the text** — News titles often arrive messy (with tags like “[Opinion]” or the publisher’s name stuck on the end). Simple text-cleaning rules remove the clutter so the computer can read them clearly.

* **It turns words into numbers with TF-IDF** —  
  The computer analyzes a training **corpus** (a collection of sample text documents) and scores each word based on how often it appears in a headline and how rare it is overall. Unusual, important words (like “ransomware” or “lawsuit”) get high scores. Common words (like “the” or “company”) get low scores. This turns every headline into a list of numbers the computer can understand.

* **A Logistic Regression model sorts the risks** —  
  The machine learning **classifier** was trained on example headlines from the corpus that were already labeled by risk type. It learned which word patterns usually mean Regulatory, Strategic, Operational, or Financial risk. When a new headline arrives, it looks at the word scores and picks the most likely risk category.

* **It remembers the past** — New headlines are saved into a CSV file and combined with older records so the app can show trends for the last 12 months, not just today.

* **It draws the charts** — Plotly creates the stacked bar chart (12-month trend), the donut chart (today's threats), and the daily trend chart so you can quickly see which types of risk are rising or falling.

* **Streamlit makes it interactive** — Everything runs inside a web dashboard where you can change filters, switch dark mode, and explore the results without writing any code yourself.
        """
    )

    st.markdown("---")

    # 3. Tools Used
    st.header("3. Tools Used")
    st.markdown(
        """
* **Python & Scikit-Learn**: Uses machine learning (TF-IDF and Logistic Regression) to read incoming headlines and sort them into different risk categories.
* **Kaggle / CSV Data**: Stores the historical news records so the app can show past trends over the full year without relying only on today's news.
* **Google RSS**: Pulls live, real-time news headlines based on the industry and region you select.
* **Streamlit**: Powers the interactive web dashboard where users can click dropdown menus, toggle dark mode, and view live charts.
* **GitHub**: Stores all the project code files and uses automated workflows to run background updates every single night.
        """
    )

    st.markdown("---")

    # 4. Limitations
    st.header("4. Limitations")
    st.markdown(
        """
* **Smart Guessing (Model Limits)**: The AI was trained on a small starter list of examples, so if it reads brand-new or unusual phrasing (like a strange banking ad), it might occasionally mislabel it.
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
    st.title("🏛️ News Headline Risk Scanner")
    st.markdown("**Real-Time News Headline Risk Scanner**")

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
        # Restrict to last 12 months
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
        sector_df = sector_df[sector_df["Date"] >= cutoff].copy()

        # Normalize to month-start timestamps for consistent alignment
        sector_df["YearMonth"] = sector_df["Date"].dt.to_period("M").dt.to_timestamp()

        risk_only_df = sector_df[sector_df["Risk_Vector"] != "General (Neutral / Positive)"]

        # Build a complete 12-month calendar (always show every month)
        end_month = pd.Timestamp.now().to_period("M").to_timestamp()
        start_month = (pd.Timestamp.now() - pd.DateOffset(months=11)).to_period("M").to_timestamp()
        full_months = pd.date_range(start=start_month, end=end_month, freq="MS")

        # Count headlines per month × risk vector
        if risk_only_df.empty:
            monthly_counts = pd.DataFrame(0, index=full_months, columns=RISK_VECTORS_ORDER)
        else:
            monthly_counts = (
                risk_only_df.groupby(["YearMonth", "Risk_Vector"])
                .size()
                .unstack(fill_value=0)
            )
            # Force every calendar month onto the index (fill missing with 0)
            monthly_counts = monthly_counts.reindex(full_months, fill_value=0)

        # Ensure all four risk columns exist and are in a fixed order
        for rv in RISK_VECTORS_ORDER:
            if rv not in monthly_counts.columns:
                monthly_counts[rv] = 0
        monthly_counts = monthly_counts[RISK_VECTORS_ORDER].astype(int)

        # Convert to fixed string labels so Plotly treats every month as a category
        x_labels = [d.strftime("%b %Y") for d in monthly_counts.index]

        with col_graph:
            fig = go.Figure()
            for rv in RISK_VECTORS_ORDER:
                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=monthly_counts[rv].tolist(),
                        name=rv,
                        marker_color=VECTOR_COLORS[rv],
                    )
                )
            fig.update_layout(
                barmode="stack",
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(
                    title="Last 12 Months",
                    categoryorder="array",
                    categoryarray=x_labels,
                    tickmode="array",
                    tickvals=x_labels,
                    ticktext=x_labels,
                    tickangle=-45,
                ),
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
    # RECENT THREATS (TODAY'S DONUT & LAST 7 DAYS BAR CHART)
    # ==============================================================================
    st.subheader("⚠️ Recent Threats – Today & Last 7 Days")

    today_date = pd.Timestamp.now().normalize()
    seven_days_ago = pd.Timestamp.now() - timedelta(days=6)

    today_df = sector_df[
        (sector_df["Date"] >= today_date)
        & (sector_df["Risk_Vector"] != "General (Neutral / Positive)")
    ].copy()

    seven_days_df = sector_df[
        (sector_df["Date"] >= seven_days_ago)
        & (sector_df["Risk_Vector"] != "General (Neutral / Positive)")
    ].copy()
    seven_days_df = seven_days_df.sort_values(by="Date", ascending=False)

    col_donut, col_bar, col_tables = st.columns([1, 1.2, 1.5])

    with col_donut:
        st.markdown("#### Today's Threats")
        if not today_df.empty:
            donut_counts = today_df["Risk_Vector"].value_counts()
        else:
            donut_counts = pd.Series(0, index=RISK_VECTORS_ORDER)

        donut_counts = donut_counts.reindex(RISK_VECTORS_ORDER, fill_value=0)
        donut_colors = [VECTOR_COLORS[rv] for rv in donut_counts.index]

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=donut_counts.index,
                    values=donut_counts.values,
                    marker=dict(colors=donut_colors),
                    textinfo="label+percent+value",
                    hole=0.4,
                    sort=False,
                )
            ]
        )
        fig_donut.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            template=plotly_template,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(color="#f0f2f6" if dark_mode else "#1a1a1a"),
        )
        st.plotly_chart(fig_donut, use_container_width=True, theme=None)

    with col_bar:
        st.markdown("#### Daily Trend – Last 7 Days")
        
        end_day = pd.Timestamp.now().normalize()
        start_day = end_day - timedelta(days=6)
        full_days = pd.date_range(start=start_day, end=end_day, freq="D")
        
        if seven_days_df.empty:
            daily_counts = pd.DataFrame(0, index=full_days, columns=RISK_VECTORS_ORDER)
        else:
            seven_days_df["DayOnly"] = seven_days_df["Date"].dt.normalize()
            daily_counts = (
                seven_days_df.groupby(["DayOnly", "Risk_Vector"])
                .size()
                .unstack(fill_value=0)
            )
            daily_counts = daily_counts.reindex(full_days, fill_value=0)
        
        for rv in RISK_VECTORS_ORDER:
            if rv not in daily_counts.columns:
                daily_counts[rv] = 0
        daily_counts = daily_counts[RISK_VECTORS_ORDER].astype(int)
        
        x_day_labels = [d.strftime("%b %d") for d in daily_counts.index]

        fig_bar = go.Figure()
        for rv in RISK_VECTORS_ORDER:
            fig_bar.add_trace(
                go.Bar(
                    x=x_day_labels,
                    y=daily_counts[rv].tolist(),
                    name=rv,
                    marker_color=VECTOR_COLORS[rv],
                )
            )
        fig_bar.update_layout(
            barmode="stack",
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Last 7 Days", tickangle=-30),
            yaxis=dict(title="Headline Count"),
            template=plotly_template,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(color="#f0f2f6" if dark_mode else "#1a1a1a"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)

    with col_tables:
        st.markdown("#### Top Headlines (Last 7 Days)")
        has_any_threats = False

        for rv in RISK_VECTORS_ORDER:
            sub_rv = seven_days_df[seven_days_df["Risk_Vector"] == rv].head(3)
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
                "No recent risk-classified headlines found for this scope over the last 7 days."
            )

    # Footer note
    st.divider()
    st.caption(
        "Classifier is a lightweight demo model (TF-IDF + Logistic Regression). "
        "Expand the training corpus or replace with a stronger NLP model for production use. "
        "Data is automatically persisted to historical_news.csv."
    )
