"""
================================================================================
NETFLIX ANALYTICS DASHBOARD  |  Premium Dark Streamlit App
================================================================================
Run with:  streamlit run dashboard.py

A portfolio-quality, Netflix-branded interactive dashboard covering:
  - KPI summary cards
  - Sidebar filters (Type, Genre, Country, Release Year, Rating)
  - Content mix, genre, country, director/actor, rating, trend & duration views
  - A live TF-IDF content-based recommendation panel
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# NETFLIX-THEMED CSS
# ------------------------------------------------------------------
NETFLIX_RED = "#E50914"
NETFLIX_DARK = "#141414"
NETFLIX_DARK2 = "#1F1F1F"
NETFLIX_GRAY = "#B3B3B3"
NETFLIX_WHITE = "#F5F5F1"

st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, {NETFLIX_DARK} 0%, #0d0d0d 100%);
        color: {NETFLIX_WHITE};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {NETFLIX_DARK2};
        border-right: 1px solid #333;
    }}
    h1, h2, h3, h4, p, span, label, div {{
        color: {NETFLIX_WHITE};
    }}
    .netflix-title {{
        font-size: 42px;
        font-weight: 800;
        color: {NETFLIX_RED};
        letter-spacing: -1px;
        margin-bottom: 0px;
    }}
    .netflix-subtitle {{
        color: {NETFLIX_GRAY};
        font-size: 16px;
        margin-top: -8px;
    }}
    .kpi-card {{
        background: linear-gradient(145deg, #1f1f1f, #141414);
        border: 1px solid #333;
        border-left: 5px solid {NETFLIX_RED};
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.45);
    }}
    .kpi-value {{
        font-size: 30px;
        font-weight: 800;
        color: {NETFLIX_WHITE};
    }}
    .kpi-label {{
        font-size: 13px;
        color: {NETFLIX_GRAY};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid #333;
        border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {NETFLIX_DARK2};
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: {NETFLIX_GRAY};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {NETFLIX_RED} !important;
        color: white !important;
    }}
    hr {{ border-color: #333; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    plot_bgcolor=NETFLIX_DARK,
    paper_bgcolor=NETFLIX_DARK,
    font=dict(color=NETFLIX_WHITE, family="Helvetica"),
)
NETFLIX_SEQ = ["#E50914", "#B20710", "#831010", "#F5F5F1", "#564d4d", "#221f1f"]


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_featured_dataset.csv")
    df["date_added"] = pd.to_datetime(df["date_added"])
    df["content_soup"] = df["content_soup"].fillna("")
    return df


@st.cache_resource
def build_recommender(df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(df["content_soup"])
    title_index = pd.Series(df.index, index=df["title"].str.lower())
    return vectorizer, tfidf_matrix, title_index


def recommend(title, df, tfidf_matrix, title_index, top_n=10):
    title_lower = title.lower().strip()
    if title_lower not in title_index:
        close = get_close_matches(title_lower, title_index.index, n=1, cutoff=0.5)
        if not close:
            return None, None
        title_lower = close[0]
    idx = title_index[title_lower]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = [i for i in sims.argsort()[::-1] if i != idx][:top_n]
    result = df.loc[order, ["title", "type", "listed_in", "release_year", "rating"]].copy()
    result["match_%"] = np.round(sims[order] * 100, 1)
    return df.loc[idx, "title"], result.reset_index(drop=True)


df_full = load_data()

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown('<p class="netflix-title">🎬 NETFLIX ANALYTICS DASHBOARD</p>', unsafe_allow_html=True)
st.markdown('<p class="netflix-subtitle">Content Strategy & Catalogue Intelligence | Built with Streamlit</p>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------
st.sidebar.markdown("## 🔎 Filters")

type_filter = st.sidebar.multiselect("Content Type", options=sorted(df_full["type"].unique()),
                                       default=list(df_full["type"].unique()))

all_genres = sorted(df_full["listed_in"].str.split(", ").explode().unique())
genre_filter = st.sidebar.multiselect("Genre", options=all_genres, default=[])

all_countries = sorted([c for c in df_full["country"].str.split(", ").explode().unique() if c != "Not Specified"])
country_filter = st.sidebar.multiselect("Country", options=all_countries, default=[])

year_min, year_max = int(df_full["release_year"].min()), int(df_full["release_year"].max())
year_range = st.sidebar.slider("Release Year Range", year_min, year_max, (2010, year_max))

rating_filter = st.sidebar.multiselect("Rating", options=sorted(df_full["rating"].unique()), default=[])

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Senior Data Scientist Portfolio Project**")
st.sidebar.markdown("Data source: Netflix Titles Dataset")

# ------------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------------
df = df_full[
    df_full["type"].isin(type_filter) &
    df_full["release_year"].between(year_range[0], year_range[1])
]
if genre_filter:
    df = df[df["listed_in"].apply(lambda x: any(g in x for g in genre_filter))]
if country_filter:
    df = df[df["country"].apply(lambda x: any(c in x for c in country_filter))]
if rating_filter:
    df = df[df["rating"].isin(rating_filter)]

if df.empty:
    st.warning("No titles match the selected filters. Please broaden your selection.")
    st.stop()

# ------------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------------
total_titles = len(df)
total_movies = (df["type"] == "Movie").sum()
total_shows = (df["type"] == "TV Show").sum()
total_countries = df["country"].str.split(", ").explode().nunique()
avg_movie_len = df.loc[df["type"] == "Movie", "duration_value"].mean()
top_genre = df["listed_in"].str.split(", ").explode().value_counts().idxmax()

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "Total Titles", f"{total_titles:,}"),
    (k2, "Movies", f"{total_movies:,}"),
    (k3, "TV Shows", f"{total_shows:,}"),
    (k4, "Countries", f"{total_countries:,}"),
    (k5, "Avg Movie Runtime", f"{avg_movie_len:.0f} min" if not np.isnan(avg_movie_len) else "N/A"),
]
for col, label, value in kpis:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🌍 Genres & Countries", "🎭 People & Ratings",
     "📈 Trends & Duration", "🤖 Recommendation Engine"]
)

# --- TAB 1: OVERVIEW ---
with tab1:
    c1, c2 = st.columns([1, 1.4])

    with c1:
        type_counts = df["type"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index, values=type_counts.values, hole=0.55,
            marker=dict(colors=[NETFLIX_RED, NETFLIX_WHITE], line=dict(color=NETFLIX_DARK, width=2)),
            textfont=dict(color=NETFLIX_DARK, size=14)
        )])
        fig.update_layout(title="Movies vs TV Shows", **PLOTLY_TEMPLATE, height=400,
                            annotations=[dict(text=f"{total_titles:,}<br>Titles", x=0.5, y=0.5,
                                               font_size=16, showarrow=False, font_color=NETFLIX_WHITE)])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        mix = df.groupby(["release_decade", "type"]).size().reset_index(name="count")
        fig = px.bar(mix, x="release_decade", y="count", color="type", barmode="group",
                      color_discrete_sequence=[NETFLIX_RED, NETFLIX_WHITE],
                      title="Content Volume by Release Decade")
        fig.update_layout(**PLOTLY_TEMPLATE, height=400, legend_title="Type")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 💡 Business Insight")
    st.info(f"The catalogue is {total_movies/total_titles*100:.1f}% Movies and "
            f"{total_shows/total_titles*100:.1f}% TV Shows within the current filter selection. "
            "Monitoring this mix over time helps balance acquisition spend between one-off "
            "films and recurring series that drive longer-term engagement.")

# --- TAB 2: GENRES & COUNTRIES ---
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        genres = df["listed_in"].str.split(", ").explode().value_counts().head(12)
        fig = px.bar(genres[::-1], orientation="h", color=genres[::-1].values,
                      color_continuous_scale=["#564d4d", NETFLIX_RED],
                      title="Top 12 Genres", labels={"value": "Titles", "index": ""})
        fig.update_layout(**PLOTLY_TEMPLATE, height=460, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        countries = df["country"].str.split(", ").explode()
        countries = countries[countries != "Not Specified"].value_counts().head(12)
        fig = px.bar(countries[::-1], orientation="h", color=countries[::-1].values,
                      color_continuous_scale=["#564d4d", NETFLIX_RED],
                      title="Top 12 Countries", labels={"value": "Titles", "index": ""})
        fig.update_layout(**PLOTLY_TEMPLATE, height=460, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 💡 Business Insight")
    st.info(f"'{genres.index[-1]}' leads genre volume and '{countries.index[-1]}' leads country "
            "production within your filters -- useful for identifying where to double down "
            "on regional content deals or where whitespace exists for expansion.")

# --- TAB 3: PEOPLE & RATINGS ---
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        directors = df[df["director"] != "Not Specified"]["director"].str.split(", ").explode().value_counts().head(10)
        fig = px.bar(directors[::-1], orientation="h", color_discrete_sequence=[NETFLIX_RED],
                      title="Top 10 Directors", labels={"value": "Titles", "index": ""})
        fig.update_layout(**PLOTLY_TEMPLATE, height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ratings = df["rating"].value_counts()
        fig = px.bar(ratings, color_discrete_sequence=[NETFLIX_RED],
                      title="Content Rating Distribution", labels={"value": "Titles", "index": "Rating"})
        fig.update_layout(**PLOTLY_TEMPLATE, height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    maturity = df.groupby(["type", "maturity_tier"]).size().reset_index(name="count")
    fig = px.bar(maturity, x="type", y="count", color="maturity_tier", barmode="stack",
                  color_discrete_sequence=NETFLIX_SEQ, title="Maturity Tier Mix by Content Type")
    fig.update_layout(**PLOTLY_TEMPLATE, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 💡 Business Insight")
    st.info(f"'{ratings.idxmax()}' is the dominant rating in the current selection, and the "
            "maturity-tier split shows how skewed the catalogue is toward adult audiences -- "
            "informing decisions about kid-friendly content investment.")

# --- TAB 4: TRENDS & DURATION ---
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        trend = df.groupby(["release_year", "type"]).size().reset_index(name="count")
        fig = px.line(trend, x="release_year", y="count", color="type",
                        color_discrete_sequence=[NETFLIX_RED, NETFLIX_WHITE],
                        title="Release Year Trend", markers=True)
        fig.update_layout(**PLOTLY_TEMPLATE, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        added = df.groupby(df["date_added"].dt.to_period("M")).size()
        added.index = added.index.to_timestamp()
        fig = px.area(x=added.index, y=added.values, title="Content Added Over Time",
                        labels={"x": "Date Added", "y": "Titles Added"},
                        color_discrete_sequence=[NETFLIX_RED])
        fig.update_layout(**PLOTLY_TEMPLATE, height=420)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        movies_dur = df[df["type"] == "Movie"]["duration_value"].dropna()
        fig = px.histogram(movies_dur, nbins=30, color_discrete_sequence=[NETFLIX_RED],
                             title="Movie Duration Distribution (minutes)")
        fig.update_layout(**PLOTLY_TEMPLATE, height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        seasons = df[df["type"] == "TV Show"]["duration_value"].dropna().value_counts().sort_index()
        fig = px.bar(seasons, color_discrete_sequence=[NETFLIX_WHITE],
                      title="TV Show Season Count Distribution",
                      labels={"index": "Seasons", "value": "Titles"})
        fig.update_layout(**PLOTLY_TEMPLATE, height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 💡 Business Insight")
    st.info("Content additions and release-year trends both show Netflix's rapid catalogue "
            "growth phase followed by more selective curation -- while most TV Shows run "
            "only 1-2 seasons, reinforcing a portfolio strategy built on many bets rather "
            "than a few long-running flagship series.")

# --- TAB 5: RECOMMENDATION ENGINE ---
with tab5:
    st.markdown("### 🤖 Content-Based Recommendation Engine")
    st.caption("Powered by TF-IDF vectorization + Cosine Similarity across genre, description, director & cast.")

    vectorizer, tfidf_matrix, title_index = build_recommender(df_full)

    default_title = "Peaky Blinders" if "peaky blinders" in title_index.index else df_full["title"].iloc[0]
    query = st.text_input("Enter a Movie or TV Show title:", value=default_title)
    top_n = st.slider("Number of recommendations", 5, 15, 10)

    if st.button("🎯 Get Recommendations", type="primary") or query:
        matched_title, recs = recommend(query, df_full, tfidf_matrix, title_index, top_n=top_n)
        if recs is None:
            st.error(f"'{query}' not found in the catalogue. Please check the spelling.")
        else:
            if matched_title.lower() != query.lower().strip():
                st.caption(f"Showing results for closest match: **{matched_title}**")
            st.markdown(f"#### Because you watched **{matched_title}**...")
            st.dataframe(
                recs.rename(columns={
                    "title": "Title", "type": "Type", "listed_in": "Genres",
                    "release_year": "Year", "rating": "Rating", "match_%": "Match %"
                }),
                use_container_width=True, hide_index=True
            )

st.markdown("---")
st.caption("Netflix Analytics Dashboard | Portfolio Data Science Project | Data: Netflix Titles Dataset (Kaggle)")
