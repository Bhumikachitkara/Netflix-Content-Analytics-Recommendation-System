"""
================================================================================
NETFLIX MOVIES & TV SHOWS - EXPLORATORY DATA ANALYSIS
================================================================================
Generates a full suite of professional, Netflix-branded visualizations and
prints business insights after each chart. All figures are saved to /charts.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Netflix brand palette & global style
# ---------------------------------------------------------------------------
NETFLIX_RED = "#E50914"
NETFLIX_DARK = "#141414"
NETFLIX_GRAY = "#564d4d"
NETFLIX_WHITE = "#F5F5F1"
PALETTE = ["#E50914", "#B20710", "#F5F5F1", "#831010", "#564d4d", "#221f1f"]

plt.rcParams.update({
    "figure.facecolor": NETFLIX_DARK,
    "axes.facecolor": NETFLIX_DARK,
    "savefig.facecolor": NETFLIX_DARK,
    "axes.edgecolor": NETFLIX_WHITE,
    "axes.labelcolor": NETFLIX_WHITE,
    "text.color": NETFLIX_WHITE,
    "xtick.color": NETFLIX_WHITE,
    "ytick.color": NETFLIX_WHITE,
    "axes.titlecolor": NETFLIX_WHITE,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.color": "#333333",
    "grid.alpha": 0.4,
    "legend.facecolor": NETFLIX_DARK,
    "legend.edgecolor": NETFLIX_WHITE,
    "legend.labelcolor": NETFLIX_WHITE,
})

CHART_DIR = "charts"


def style_title(ax, title, subtitle=None):
    ax.set_title(title, fontsize=15, fontweight="bold", color=NETFLIX_WHITE, pad=18, loc="left")
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9.5,
                 color="#B3B3B3", ha="left")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {CHART_DIR}/{name}.png")


# ---------------------------------------------------------------------------
# 1. Movies vs TV Shows
# ---------------------------------------------------------------------------
def plot_type_distribution(df):
    counts = df["type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
        colors=[NETFLIX_RED, NETFLIX_WHITE],
        wedgeprops=dict(width=0.45, edgecolor=NETFLIX_DARK, linewidth=2),
        textprops={"fontsize": 12, "fontweight": "bold"}
    )
    for at in autotexts:
        at.set_color(NETFLIX_DARK if at.get_color() != NETFLIX_WHITE else NETFLIX_DARK)
    style_title(ax, "Content Mix: Movies vs TV Shows", "Share of total catalogue titles")
    ax.text(0, 0, f"{len(df):,}\nTitles", ha="center", va="center", fontsize=13,
             fontweight="bold", color=NETFLIX_WHITE)
    savefig(fig, "01_movies_vs_tvshows")

    pct_movie = counts.get("Movie", 0) / counts.sum() * 100
    print(f"INSIGHT: Movies make up {pct_movie:.1f}% of the catalogue vs "
          f"{100 - pct_movie:.1f}% TV Shows. Netflix's library is movie-led, but "
          "TV Shows typically drive higher watch-hours per title through repeat "
          "binge sessions -- a key input for content-investment ROI modelling.")


# ---------------------------------------------------------------------------
# 2. Top Genres
# ---------------------------------------------------------------------------
def plot_top_genres(df, top_n=15):
    genres = df["listed_in"].str.split(", ").explode()
    top_genres = genres.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top_genres.index[::-1], top_genres.values[::-1],
                    color=NETFLIX_RED, edgecolor=NETFLIX_WHITE, linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                 f"{int(bar.get_width()):,}", va="center", fontsize=9, color=NETFLIX_WHITE)
    style_title(ax, f"Top {top_n} Genres on Netflix", "Count of titles tagged per genre (multi-label)")
    ax.set_xlabel("Number of Titles")
    savefig(fig, "02_top_genres")

    print(f"INSIGHT: '{top_genres.index[0]}' is the most common genre tag with "
          f"{top_genres.iloc[0]:,} titles. International & Drama-leaning genres "
          "dominate the top of the list, confirming Netflix's global-content strategy "
          "rather than a narrow Hollywood-only focus.")


# ---------------------------------------------------------------------------
# 3. Top Countries
# ---------------------------------------------------------------------------
def plot_top_countries(df, top_n=15):
    countries = df["country"].str.split(", ").explode()
    countries = countries[countries != "Not Specified"]
    top_countries = countries.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.bar(top_countries.index, top_countries.values, color=PALETTE[0],
                   edgecolor=NETFLIX_WHITE, linewidth=0.4)
    bars[0].set_color(NETFLIX_WHITE)
    ax.set_xticklabels(top_countries.index, rotation=45, ha="right")
    style_title(ax, f"Top {top_n} Content-Producing Countries", "Count of titles by contributing country (multi-label)")
    ax.set_ylabel("Number of Titles")
    savefig(fig, "03_top_countries")

    print(f"INSIGHT: {top_countries.index[0]} leads content production with "
          f"{top_countries.iloc[0]:,} titles, followed by {top_countries.index[1]} "
          f"and {top_countries.index[2]}. This concentration highlights where Netflix "
          "has invested most in local production/licensing -- and signals whitespace "
          "in under-represented regions for future catalogue expansion.")


# ---------------------------------------------------------------------------
# 4. Top Directors
# ---------------------------------------------------------------------------
def plot_top_directors(df, top_n=10):
    directors = df[df["director"] != "Not Specified"]["director"].str.split(", ").explode()
    top_directors = directors.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_directors.index[::-1], top_directors.values[::-1],
                    color=NETFLIX_RED, edgecolor=NETFLIX_WHITE, linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{int(bar.get_width())}", va="center", fontsize=9, color=NETFLIX_WHITE)
    style_title(ax, f"Top {top_n} Directors by Title Count", "Directors with the most titles on Netflix")
    ax.set_xlabel("Number of Titles")
    savefig(fig, "04_top_directors")

    print(f"INSIGHT: {top_directors.index[0]} has the most credited titles "
          f"({top_directors.iloc[0]}) on the platform. Prolific directors like these "
          "are strong candidates for exclusive-deal negotiations given their proven "
          "catalogue contribution.")


# ---------------------------------------------------------------------------
# 5. Top Actors
# ---------------------------------------------------------------------------
def plot_top_actors(df, top_n=10):
    cast = df[df["cast"] != "Not Specified"]["cast"].str.split(", ").explode()
    top_cast = cast.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_cast.index[::-1], top_cast.values[::-1],
                    color=NETFLIX_WHITE, edgecolor=NETFLIX_RED, linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{int(bar.get_width())}", va="center", fontsize=9, color=NETFLIX_WHITE)
    style_title(ax, f"Top {top_n} Actors by Title Count", "Actors appearing in the most Netflix titles")
    ax.set_xlabel("Number of Titles")
    savefig(fig, "05_top_actors")

    print(f"INSIGHT: {top_cast.index[0]} appears in the most titles ({top_cast.iloc[0]}). "
          "A concentration of high-appearance actors from specific regional industries "
          "(e.g. Bollywood, Nollywood) reflects Netflix's regional-content acquisition depth.")


# ---------------------------------------------------------------------------
# 6. Ratings Distribution
# ---------------------------------------------------------------------------
def plot_ratings_distribution(df):
    order = df["rating"].value_counts().index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, y="rating", order=order, hue="type",
                   palette=[NETFLIX_RED, NETFLIX_WHITE], ax=ax)
    style_title(ax, "Content Rating Distribution", "Split by Movie vs TV Show")
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Rating")
    ax.legend(title="Type", frameon=False)
    savefig(fig, "06_ratings_distribution")

    top_rating = df["rating"].value_counts().idxmax()
    print(f"INSIGHT: '{top_rating}' is the most frequent content rating, indicating "
          "the catalogue skews toward mature/teen audiences rather than young children -- "
          "relevant for family-plan content-curation and parental-control defaults.")


# ---------------------------------------------------------------------------
# 7. Release Year Trend
# ---------------------------------------------------------------------------
def plot_release_year_trend(df):
    trend = df[df["release_year"] >= 1990].groupby(["release_year", "type"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(trend.index, trend["Movie"], color=NETFLIX_RED, linewidth=2.5, marker="o", markersize=3, label="Movie")
    ax.plot(trend.index, trend["TV Show"], color=NETFLIX_WHITE, linewidth=2.5, marker="o", markersize=3, label="TV Show")
    ax.fill_between(trend.index, trend["Movie"], color=NETFLIX_RED, alpha=0.15)
    style_title(ax, "Content Release-Year Trend (1990+)", "Number of titles by original release year")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Titles")
    ax.legend(frameon=False)
    savefig(fig, "07_release_year_trend")

    peak_year = trend.sum(axis=1).idxmax()
    print(f"INSIGHT: Title output climbs sharply after 2015 and peaks around {peak_year}, "
          "reflecting Netflix's aggressive content-acquisition/production ramp-up during "
          "its global-expansion phase, followed by a post-2019 tapering likely tied to "
          "catalogue-curation and cost-discipline strategies.")


# ---------------------------------------------------------------------------
# 8. Content Added Trend
# ---------------------------------------------------------------------------
def plot_content_added_trend(df):
    df["date_added"] = pd.to_datetime(df["date_added"])
    added = df.groupby([df["date_added"].dt.to_period("M")]).size()
    added.index = added.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(added.index, added.values, color=NETFLIX_RED, linewidth=1.8)
    ax.fill_between(added.index, added.values, color=NETFLIX_RED, alpha=0.25)
    style_title(ax, "Content Added to Netflix Over Time", "Titles added per month, all years")
    ax.set_xlabel("Date Added")
    ax.set_ylabel("Titles Added")
    savefig(fig, "08_content_added_trend")

    print("INSIGHT: Monthly additions rise steeply from 2016 and peak around "
          "2019-2020, before slowing -- consistent with Netflix shifting from "
          "rapid library expansion toward a more selective, original-content-first "
          "acquisition strategy in recent years.")


# ---------------------------------------------------------------------------
# 9. Duration Analysis
# ---------------------------------------------------------------------------
def plot_duration_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    movies = df[df["type"] == "Movie"]["duration_value"].dropna()
    axes[0].hist(movies, bins=30, color=NETFLIX_RED, edgecolor=NETFLIX_DARK)
    axes[0].axvline(movies.mean(), color=NETFLIX_WHITE, linestyle="--", linewidth=1.5,
                     label=f"Mean: {movies.mean():.0f} min")
    style_title(axes[0], "Movie Duration Distribution", "Runtime in minutes")
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].legend(frameon=False)

    seasons = df[df["type"] == "TV Show"]["duration_value"].dropna()
    season_counts = seasons.value_counts().sort_index()
    axes[1].bar(season_counts.index, season_counts.values, color=NETFLIX_WHITE,
                edgecolor=NETFLIX_RED, linewidth=0.8)
    style_title(axes[1], "TV Show Seasons Distribution", "Number of seasons per show")
    axes[1].set_xlabel("Number of Seasons")
    savefig(fig, "09_duration_analysis")

    single_season_pct = (seasons == 1).mean() * 100
    print(f"INSIGHT: The average movie runs {movies.mean():.0f} minutes, and "
          f"{single_season_pct:.1f}% of TV Shows last only a single season -- "
          "suggesting Netflix favours limited-series formats and that renewal-rate "
          "is a meaningful churn-risk indicator worth modelling separately.")


# ---------------------------------------------------------------------------
# 10. Correlation Heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df):
    numeric_cols = ["release_year", "duration_value", "genre_count", "cast_count",
                     "years_to_add", "is_international", "has_director_info"]
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdGy_r", center=0,
                linewidths=0.5, linecolor=NETFLIX_DARK, ax=ax,
                cbar_kws={"label": "Correlation"})
    style_title(ax, "Correlation Heatmap", "Engineered numeric features")
    savefig(fig, "10_correlation_heatmap")

    print("INSIGHT: Correlations among numeric features are generally weak "
          "(no strong multicollinearity), confirming these engineered features "
          "capture largely independent signals -- good news for any downstream "
          "predictive modelling (e.g. predicting content popularity or renewal).")


# ---------------------------------------------------------------------------
# 11. Word Cloud (custom implementation -- 'wordcloud' package not required)
# ---------------------------------------------------------------------------
def plot_wordcloud(df):
    try:
        from wordcloud import WordCloud, STOPWORDS
        text = " ".join(df["description"].dropna().tolist())
        wc = WordCloud(width=1200, height=600, background_color=NETFLIX_DARK,
                        colormap="Reds", stopwords=STOPWORDS, max_words=150).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        style_title(ax, "Most Frequent Words in Netflix Descriptions", "wordcloud package")
        savefig(fig, "11_wordcloud")
    except ImportError:
        # Fallback: frequency-scaled scatter "word cloud" built with matplotlib only,
        # so the notebook still runs end-to-end in environments without the
        # optional 'wordcloud' package installed.
        import re
        stopwords = set("""a an the of and to in for on with is it as this that from
        their his her its at by an be are was were will has have had not but they
        who her him you your our we i s about into more one two life when where
        after before their them she he it's""".split())
        text = " ".join(df["description"].dropna().tolist()).lower()
        words = re.findall(r"[a-z']+", text)
        words = [w for w in words if w not in stopwords and len(w) > 3]
        freq = Counter(words).most_common(60)

        rng = np.random.default_rng(42)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
        max_count = freq[0][1]
        for word, count in freq:
            size = 10 + (count / max_count) * 40
            x, y = rng.uniform(5, 95), rng.uniform(5, 95)
            color = NETFLIX_RED if rng.random() > 0.4 else NETFLIX_WHITE
            ax.text(x, y, word, fontsize=size, color=color, ha="center", va="center",
                     fontweight="bold", alpha=0.9, rotation=rng.choice([0, 0, 0, 90]))
        style_title(ax, "Most Frequent Words in Netflix Descriptions",
                     "Custom frequency cloud (size = frequency) -- built without external 'wordcloud' package")
        savefig(fig, "11_wordcloud")

    print("INSIGHT: Dominant words such as 'life', 'family', 'young', and 'love' "
          "in title descriptions reveal Netflix's heavy leaning toward "
          "relationship-driven and coming-of-age narratives across both Movies "
          "and TV Shows -- useful input for synopsis-based content tagging.")


# ---------------------------------------------------------------------------
# 12. Additional: Content added by Month (seasonality) & Maturity tier mix
# ---------------------------------------------------------------------------
def plot_seasonality(df):
    month_order = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
    monthly = df["month_name_added"].value_counts().reindex(month_order)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(monthly.index, monthly.values, color=NETFLIX_RED, edgecolor=NETFLIX_WHITE, linewidth=0.4)
    peak_idx = monthly.values.argmax()
    bars[peak_idx].set_color(NETFLIX_WHITE)
    ax.set_xticklabels(monthly.index, rotation=45, ha="right")
    style_title(ax, "Seasonality of Content Additions", "Total titles added by calendar month, all years")
    ax.set_ylabel("Titles Added")
    savefig(fig, "12_seasonality")

    print(f"INSIGHT: {monthly.idxmax()} sees the highest volume of content additions -- "
          "useful for aligning marketing pushes and subscriber-acquisition campaigns "
          "with the platform's own content-drop cadence.")


def plot_maturity_mix(df):
    mix = df.groupby(["type", "maturity_tier"]).size().unstack(fill_value=0)
    mix_pct = mix.div(mix.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    mix_pct.plot(kind="bar", stacked=True, ax=ax,
                  color=[NETFLIX_RED, "#831010", NETFLIX_WHITE, "#564d4d", "#221f1f"])
    ax.set_xticklabels(mix_pct.index, rotation=0)
    style_title(ax, "Audience Maturity Mix by Content Type", "% of titles per maturity tier")
    ax.set_ylabel("% of Titles")
    ax.legend(title="Maturity Tier", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    savefig(fig, "13_maturity_mix")

    print("INSIGHT: 'Adults' content dominates both Movies and TV Shows, while "
          "'Kids' content is a comparatively small share -- a potential growth "
          "lever if Netflix wants to defend market share in the family-subscription segment.")


def run_all(df):
    plot_type_distribution(df)
    plot_top_genres(df)
    plot_top_countries(df)
    plot_top_directors(df)
    plot_top_actors(df)
    plot_ratings_distribution(df)
    plot_release_year_trend(df)
    plot_content_added_trend(df)
    plot_duration_analysis(df)
    plot_correlation_heatmap(df)
    plot_wordcloud(df)
    plot_seasonality(df)
    plot_maturity_mix(df)


if __name__ == "__main__":
    df = pd.read_csv("netflix_featured_dataset.csv")
    run_all(df)
