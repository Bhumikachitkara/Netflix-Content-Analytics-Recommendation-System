# Final Project Report: Netflix Movies & TV Shows Data Science Project

**Author:** Senior Data Scientist Portfolio Project
**Dataset:** Netflix Titles (public Kaggle metadata export)
**Deliverables:** Cleaned dataset, EDA suite, feature-engineered dataset, TF-IDF recommendation engine, interactive Streamlit dashboard

---

## 1. Executive Summary

This report documents an end-to-end data science project built on the Netflix Titles dataset (8,807 raw records / 8,797 after cleaning). The objective was to transform raw, messy catalogue metadata into actionable content-strategy insights and a working recommendation system, packaged as a portfolio-quality deliverable.

**Key outcomes:**
- Zero-null, analysis-ready cleaned dataset (`cleaned_netflix_dataset.csv`)
- 15 engineered features supporting both analytics and recommendation
- 13 professional, Netflix-branded visualizations, each with a written business insight
- A functioning TF-IDF + Cosine Similarity content-based recommender returning coherent Top-10 results
- A premium, interactive Streamlit dashboard with KPI cards, filters, and a live recommendation panel

---

## 2. Business Problem

Netflix's competitive advantage rests heavily on its content catalogue: what it acquires, produces, and how it's organized directly affects subscriber acquisition and retention. Key business questions this project addresses:

1. What is the current balance of Movies vs. TV Shows, and how has it shifted over time?
2. Which genres, countries, directors, and actors define the catalogue's identity?
3. How does the content skew across audience-maturity tiers?
4. Can a recommendation system be built purely from catalogue metadata (no user history), and how reliable are its suggestions?

---

## 3. Data Understanding

The raw dataset contains 12 columns: `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in` (genres), and `description`.

**Data quality issues identified:**
- `director`: 2,634 missing (~30%)
- `cast`: 825 missing (~9%)
- `country`: 831 missing (~9%)
- `date_added`: 10 missing
- `rating`: 4 missing
- `duration`: 3 missing
- `duration` mixes two units depending on `type` (minutes for Movies, Seasons for TV Shows)
- `date_added` stored as free-text (e.g. "September 25, 2021") rather than a proper date type

---

## 4. Data Cleaning Methodology

| Step | Action | Rationale |
|---|---|---|
| Duplicate removal | Dropped on `show_id` and (`title`,`type`,`director`,`country`) | Prevent double-counting in aggregations |
| Text standardization | Trimmed whitespace, normalized comma-separated lists, fixed rating typos (`UR`→`NR`, `TV-Y7-FV`→`TV-Y7`) | Ensures consistent groupby/explode behavior |
| Missing categorical values | Filled `director`/`cast`/`country` with `"Not Specified"` | Preserves ~30% of rows for genre/type analysis rather than discarding them; explicit label avoids silent bias |
| Missing `rating`/`duration` | Imputed with the mode per content `type` | Small volume (<0.1%), mode-per-type avoids cross-contaminating Movie/TV Show distributions |
| Missing `date_added` | Dropped (10 rows) | Cannot be safely imputed without fabricating trend signal for time-series analysis |
| Dtype fixes | Parsed `date_added` to datetime; split `duration` into `duration_value` (numeric) + `duration_unit` | Enables proper time-series and duration analysis |
| Inconsistent values | Extracted `primary_country`; restricted `type` to {Movie, TV Show} | Cleans edge cases and supports single-label aggregation |

**Result:** 8,797 rows × 15 columns, zero remaining missing values, saved to `cleaned_netflix_dataset.csv`.

---

## 5. Feature Engineering

| Feature | Description |
|---|---|
| `year_added`, `month_added`, `month_name_added`, `weekday_added` | Calendar breakdown of when content was added |
| `years_to_add` | Gap between original release and Netflix addition (licensing/production lag) |
| `primary_genre`, `genre_count` | Simplified single-label genre + multi-label count |
| `cast_count` | Proxy for production scale |
| `has_director_info` | Binary flag for director data availability |
| `is_international` | Flag for non-US or multi-country productions |
| `duration_minutes`, `num_seasons` | Unified numeric duration split by content type |
| `maturity_tier` | Kids / Family / Teens / Adults / Unrated, mapped from `rating` |
| `release_decade` | Decade bucket for long-run trend analysis |
| `content_soup` | Combined weighted text (genre×2 + description + director + top-5 cast) for the recommender |

---

## 6. Exploratory Data Analysis — Summary of Insights

1. **Content Mix:** ~70% Movies / 30% TV Shows.
2. **Top Genres:** International Movies, Dramas, and Comedies lead — reflects a global-content strategy.
3. **Top Countries:** United States, India, and United Kingdom dominate production volume.
4. **Top Directors/Actors:** A small set of prolific contributors (many from regional industries like Bollywood) account for disproportionate title counts.
5. **Ratings:** TV-MA is the most frequent rating; the catalogue skews toward mature/teen audiences.
6. **Release-Year Trend:** Sharp growth from 2015, peaking around 2018, then tapering — mirrors Netflix's global-expansion and subsequent curation phases.
7. **Content-Added Trend:** Monthly additions peak around 2019-2020 before slowing.
8. **Duration:** Average movie runtime ~100 minutes; ~67% of TV Shows are single-season.
9. **Correlation Heatmap:** Weak correlations among engineered numeric features — low multicollinearity risk for future modelling.
10. **Word Cloud:** "life," "family," "young," and "love" dominate descriptions, indicating a relationship/coming-of-age narrative focus.
11. **Seasonality:** Content additions peak in specific months, useful for aligning marketing campaigns.
12. **Maturity Mix:** Adult-rated content dominates both Movies and TV Shows; Kids' content is a minority share.

---

## 7. Recommendation System

**Approach:** Content-based filtering using TF-IDF vectorization (max 20,000 features, English stop words removed) over the `content_soup` feature, with Cosine Similarity to rank the most similar titles. Fuzzy title matching (via `difflib.get_close_matches`) handles minor spelling variations.

**Validation examples:**
- *"Peaky Blinders"* → British crime/drama TV Shows (Paranoid, Criminal: UK, Happy Valley, etc.)
- *"The Conjuring"* → Horror/thriller Movies (The Conjuring 2, Insidious, Creep, etc.)
- *"3 Idiots"* → Indian comedy-dramas (PK, Rang De Basanti, Taare Zameen Par, etc.)

These results demonstrate the recommender captures genuine thematic/genre coherence, and — critically — works without any user viewing history, making it ideal for cold-start scenarios.

---

## 8. Dashboard

A Streamlit dashboard (`dashboard.py`) delivers an interactive, Netflix-branded (black/red/white) experience:
- **KPI Cards:** Total Titles, Movies, TV Shows, Countries, Avg Movie Runtime
- **Filters:** Content Type, Genre, Country, Release Year range, Rating
- **Tabs:** Overview, Genres & Countries, People & Ratings, Trends & Duration, Recommendation Engine
- **Recommendation Panel:** Live TF-IDF-powered "Because you watched..." suggestions

---

## 9. Conclusion

This project successfully converted a raw, imperfect metadata export into a fully analysis-ready dataset, extracted a comprehensive set of business insights through visual analytics, engineered features tailored to both descriptive and predictive use cases, and delivered a working, explainable recommendation engine — all wrapped in a polished, interactive dashboard suitable for a professional portfolio.

## 10. Future Scope

- Incorporate real viewership and rating-quality data (IMDb/Rotten Tomatoes) for popularity-aware recommendations
- Build a hybrid recommender blending content-based and collaborative filtering signals
- Apply advanced NLP topic modelling (e.g., BERTopic, embeddings) to descriptions for richer thematic clustering
- Develop a content-gap detection model to flag under-served genre × country × maturity-tier combinations
- Deploy the recommendation engine as a real-time API for production integration
