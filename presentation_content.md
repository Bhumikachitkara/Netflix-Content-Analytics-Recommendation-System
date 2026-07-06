# Netflix Data Analysis — Presentation Content (12 Slides)

Use this content directly in PowerPoint/Google Slides/Claude for PowerPoint.

---

## Slide 1 — Title Slide
**Netflix Movies & TV Shows: A Data-Driven Content Strategy Analysis**
Subtitle: End-to-End Data Science Project — Cleaning · EDA · Recommendation Engine · Dashboard
Presenter: [Your Name] | Portfolio Data Science Project

---

## Slide 2 — Business Problem
- Netflix's catalogue spans thousands of titles across genres, countries, and formats
- Understanding catalogue composition drives: content investment, subscriber retention, regional expansion, personalization
- **Goal:** Analyze the Netflix Titles dataset to extract actionable content-strategy insights and build a working recommendation engine

---

## Slide 3 — Dataset Overview
- ~8,800 titles (Movies & TV Shows), 12 raw attributes
- Key fields: title, type, director, cast, country, date_added, release_year, rating, duration, genres, description
- Data quality challenges: missing director/cast/country (up to 30%), inconsistent rating codes, mixed duration units

---

## Slide 4 — Data Cleaning Approach
- Removed duplicates via `show_id` and business-key matching
- Preserved missing categorical fields as `"Not Specified"` rather than dropping rows
- Dropped 10 rows with missing `date_added` (unsafe to impute for trend analysis)
- Standardized rating codes, split `duration` into value + unit
- Result: 8,797 clean rows, zero remaining nulls

---

## Slide 5 — Feature Engineering
- Date features: year/month/weekday added, licensing lag (`years_to_add`)
- Content features: primary genre, genre count, cast size, international flag
- `maturity_tier`: Kids / Family / Teens / Adults / Unrated
- `content_soup`: combined text feature powering the recommendation engine

---

## Slide 6 — Content Mix: Movies vs TV Shows
- ~70% Movies, ~30% TV Shows
- Movie-led library, but TV Shows likely drive disproportionate watch-hours
- **Insight:** Title count ≠ engagement value — balance acquisition spend accordingly

---

## Slide 7 — Genre & Country Landscape
- Top genres: International Movies, Dramas, Comedies — confirms global-first strategy
- Top countries: United States, India, United Kingdom
- **Insight:** Whitespace exists in under-represented high-growth media markets

---

## Slide 8 — Ratings & Audience Maturity
- TV-MA is the most common rating — catalogue skews toward mature/teen audiences
- Kids' content is comparatively under-represented
- **Insight:** Potential growth lever for defending family-subscription market share

---

## Slide 9 — Trends Over Time
- Release-year and content-added trends both show sharp growth 2015–2019
- Post-2019 tapering signals a shift toward selective, original-content-first curation
- Most TV Shows run only 1–2 seasons — a "many bets" portfolio strategy

---

## Slide 10 — Recommendation Engine
- Content-based approach: TF-IDF + Cosine Similarity
- Input features: genre (weighted), description, director, cast
- Example: "Peaky Blinders" → British crime dramas (Paranoid, Criminal: UK, Happy Valley...)
- Works even for brand-new titles with zero viewing history (cold-start friendly)

---

## Slide 11 — Interactive Dashboard
- Netflix-branded dark UI (black/red/white) built with Streamlit
- KPI cards, dynamic filters (type, genre, country, year, rating)
- 5 tabbed views including a live recommendation panel
- Portfolio-ready, deployable to Streamlit Community Cloud

---

## Slide 12 — Conclusion & Future Scope
**Conclusion:** Delivered a full production-style pipeline — cleaning, EDA, feature engineering, recommender, dashboard — all from raw metadata.

**Future Scope:**
- Integrate real viewership/ratings data
- Hybrid recommender (content-based + collaborative filtering)
- NLP topic modelling on descriptions
- Content-gap detection for acquisition strategy
- Deploy as a real-time recommendation API

**Thank you — Questions?**
