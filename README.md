# 🎬 Netflix Movies & TV Shows — Data Science Portfolio Project

A complete, end-to-end data science project on the Netflix Titles dataset: data cleaning, exploratory data analysis, feature engineering, a TF-IDF content-based recommendation engine, and a premium Netflix-themed interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Business Problem

Netflix's catalogue metadata is critical for **content strategy, regional expansion, subscriber retention, and personalization**. This project analyzes ~8,800 titles to answer:

- What's the mix of Movies vs TV Shows, and how has it evolved?
- Which genres, countries, directors, and actors dominate the catalogue?
- How mature/family-friendly is the content mix?
- Can we build a lightweight, explainable recommendation engine from metadata alone?

---

## 🗂️ Project Structure

```
netflix_project/
├── Netflix_Project.ipynb          # Full analysis notebook (cleaning → EDA → FE → recommender)
├── cleaned_netflix_dataset.csv    # Cleaned, analysis-ready dataset
├── netflix_featured_dataset.csv   # Cleaned dataset + engineered features
├── dashboard.py                   # Streamlit interactive dashboard (Netflix-themed)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── presentation_content.md        # 12-slide presentation content
├── Final_Project_Report.md        # Full written project report
├── data/
│   └── netflix_titles.csv         # Raw source dataset
├── charts/                        # Exported static chart images (PNG)
└── src/                           # Modular pipeline scripts
    ├── 01_data_cleaning.py
    ├── 02_feature_engineering.py
    ├── 03_eda.py
    └── 04_recommendation_system.py
```

---
---

## 🧹 Data Cleaning Highlights

| Issue | Resolution |
|---|---|
| Missing `director` / `cast` / `country` (up to 30%) | Filled with explicit `"Not Specified"` label to preserve rows for genre/country analysis |
| Missing `date_added` (10 rows) | Dropped — cannot be safely imputed for time-series trend analysis |
| Missing `rating` / `duration` (few rows) | Imputed using the mode per content type |
| Inconsistent rating codes (`UR`, `TV-Y7-FV`) | Standardized to correct MPAA/TV codes |
| Mixed `duration` units (min vs Seasons) | Split into `duration_value` + `duration_unit` |
| Duplicate records | Removed via `show_id` and business-key dedup |

---

## 🛠️ Feature Engineering

- `year_added`, `month_added`, `weekday_added`, `years_to_add` (licensing lag)
- `primary_genre`, `genre_count`
- `cast_count`, `has_director_info`, `is_international`
- `duration_minutes` / `num_seasons` (unified duration)
- `maturity_tier` (Kids / Family / Teens / Adults / Unrated)
- `release_decade`
- `content_soup` — combined text feature (genre + description + director + cast) for the recommender

---

## 📊 Exploratory Data Analysis

13 professional, Netflix-branded charts covering: content mix, top genres, top countries, top directors, top actors, ratings distribution, release-year trend, content-added trend, duration analysis, correlation heatmap, word cloud, seasonality, and audience maturity mix — each paired with a written business insight.

---

## 🤖 Recommendation System

A **content-based recommender** using **TF-IDF vectorization + Cosine Similarity** over a combined text feature (genre, description, director, cast). Enter any title and receive the Top-10 most similar titles — including fuzzy-matching for slightly misspelled titles.

```python
recommender.recommend("Peaky Blinders", top_n=10)
```

---

## 📈 Dashboard

A premium, dark, Netflix-branded Streamlit dashboard featuring:
- KPI summary cards (Total Titles, Movies, TV Shows, Countries, Avg Runtime)
- Sidebar filters: Type, Genre, Country, Release Year range, Rating
- Tabbed views: Overview · Genres & Countries · People & Ratings · Trends & Duration · Recommendation Engine
- Live TF-IDF recommendation panel

---

## 🔮 Future Scope

- Integrate real viewership/ratings data for quality-weighted recommendations
- Hybrid recommender combining content-based + collaborative filtering
- NLP topic modelling on descriptions (e.g. BERTopic)
- Content-gap detection model for acquisition strategy
- Production deployment as a real-time recommendation API

---

## 📄 License

This project is released under the MIT License. Dataset sourced from the publicly available Netflix Titles dataset (Kaggle).
