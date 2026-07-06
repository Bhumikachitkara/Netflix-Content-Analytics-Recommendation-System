"""
================================================================================
NETFLIX MOVIES & TV SHOWS - FEATURE ENGINEERING MODULE
================================================================================
Creates derived features that power both the EDA narrative and the
recommendation engine.
================================================================================
"""

import pandas as pd
import numpy as np

CLEAN_PATH = "cleaned_netflix_dataset.csv"
FEATURED_PATH = "netflix_featured_dataset.csv"


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date_added"] = pd.to_datetime(df["date_added"])
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month
    df["month_name_added"] = df["date_added"].dt.month_name()
    df["day_added"] = df["date_added"].dt.day
    df["weekday_added"] = df["date_added"].dt.day_name()

    # Content "freshness" gap: years between theatrical/original release and
    # the date it actually landed on Netflix. Useful for content-acquisition
    # strategy (licensed catch-up content vs. day-and-date originals).
    df["years_to_add"] = df["year_added"] - df["release_year"]
    df["years_to_add"] = df["years_to_add"].clip(lower=0)
    return df


def add_content_features(df: pd.DataFrame) -> pd.DataFrame:
    # Primary/first genre -- handy for single-label groupbys and charts
    df["primary_genre"] = df["listed_in"].apply(lambda x: x.split(",")[0].strip())
    df["genre_count"] = df["listed_in"].apply(lambda x: len(str(x).split(",")))

    # Cast size (proxy for production scale) -- 0 when "Not Specified"
    df["cast_count"] = df["cast"].apply(
        lambda x: 0 if x == "Not Specified" else len(str(x).split(","))
    )
    df["has_director_info"] = (df["director"] != "Not Specified").astype(int)
    df["is_international"] = df["country"].apply(
        lambda x: 1 if ("," in str(x) or (x != "Not Specified" and "United States" not in str(x))) else 0
    )

    # Movie runtime in minutes / TV Show season count as one unified numeric feature
    df["duration_minutes"] = np.where(df["duration_unit"] == "min", df["duration_value"], np.nan)
    df["num_seasons"] = np.where(df["duration_unit"] == "Season(s)", df["duration_value"], np.nan)

    # Simple content "maturity tier" grouping for audience-suitability analysis
    maturity_map = {
        "G": "Kids", "TV-Y": "Kids", "TV-Y7": "Kids", "TV-G": "Kids",
        "PG": "Family", "TV-PG": "Family",
        "PG-13": "Teens", "TV-14": "Teens",
        "R": "Adults", "TV-MA": "Adults", "NC-17": "Adults", "NR": "Unrated",
    }
    df["maturity_tier"] = df["rating"].map(maturity_map).fillna("Unrated")

    # Decade bucket for long-run release-year trend analysis
    df["release_decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"

    return df


def add_text_feature_for_recommender(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the most semantically useful text fields into a single 'soup'
    that the TF-IDF vectorizer will consume for the recommendation engine."""
    def soup(row):
        parts = [
            str(row["listed_in"]) * 2,   # weight genre higher
            str(row["description"]),
            str(row["director"]) if row["director"] != "Not Specified" else "",
            " ".join(str(row["cast"]).split(",")[:5]) if row["cast"] != "Not Specified" else "",
        ]
        return " ".join(parts).lower()

    df["content_soup"] = df.apply(soup, axis=1)
    return df


def engineer_features(path: str = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_date_features(df)
    df = add_content_features(df)
    df = add_text_feature_for_recommender(df)
    print(f"Feature engineering complete: {df.shape[1]} total columns")
    new_cols = [
        "year_added", "month_added", "month_name_added", "weekday_added", "years_to_add",
        "primary_genre", "genre_count", "cast_count", "has_director_info", "is_international",
        "duration_minutes", "num_seasons", "maturity_tier", "release_decade", "content_soup",
    ]
    print("New engineered features:", new_cols)
    return df


if __name__ == "__main__":
    featured = engineer_features()
    featured.to_csv(FEATURED_PATH, index=False)
    print(f"Saved feature-engineered dataset -> {FEATURED_PATH}")
