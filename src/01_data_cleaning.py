"""
================================================================================
NETFLIX MOVIES & TV SHOWS - DATA CLEANING MODULE
================================================================================
Author : Senior Data Scientist
Purpose: Load the raw Netflix catalogue export, resolve data-quality issues,
         and persist an analysis-ready dataset.

Business Problem
-----------------
Netflix's content catalogue metadata (titles, cast, country, genres, ratings,
duration, etc.) is scraped/exported from an internal system and contains the
usual real-world messiness: missing directors/cast/country, inconsistent
date formats, mixed duration units for Movies vs TV Shows, and free-text
genre tags. Before this data can power analytics (content strategy, regional
programming decisions) or a recommendation engine, it must be cleaned and
standardised.
================================================================================
"""

import pandas as pd
import numpy as np
import re

RAW_PATH = "data/netflix_titles.csv"
CLEAN_PATH = "cleaned_netflix_dataset.csv"

pd.set_option("display.max_columns", None)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Raw dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # Duplicate check on the natural business key (title + type + director + country)
    df = df.drop_duplicates(subset=["title", "type", "director", "country"], keep="first")
    df = df.drop_duplicates(subset=["show_id"], keep="first")
    print(f"Duplicates removed: {before - len(df):,} rows")
    return df


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = ["type", "title", "director", "cast", "country", "rating", "listed_in", "description"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "": np.nan, "None": np.nan})

    # Title casing for consistency where useful (director / cast / country names)
    df["country"] = df["country"].str.replace(r"\s*,\s*", ", ", regex=True)
    df["director"] = df["director"].str.replace(r"\s*,\s*", ", ", regex=True)
    df["cast"] = df["cast"].str.replace(r"\s*,\s*", ", ", regex=True)
    df["listed_in"] = df["listed_in"].str.replace(r"\s*,\s*", ", ", regex=True)

    # Rating typos / inconsistent codes seen in the public Netflix dataset
    rating_fix = {"UR": "NR", "TV-Y7-FV": "TV-Y7"}
    df["rating"] = df["rating"].replace(rating_fix)

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Categorical text fields: explicit "Unknown" / "Not Specified" placeholders
    # (kept distinct from NaN so aggregation logic later can decide how to treat them)
    df["director"] = df["director"].fillna("Not Specified")
    df["cast"] = df["cast"].fillna("Not Specified")
    df["country"] = df["country"].fillna("Not Specified")

    # A handful of rows (4) are missing 'rating' but have a duration value that
    # accidentally landed in the rating column upstream -- fix, then fill remainder
    # with the mode of that content type.
    mask_shift = df["rating"].isna() & df["duration"].isna()
    if mask_shift.any():
        df.loc[mask_shift, "rating"] = np.nan

    for content_type in df["type"].unique():
        type_mode = df.loc[df["type"] == content_type, "rating"].mode()
        if len(type_mode):
            df.loc[(df["type"] == content_type) & (df["rating"].isna()), "rating"] = type_mode[0]

    # date_added: 10 missing -> cannot be safely imputed, drop those rows since
    # "content added" trend analysis depends on this field and imputing dates
    # would fabricate trend signal.
    before = len(df)
    df = df.dropna(subset=["date_added"])
    print(f"Rows dropped due to missing 'date_added': {before - len(df):,}")

    # duration: 3 missing (all TV Shows in the public dataset) -> impute with
    # the mode duration for that content type.
    for content_type in df["type"].unique():
        type_mode = df.loc[df["type"] == content_type, "duration"].mode()
        if len(type_mode):
            df.loc[(df["type"] == content_type) & (df["duration"].isna()), "duration"] = type_mode[0]

    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")
    df["release_year"] = df["release_year"].astype(int)

    # Split duration into numeric value + unit (min for Movies, Seasons for TV Shows)
    df["duration_value"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df["duration_unit"] = df["duration"].apply(
        lambda x: "Season(s)" if "Season" in str(x) else ("min" if "min" in str(x) else np.nan)
    )
    return df


def fix_inconsistent_values(df: pd.DataFrame) -> pd.DataFrame:
    # Some rows have the primary country listed first among multiple -- keep
    # a clean 'primary_country' feature for country-level aggregation while
    # preserving the full multi-country string in 'country'.
    df["primary_country"] = df["country"].apply(lambda x: x.split(",")[0].strip())

    # listed_in -> normalise duplicate genre spellings
    df["listed_in"] = df["listed_in"].str.replace("TV Shows", "TV Shows", regex=False)

    # type should only ever be Movie / TV Show
    df = df[df["type"].isin(["Movie", "TV Show"])]

    return df


def clean_pipeline(path: str = RAW_PATH) -> pd.DataFrame:
    df = load_data(path)
    df = drop_duplicates(df)
    df = standardize_text_columns(df)
    df = handle_missing_values(df)
    df = fix_dtypes(df)
    df = fix_inconsistent_values(df)

    df = df.reset_index(drop=True)
    print(f"\nFinal cleaned dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Remaining nulls:\n{df.isna().sum()[df.isna().sum() > 0]}")
    return df


if __name__ == "__main__":
    cleaned = clean_pipeline()
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_PATH}")
