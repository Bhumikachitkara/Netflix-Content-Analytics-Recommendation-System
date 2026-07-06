"""
================================================================================
NETFLIX RECOMMENDATION SYSTEM - CONTENT-BASED (TF-IDF + COSINE SIMILARITY)
================================================================================
Given a title, recommend the Top-10 most similar titles based on genre,
description, director and cast text similarity.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


class NetflixRecommender:
    def __init__(self, df: pd.DataFrame, text_col: str = "content_soup"):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self._title_index = pd.Series(self.df.index, index=self.df["title"].str.lower())
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df[text_col])
        print(f"TF-IDF matrix built: {self.tfidf_matrix.shape[0]:,} titles x "
              f"{self.tfidf_matrix.shape[1]:,} vocabulary terms")

    def _resolve_title(self, title: str):
        title_lower = title.lower().strip()
        if title_lower in self._title_index:
            return title_lower
        close = get_close_matches(title_lower, self._title_index.index, n=1, cutoff=0.6)
        return close[0] if close else None

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        matched = self._resolve_title(title)
        if matched is None:
            print(f"'{title}' not found in catalogue. Try checking spelling.")
            return pd.DataFrame()

        idx = self._title_index[matched]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        similar_idx = sim_scores.argsort()[::-1]
        similar_idx = [i for i in similar_idx if i != idx][:top_n]

        result = self.df.loc[similar_idx, ["title", "type", "listed_in", "release_year", "rating"]].copy()
        result["similarity_score"] = np.round(sim_scores[similar_idx], 3)
        result = result.reset_index(drop=True)
        result.index = result.index + 1

        if matched != title.lower().strip():
            print(f"(Matched '{title}' -> '{self.df.loc[idx, 'title']}')")
        return result


if __name__ == "__main__":
    df = pd.read_csv("netflix_featured_dataset.csv")
    df["content_soup"] = df["content_soup"].fillna("")

    recommender = NetflixRecommender(df)

    sample_titles = ["Peaky Blinders", "The Conjuring", "3 Idiots"]
    for t in sample_titles:
        print(f"\n{'='*70}\nTop 10 recommendations similar to: '{t}'\n{'='*70}")
        recs = recommender.recommend(t, top_n=10)
        print(recs.to_string())
