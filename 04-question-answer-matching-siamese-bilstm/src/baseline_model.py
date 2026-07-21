from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfPairBaseline:
    def __init__(self, *, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, sublinear_tf=True)

    def fit(self, text_a: list[str], text_b: list[str]) -> "TfidfPairBaseline":
        self.vectorizer.fit(list(text_a) + list(text_b))
        return self

    def score_pairs(self, text_a: list[str], text_b: list[str]) -> np.ndarray:
        a_matrix = self.vectorizer.transform(text_a)
        b_matrix = self.vectorizer.transform(text_b)
        return np.array([cosine_similarity(a_matrix[i], b_matrix[i])[0, 0] for i in range(len(text_a))])
