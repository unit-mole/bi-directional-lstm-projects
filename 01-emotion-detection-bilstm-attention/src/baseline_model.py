"""Optional TF-IDF logistic-regression baseline."""
from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_baseline():
    return Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=20000)),("classifier",LogisticRegression(max_iter=1000,class_weight="balanced"))])
