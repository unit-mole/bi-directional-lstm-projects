from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

from .text_preprocessing import compare_skills


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float


def tune_threshold(y_true: Iterable[int], probabilities: Iterable[float]) -> ThresholdResult:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    best = ThresholdResult(threshold=0.50, f1=-1.0)
    for threshold in np.linspace(0.20, 0.80, 121):
        score = f1_score(y, (p >= threshold).astype(int), zero_division=0)
        if score > best.f1:
            best = ThresholdResult(float(threshold), float(score))
    return best


def binary_metrics(y_true: Iterable[int], probabilities: Iterable[float], *, threshold: float) -> dict[str, float | list[list[int]]]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    prediction = (p >= threshold).astype(int)
    metrics: dict[str, float | list[list[int]]] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1_score": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
        metrics["pr_auc"] = float(average_precision_score(y, p))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def tfidf_pair_scores(resumes: list[str], jobs: list[str]) -> np.ndarray:
    scores: list[float] = []
    for resume, job in zip(resumes, jobs):
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform([resume, job])
        scores.append(float(cosine_similarity(matrix[0], matrix[1])[0, 0]))
    return np.asarray(scores)


def skill_overlap_scores(resumes: list[str], jobs: list[str]) -> np.ndarray:
    return np.asarray([compare_skills(r, j)["skill_coverage"] for r, j in zip(resumes, jobs)], dtype=float)


def baseline_comparison(
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, scoring_function in [
        ("TF-IDF cosine similarity", tfidf_pair_scores),
        ("Transparent skill overlap", skill_overlap_scores),
    ]:
        val_scores = scoring_function(validation["resume_text"].tolist(), validation["job_description"].tolist())
        tuned = tune_threshold(validation["label"], val_scores)
        test_scores = scoring_function(test["resume_text"].tolist(), test["job_description"].tolist())
        values = binary_metrics(test["label"], test_scores, threshold=tuned.threshold)
        rows.append({"model": name, **{k: v for k, v in values.items() if k != "confusion_matrix"}})
    return pd.DataFrame(rows)


def ranking_metrics(ranking_frame: pd.DataFrame, *, top_k: int = 3) -> dict[str, float]:
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    for _, group in ranking_frame.groupby("job_id"):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        relevance = (ordered["resume_category"] == ordered["job_category"]).astype(int).to_numpy()
        relevant_positions = np.flatnonzero(relevance == 1)
        reciprocal_ranks.append(0.0 if len(relevant_positions) == 0 else 1.0 / (int(relevant_positions[0]) + 1))
        recalls.append(float(relevance[:top_k].sum() > 0))
        ndcgs.append(float(ndcg_score([relevance], [ordered["score"].to_numpy()])))
    return {
        f"recall_at_{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
