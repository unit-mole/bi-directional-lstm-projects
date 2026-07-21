"""A transparent TF-IDF + Logistic Regression comparison baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .data_preprocessing import load_and_clean_dataset, split_dataframe, validate_class_support


def train_baseline(data_path: str | Path, output_dir: str | Path) -> dict[str, float]:
    frame, _ = load_and_clean_dataset(data_path)
    validate_class_support(frame, minimum_samples_per_class=3)
    train_df, _, test_df = split_dataframe(frame)

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30_000)),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipeline.fit(train_df["text_clean"], train_df["emotion"])
    predictions = pipeline.predict(test_df["text_clean"])
    metrics = {
        "accuracy": float(accuracy_score(test_df["emotion"], predictions)),
        "macro_f1": float(f1_score(test_df["emotion"], predictions, average="macro")),
        "weighted_f1": float(f1_score(test_df["emotion"], predictions, average="weighted")),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "tfidf_logistic_regression.joblib")
    (output_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": "TF-IDF + Logistic Regression", **metrics}]).to_csv(
        output_dir / "baseline_comparison.csv", index=False
    )
    return metrics
