from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineMetrics:
    model_name: str
    evaluation_protocol: str
    accuracy: float
    macro_f1: float
    weighted_f1: float

    def to_record(self) -> dict[str, str | float]:
        return self.__dict__.copy()


def evaluate_tfidf_logistic_baseline(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "clinical_text",
    label_column: str = "label",
    random_seed: int = 42,
) -> tuple[BaselineMetrics, pd.DataFrame]:
    counts = dataframe[label_column].value_counts()
    minimum_support = int(counts.min())
    if minimum_support < 2:
        raise ValueError(
            "At least two rows per class are required for stratified CV."
        )

    folds = min(5, minimum_support)
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_seed,
    )
    predictions = cross_val_predict(
        pipeline,
        dataframe[text_column].astype(str),
        dataframe[label_column].astype(str),
        cv=cross_validation,
    )

    true_labels = dataframe[label_column].astype(str)
    metrics = BaselineMetrics(
        model_name="TF-IDF + Logistic Regression",
        evaluation_protocol=f"{folds}-fold stratified cross-validation",
        accuracy=float(accuracy_score(true_labels, predictions)),
        macro_f1=float(
            f1_score(true_labels, predictions, average="macro", zero_division=0)
        ),
        weighted_f1=float(
            f1_score(
                true_labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    )
    analysis = pd.DataFrame(
        {
            "clinical_text": dataframe[text_column].astype(str),
            "true_label": true_labels,
            "predicted_label": predictions,
            "is_correct": true_labels.to_numpy() == predictions,
        }
    )
    return metrics, analysis
