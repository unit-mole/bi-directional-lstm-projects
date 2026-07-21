from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_multiclass_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    class_labels: list[str],
    probabilities: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    true_array = np.asarray(y_true, dtype=int)
    pred_array = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_labels)))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(true_array, pred_array)),
        "macro_precision": float(
            precision_score(true_array, pred_array, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(true_array, pred_array, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(true_array, pred_array, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(true_array, pred_array, labels=labels, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(true_array, pred_array, labels=labels, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(true_array, pred_array, labels=labels, average="weighted", zero_division=0)
        ),
        "test_rows": int(len(true_array)),
    }

    if probabilities is not None and len(probabilities) == len(true_array):
        top_k = min(3, probabilities.shape[1])
        top_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
        metrics[f"top_{top_k}_accuracy"] = float(
            np.mean(
                [
                    target in candidates
                    for target, candidates in zip(true_array, top_indices)
                ]
            )
        )

    report = classification_report(
        true_array,
        pred_array,
        labels=labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).T.reset_index(names="class_or_average")

    matrix = confusion_matrix(true_array, pred_array, labels=labels)
    confusion_frame = pd.DataFrame(
        matrix,
        index=class_labels,
        columns=class_labels,
    )
    return metrics, report_frame, confusion_frame


def build_error_analysis(
    texts: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: np.ndarray,
    *,
    class_labels: list[str],
) -> pd.DataFrame:
    true_array = np.asarray(y_true, dtype=int)
    pred_array = np.asarray(y_pred, dtype=int)
    probability_array = np.asarray(probabilities)

    rows: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        predicted_id = int(pred_array[index])
        true_id = int(true_array[index])
        rows.append(
            {
                "clinical_text": str(text),
                "true_label": class_labels[true_id],
                "predicted_label": class_labels[predicted_id],
                "confidence": float(probability_array[index, predicted_id]),
                "is_correct": bool(true_id == predicted_id),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["is_correct", "confidence"],
        ascending=[True, False],
    )


def save_evaluation_outputs(
    *,
    metrics: dict[str, Any],
    report_frame: pd.DataFrame,
    confusion_frame: pd.DataFrame,
    error_frame: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    report_frame.to_csv(destination / "classification_report.csv", index=False)
    confusion_frame.to_csv(destination / "confusion_matrix.csv")
    error_frame.to_csv(destination / "error_analysis.csv", index=False)
