"""Evaluation, reporting, and error-analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

from .visualization import save_confusion_matrix


def evaluate_and_save(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_frame: pd.DataFrame,
    label_mapping: dict[str, str],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = np.asarray(model.predict(x_test, verbose=0))
    predictions = probabilities.argmax(axis=1)
    labels = sorted(int(index) for index in label_mapping)
    target_names = [label_mapping[str(index)] for index in labels]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "test_rows": int(len(y_test)),
    }
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    save_confusion_matrix(matrix, target_names, output_dir / "confusion_matrix.png")
    pd.DataFrame(matrix, index=target_names, columns=target_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    analysis = test_frame[["text", "emotion"]].copy().reset_index(drop=True)
    analysis["true_label_id"] = y_test
    analysis["predicted_label_id"] = predictions
    analysis["true_emotion"] = [label_mapping[str(int(value))] for value in y_test]
    analysis["predicted_emotion"] = [label_mapping[str(int(value))] for value in predictions]
    analysis["confidence"] = probabilities.max(axis=1)
    analysis["is_correct"] = analysis["true_label_id"] == analysis["predicted_label_id"]
    analysis.to_csv(output_dir / "prediction_analysis.csv", index=False)
    analysis[analysis["is_correct"] == False].sort_values("confidence", ascending=False).to_csv(
        output_dir / "misclassified_examples.csv", index=False
    )
    return metrics
