"""Entity-level and token-level evaluation with persisted reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report as token_classification_report
from sklearn.metrics import confusion_matrix

from .crf_layer import viterbi_decode_numpy
from .visualization import plot_confusion_matrix


def decode_crf_predictions(model, x: np.ndarray, lengths: Sequence[int]) -> list[list[int]]:
    emissions = model(x, training=False).numpy()
    return viterbi_decode_numpy(emissions, model.transition_params.numpy(), lengths)


def evaluate_sequences(
    true_tags: Sequence[Sequence[str]],
    predicted_tags: Sequence[Sequence[str]],
    output_dir: str | Path,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "entity_precision": float(precision_score(true_tags, predicted_tags)),
        "entity_recall": float(recall_score(true_tags, predicted_tags)),
        "entity_f1": float(f1_score(true_tags, predicted_tags)),
        "seqeval_token_accuracy": float(accuracy_score(true_tags, predicted_tags)),
    }
    report_dict = classification_report(true_tags, predicted_tags, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).T.to_csv(
        output_dir / "entity_level_classification_report.csv", index_label="label"
    )

    flat_true = [tag for sequence in true_tags for tag in sequence]
    flat_predicted = [tag for sequence in predicted_tags for tag in sequence]
    token_report = token_classification_report(
        flat_true, flat_predicted, output_dict=True, zero_division=0
    )
    pd.DataFrame(token_report).T.to_csv(
        output_dir / "token_level_classification_report.csv", index_label="label"
    )
    labels = sorted(set(flat_true).union(flat_predicted))
    matrix = confusion_matrix(flat_true, flat_predicted, labels=labels)
    plot_confusion_matrix(matrix, labels, output_dir / "confusion_matrix.png")

    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def build_error_analysis(
    tokens: Sequence[Sequence[str]],
    true_tags: Sequence[Sequence[str]],
    predicted_tags: Sequence[Sequence[str]],
    output_path: str | Path,
    limit: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sentence_id, (sentence, truth, prediction) in enumerate(
        zip(tokens, true_tags, predicted_tags)
    ):
        for token_index, (token, true_tag, predicted_tag) in enumerate(
            zip(sentence, truth, prediction)
        ):
            if true_tag != predicted_tag:
                rows.append({
                    "sentence_id": sentence_id,
                    "token_index": token_index,
                    "token": token,
                    "true_tag": true_tag,
                    "predicted_tag": predicted_tag,
                    "error_type": "boundary_or_type_error",
                })
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame
