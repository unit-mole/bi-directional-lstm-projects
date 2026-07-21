from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PredictionResult:
    input_text: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    top_probabilities: list[tuple[str, float]]
    important_terms: list[tuple[str, float]]

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "input_text": self.input_text,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
        }
        for rank, (label, probability) in enumerate(
            self.top_probabilities,
            start=1,
        ):
            record[f"top_{rank}_label"] = label
            record[f"top_{rank}_probability"] = probability
        return record


def rank_probabilities(
    probabilities: Iterable[float],
    label_mapping: dict[int, str],
    *,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    values = np.asarray(list(probabilities), dtype=float)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    if len(values) != len(label_mapping):
        raise ValueError(
            "Probability count does not match label mapping count."
        )

    ranked_ids = np.argsort(values)[::-1][: max(1, min(top_k, len(values)))]
    return [
        (label_mapping[int(class_id)], float(values[class_id]))
        for class_id in ranked_ids
    ]
