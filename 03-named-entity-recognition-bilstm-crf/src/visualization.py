"""Matplotlib visualizations for training, tags, entities, and confusion matrices."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save(figure: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(history: dict[str, Sequence[float]], output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    if "loss" in history:
        axis.plot(history["loss"], marker="o", label="Training loss")
    if "val_loss" in history:
        axis.plot(history["val_loss"], marker="o", label="Validation loss")
    axis.set_title("BiLSTM-CRF Training Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(figure, output_path)


def plot_distribution(values: Iterable[str], title: str, output_path: str | Path) -> None:
    counts = Counter(values)
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = list(counts)
    bars = axis.bar(labels, [counts[label] for label in labels])
    axis.set_title(title)
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=45)
    for bar in bars:
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), ha="center", va="bottom")
    _save(figure, output_path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", aspect="auto")
    figure.colorbar(image, ax=axis)
    axis.set_title("Token Tag Confusion Matrix")
    axis.set_xlabel("Predicted tag")
    axis.set_ylabel("True tag")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
    _save(figure, output_path)
