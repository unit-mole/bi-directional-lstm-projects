"""Static visualizations saved during training and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_eda_outputs(frame: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = frame["emotion"].value_counts().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=axis)
    axis.set_title("Emotion Class Distribution")
    axis.set_xlabel("Emotion")
    axis.set_ylabel("Text samples")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(figure)

    lengths = frame["text_clean"].str.split().str.len()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(lengths, bins=min(30, max(5, int(lengths.nunique()))))
    axis.set_title("Text Length Distribution")
    axis.set_xlabel("Tokens")
    axis.set_ylabel("Frequency")
    figure.tight_layout()
    figure.savefig(output_dir / "text_length_distribution.png", dpi=160)
    plt.close(figure)

    average_lengths = frame.assign(text_length=lengths).groupby("emotion")["text_length"].mean()
    average_lengths.rename("average_token_length").to_csv(
        output_dir / "average_text_length_by_emotion.csv"
    )


def save_training_curves(history: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    for metric, validation_metric, filename, ylabel in [
        ("accuracy", "val_accuracy", "training_accuracy_curve.png", "Accuracy"),
        ("loss", "val_loss", "training_loss_curve.png", "Loss"),
    ]:
        if metric not in history or validation_metric not in history:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history.index + 1, history[metric], marker="o", label="Training")
        axis.plot(history.index + 1, history[validation_metric], marker="o", label="Validation")
        axis.set_title(f"Training and Validation {ylabel}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Emotion Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
