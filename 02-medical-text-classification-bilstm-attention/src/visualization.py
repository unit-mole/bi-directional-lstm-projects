from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_class_distribution(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> None:
    counts = dataframe["label"].value_counts().sort_values(ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    counts.plot(kind="barh", ax=axis)
    axis.set_title("Medical Specialty Distribution")
    axis.set_xlabel("Rows")
    axis.set_ylabel("Medical specialty")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_text_length_distribution(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> None:
    lengths = dataframe["clinical_text"].astype(str).str.split().str.len()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(lengths, bins=min(20, max(5, len(lengths))))
    axis.set_title("Clinical Text Length Distribution")
    axis.set_xlabel("Words per text")
    axis.set_ylabel("Frequency")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_training_curves(
    history_frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    if "loss" in history_frame:
        axis.plot(history_frame.index + 1, history_frame["loss"], label="train loss")
    if "val_loss" in history_frame:
        axis.plot(
            history_frame.index + 1,
            history_frame["val_loss"],
            label="validation loss",
        )
    axis.set_title("Training and Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confusion_matrix(
    confusion_frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    matrix = confusion_frame.to_numpy()
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(confusion_frame.columns)))
    axis.set_xticklabels(confusion_frame.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(confusion_frame.index)))
    axis.set_yticklabels(confusion_frame.index)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_model_architecture(output_path: str | Path) -> None:
    steps = [
        "Medical text",
        "Tokenizer + padding",
        "Embedding",
        "Bidirectional LSTM",
        "Temporal attention",
        "Dense + dropout",
        "Softmax classes",
    ]
    figure, axis = plt.subplots(figsize=(12, 2.8))
    axis.axis("off")
    x_positions = np.linspace(0.05, 0.95, len(steps))
    for index, (x_value, label) in enumerate(zip(x_positions, steps)):
        axis.text(
            x_value,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white"},
            transform=axis.transAxes,
        )
        if index < len(steps) - 1:
            axis.annotate(
                "",
                xy=(x_positions[index + 1] - 0.055, 0.5),
                xytext=(x_value + 0.055, 0.5),
                xycoords=axis.transAxes,
                arrowprops={"arrowstyle": "->"},
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
