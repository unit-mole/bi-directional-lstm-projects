from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_training_curves(history: dict[str, list[float]], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for metric in ("loss", "accuracy"):
        if metric not in history:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(history[metric], label=f"Training {metric}")
        validation_key = f"val_{metric}"
        if validation_key in history:
            plt.plot(history[validation_key], label=f"Validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.title(f"Training and Validation {metric.title()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(destination / f"training_{metric}.png", dpi=160)
        plt.close()


def save_evaluation_figures(y_true, probabilities, *, threshold: float, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prediction = (p >= threshold).astype(int)

    ConfusionMatrixDisplay.from_predictions(y, prediction, labels=[0, 1], display_labels=["No Match", "Match"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(destination / "confusion_matrix.png", dpi=160)
    plt.close()

    if len(np.unique(y)) == 2:
        RocCurveDisplay.from_predictions(y, p)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(destination / "roc_curve.png", dpi=160)
        plt.close()

        PrecisionRecallDisplay.from_predictions(y, p)
        plt.title("Precision–Recall Curve")
        plt.tight_layout()
        plt.savefig(destination / "precision_recall_curve.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(p[y == 0], bins=10, alpha=0.65, label="No Match")
    plt.hist(p[y == 1], bins=10, alpha=0.65, label="Match")
    plt.axvline(threshold, linestyle="--", label=f"Threshold {threshold:.2f}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Pair count")
    plt.title("Similarity Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "similarity_score_distribution.png", dpi=160)
    plt.close()


def save_dataset_figures(pairs: pd.DataFrame, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    counts = pairs["label"].map({0: "No Match", 1: "Match"}).value_counts()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar")
    plt.ylabel("Pairs")
    plt.title("Pair Label Distribution")
    plt.tight_layout()
    plt.savefig(destination / "label_distribution.png", dpi=160)
    plt.close()

    for column, filename, title in [
        ("resume_text", "resume_length_distribution.png", "Resume Length Distribution"),
        ("job_description", "job_description_length_distribution.png", "Job Description Length Distribution"),
    ]:
        lengths = pairs[column].str.split().str.len()
        plt.figure(figsize=(7, 4))
        plt.hist(lengths, bins=min(15, max(int(lengths.nunique()), 5)))
        plt.xlabel("Words")
        plt.ylabel("Documents")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(destination / filename, dpi=160)
        plt.close()


def save_architecture_diagram(output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    boxes = [
        (0.10, 0.70, "Resume\nsequence"),
        (0.10, 0.25, "Job description\nsequence"),
        (0.38, 0.48, "Shared embedding +\nBidirectional LSTM encoder"),
        (0.65, 0.48, "Comparison features\n|r-j|, r x j, cosine"),
        (0.88, 0.48, "Dense classifier\nMatch probability"),
    ]
    for x, y, label in boxes:
        ax.text(x, y, label, ha="center", va="center", fontsize=12,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "black"})
    arrows = [((0.18, 0.70), (0.31, 0.55)), ((0.18, 0.25), (0.31, 0.43)),
              ((0.49, 0.48), (0.58, 0.48)), ((0.75, 0.48), (0.81, 0.48))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.set_title("Shared Siamese BiLSTM Architecture", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
