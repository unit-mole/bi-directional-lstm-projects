from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.index + 1, history["loss"], marker="o", label="Training loss")
    ax.plot(history.index + 1, history["val_loss"], marker="o", label="Validation loss")
    ax.set(title="Training and validation loss", xlabel="Epoch", ylabel="Sparse categorical cross-entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_attention_heatmap(
    attention: np.ndarray,
    code_tokens: list[str],
    generated_tokens: list[str],
    output_path: str | Path,
) -> None:
    rows = min(len(generated_tokens), attention.shape[0])
    cols = min(len(code_tokens), attention.shape[1], 40)
    fig, ax = plt.subplots(figsize=(max(9, cols * 0.25), max(4, rows * 0.35)))
    image = ax.imshow(attention[:rows, :cols], aspect="auto")
    ax.set_xticks(range(cols), code_tokens[:cols], rotation=70, ha="right")
    ax.set_yticks(range(rows), generated_tokens[:rows])
    ax.set(xlabel="Source-code tokens", ylabel="Generated tokens", title="Attention alignment")
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
