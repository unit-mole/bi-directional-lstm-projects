from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_training_curves(history: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for metric in ("loss", "accuracy"):
        val_metric = f"val_{metric}"
        if metric not in history or val_metric not in history:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(history[metric], label=f"training {metric}")
        plt.plot(history[val_metric], label=f"validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.title(f"Training and Validation {metric.title()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"training_{metric}_curve.png", dpi=160)
        plt.close()
