from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG
from src.visualization import (
    save_architecture_diagram,
    save_dataset_figures,
    save_evaluation_figures,
    save_training_curves,
)


def main() -> None:
    pairs = pd.read_csv(CONFIG.training_pairs_path)
    history = pd.read_csv(CONFIG.outputs_dir / "metrics" / "training_history.csv").to_dict(orient="list")
    predictions = pd.read_csv(CONFIG.outputs_dir / "predictions" / "sample_predictions.csv")
    metadata = json.loads(CONFIG.metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata["prediction_threshold"])
    figure_dir = CONFIG.outputs_dir / "figures"
    save_training_curves(history, figure_dir)
    save_evaluation_figures(
        predictions["label"],
        predictions["match_probability"],
        threshold=threshold,
        output_dir=figure_dir,
    )
    save_dataset_figures(pairs, figure_dir)
    save_architecture_diagram(CONFIG.project_dir / "images" / "architecture.png")
    print("Figures generated.")


if __name__ == "__main__":
    main()
