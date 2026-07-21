from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_pair_dataset
from src.inference_pipeline import QAMatcher
from src.model_evaluation import evaluate_probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    frame = load_pair_dataset(args.data, require_label=True)
    matcher = QAMatcher.from_artifacts(PROJECT_ROOT / "models")
    scored = matcher.predict_frame(frame.rename(columns={"text_a": "text_a", "text_b": "text_b"}))
    metrics = evaluate_probabilities(frame["label"], scored["match_probability"], threshold=matcher.threshold)
    print(json.dumps(metrics, indent=2))
    scored.to_csv(PROJECT_ROOT / "outputs" / "evaluation_predictions.csv", index=False)


if __name__ == "__main__":
    main()
