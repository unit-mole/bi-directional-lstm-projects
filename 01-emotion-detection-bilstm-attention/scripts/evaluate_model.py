"""Evaluate an existing trained attention model on a fresh labeled CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_and_clean_dataset
from src.inference_pipeline import EmotionInferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    frame, audit = load_and_clean_dataset(args.data)
    pipeline = EmotionInferencePipeline(PROJECT_ROOT / "models").load()
    results = pipeline.predict_many(frame["text"].tolist())
    predictions = [result.predicted_emotion for result in results]
    truth = frame["emotion"].tolist()

    metrics = {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, predictions, average="weighted", zero_division=0),
        "dataset_audit": audit.to_dict(),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(truth, predictions, zero_division=0))


if __name__ == "__main__":
    main()
