from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import evaluate_tfidf_logistic_baseline
from src.config import DEFAULT_SAMPLE_DATA, OUTPUT_DIR
from src.data_preprocessing import load_and_prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TF-IDF + Logistic Regression baseline."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_SAMPLE_DATA)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe, _ = load_and_prepare_dataset(
        args.data,
        text_column=args.text_column,
        label_column=args.label_column,
        minimum_class_count=2,
        preprocessing_mode="clinical_safe",
    )
    metrics, analysis = evaluate_tfidf_logistic_baseline(dataframe)
    pd.DataFrame([metrics.to_record()]).to_csv(
        OUTPUT_DIR / "baseline_metrics.csv",
        index=False,
    )
    analysis.to_csv(
        OUTPUT_DIR / "baseline_prediction_analysis.csv",
        index=False,
    )
    print(metrics.to_record())


if __name__ == "__main__":
    main()
