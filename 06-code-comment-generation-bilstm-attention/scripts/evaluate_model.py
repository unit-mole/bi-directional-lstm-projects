from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_tabular_dataset, standardize_dataframe
from src.inference_pipeline import CodeCommentInferencePipeline
from src.model_evaluation import aggregate_metrics, evaluate_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_code_comment_pairs.csv")
    parser.add_argument("--method", choices=["greedy", "beam"], default="beam")
    args = parser.parse_args()

    data = standardize_dataframe(load_tabular_dataset(PROJECT_ROOT / args.data))
    pipeline = CodeCommentInferencePipeline(PROJECT_ROOT).load()
    rows = []
    metrics = []
    for record in data.to_dict("records"):
        result = pipeline.generate(record["code"], method=args.method)
        row_metrics = evaluate_pair(record["comment_clean"], result.comment)
        metrics.append(row_metrics)
        rows.append({
            "code": record["code"],
            "reference_comment": record["comment_clean"],
            "generated_comment": result.comment,
            **row_metrics,
        })
    output = PROJECT_ROOT / "outputs" / "generated_comment_examples.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    aggregate = aggregate_metrics(metrics)
    (PROJECT_ROOT / "outputs" / "model_metrics_latest.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
