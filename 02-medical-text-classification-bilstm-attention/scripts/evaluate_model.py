from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SAMPLE_DATA, OUTPUT_DIR
from src.data_preprocessing import load_and_prepare_dataset
from src.inference_pipeline import MedicalTextInferencePipeline
from src.model_evaluation import (
    build_error_analysis,
    evaluate_multiclass_predictions,
    save_evaluation_outputs,
)
from src.visualization import save_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved model on a labeled CSV."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_SAMPLE_DATA)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = MedicalTextInferencePipeline()
    pipeline.load()

    dataframe, _ = load_and_prepare_dataset(
        args.data,
        text_column=args.text_column,
        label_column=args.label_column,
        minimum_class_count=1,
        preprocessing_mode=pipeline.metadata["preprocessing_mode"],
        remove_duplicate_text=False,
    )

    mapping_inverse = {
        label: index
        for index, label in pipeline.label_mapping.items()
    }
    unknown_labels = sorted(set(dataframe["label"]) - set(mapping_inverse))
    if unknown_labels:
        raise ValueError(
            f"Dataset contains labels not learned by the model: {unknown_labels}"
        )

    prediction_frame = pipeline.predict_batch(dataframe["clinical_text"])
    y_true = dataframe["label"].map(mapping_inverse).to_numpy()
    y_pred = prediction_frame["predicted_label"].map(mapping_inverse).to_numpy()
    probability_columns = [
        f"probability__{label}"
        for label in pipeline.class_labels
    ]
    probabilities = prediction_frame[probability_columns].to_numpy()

    metrics, report, confusion = evaluate_multiclass_predictions(
        y_true,
        y_pred,
        class_labels=pipeline.class_labels,
        probabilities=probabilities,
    )
    errors = build_error_analysis(
        dataframe["clinical_text"].tolist(),
        y_true,
        y_pred,
        probabilities,
        class_labels=pipeline.class_labels,
    )
    save_evaluation_outputs(
        metrics=metrics,
        report_frame=report,
        confusion_frame=confusion,
        error_frame=errors,
        output_directory=OUTPUT_DIR,
    )
    save_confusion_matrix(
        confusion,
        OUTPUT_DIR / "confusion_matrix.png",
    )
    print(metrics)


if __name__ == "__main__":
    main()
