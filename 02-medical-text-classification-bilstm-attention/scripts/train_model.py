from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SAMPLE_DATA, MODEL_DIR, OUTPUT_DIR, TrainingConfig
from src.data_preprocessing import load_and_prepare_dataset
from src.model_training import train_model
from src.visualization import (
    save_class_distribution,
    save_model_architecture,
    save_text_length_distribution,
    save_training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the medical text BiLSTM + Attention model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_SAMPLE_DATA,
        help="CSV dataset path.",
    )
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--preprocessing-mode",
        choices=["clinical_safe", "legacy"],
        default="clinical_safe",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        preprocessing_mode=args.preprocessing_mode,
    )
    dataframe, audit = load_and_prepare_dataset(
        args.data,
        text_column=args.text_column,
        label_column=args.label_column,
        minimum_class_count=config.minimum_class_count,
        preprocessing_mode=config.preprocessing_mode,
    )

    if len(dataframe) <= 100:
        print(
            "WARNING: The selected dataset is very small. Training will verify "
            "the pipeline but will not produce credible performance estimates."
        )

    save_class_distribution(
        dataframe,
        OUTPUT_DIR / "class_distribution.png",
    )
    save_text_length_distribution(
        dataframe,
        OUTPUT_DIR / "text_length_distribution.png",
    )
    save_model_architecture(
        OUTPUT_DIR / "model_architecture.png",
    )

    result = train_model(
        dataframe,
        model_directory=MODEL_DIR,
        output_directory=OUTPUT_DIR,
        config=config,
    )
    save_training_curves(
        result["history"],
        OUTPUT_DIR / "training_curve.png",
    )

    (OUTPUT_DIR / "data_quality_summary.json").write_text(
        json.dumps(audit.to_dict(), indent=2),
        encoding="utf-8",
    )
    print("Training complete.")
    print(f"Model artifacts: {MODEL_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
