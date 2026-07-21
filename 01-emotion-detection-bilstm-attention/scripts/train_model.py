"""Command-line entry point for attention-model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainingConfig
from src.model_training import train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BiLSTM-with-attention emotion model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "emotion_dataset.csv",
        help="CSV containing a text column and an emotion/label column.",
    )
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--max-sequence-length", type=int, default=TrainingConfig.max_sequence_length)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
    )
    metadata = train_pipeline(args.data, config=config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
