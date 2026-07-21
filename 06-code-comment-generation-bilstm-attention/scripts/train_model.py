from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ModelConfig, ProjectPaths
from src.data_preprocessing import load_codesearchnet_splits
from src.model_training import train_attention_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the true BiLSTM-attention checkpoint.")
    parser.add_argument("--train-samples", type=int, default=20_000)
    parser.add_argument("--validation-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ModelConfig(epochs=args.epochs, batch_size=args.batch_size)
    random.seed(config.seed)
    np.random.seed(config.seed)
    import tensorflow as tf
    tf.random.set_seed(config.seed)

    splits = load_codesearchnet_splits(
        language=config.language,
        max_train_samples=args.train_samples,
        max_validation_samples=args.validation_samples,
        max_test_samples=args.test_samples,
        preprocessing_mode=config.preprocessing_mode,
    )
    model, history, metadata = train_attention_model(
        splits["train"], splits["validation"],
        config=config, paths=ProjectPaths(PROJECT_ROOT),
    )
    print(model.summary())
    print(history.tail())
    print(metadata)
    print("Training complete. Run scripts/evaluate_model.py next.")


if __name__ == "__main__":
    main()
