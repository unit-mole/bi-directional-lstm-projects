from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Siamese BiLSTM semantic matching model.")
    parser.add_argument("--data", type=Path, required=True, help="CSV containing two text columns and a binary label.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = train(
        data_path=args.data,
        output_model_dir=PROJECT_ROOT / "models",
        output_dir=PROJECT_ROOT / "outputs",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("Training completed.")
    print(metadata)


if __name__ == "__main__":
    main()
