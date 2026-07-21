"""Train the TF-IDF + Logistic Regression comparison baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_model import train_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "emotion_dataset.csv")
    args = parser.parse_args()
    print(json.dumps(train_baseline(args.data, PROJECT_ROOT / "outputs"), indent=2))


if __name__ == "__main__":
    main()
