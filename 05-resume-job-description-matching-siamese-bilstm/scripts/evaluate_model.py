from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG
from src.inference_pipeline import ResumeJobMatcher
from src.model_evaluation import binary_metrics


def main() -> None:
    pairs = pd.read_csv(CONFIG.training_pairs_path)
    test = pairs[pairs["split"] == "test"].copy()
    matcher = ResumeJobMatcher(config=CONFIG, allow_fallback=False)
    probabilities = [matcher.predict(r, j)["fit_score"] for r, j in zip(test["resume_text"], test["job_description"])]
    threshold = float(matcher.metadata.get("prediction_threshold", 0.5))
    metrics = binary_metrics(test["label"], probabilities, threshold=threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
