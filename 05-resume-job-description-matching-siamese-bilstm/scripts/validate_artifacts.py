from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-tensorflow", action="store_true")
    args = parser.parse_args()

    required = [CONFIG.model_path, CONFIG.tokenizer_path, CONFIG.metadata_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing artifacts: {missing}")

    metadata = json.loads(CONFIG.metadata_path.read_text(encoding="utf-8"))
    assert metadata.get("architecture", {}).get("shared_encoder") is True
    assert 0.0 < float(metadata.get("prediction_threshold", 0.0)) < 1.0

    if not args.allow_missing_tensorflow:
        from src.inference_pipeline import ResumeJobMatcher
        matcher = ResumeJobMatcher(config=CONFIG, allow_fallback=False)
        result = matcher.predict(
            "Data scientist with Python SQL machine learning and NLP experience.",
            "Seeking a machine learning engineer with Python NLP SQL and model deployment skills.",
        )
        assert 0.0 <= result["fit_score"] <= 1.0
    print("Artifact validation passed.")


if __name__ == "__main__":
    main()
