from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = [
    "README.md",
    "app/streamlit_app.py",
    "src/inference_pipeline.py",
    "src/seq2seq_model.py",
    "models/model_metadata.json",
    "models/code_tokenizer_config.json",
    "models/comment_tokenizer_config.json",
    "data/sample_code_comment_pairs.csv",
    "requirements.txt",
    "Dockerfile",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    missing = [item for item in REQUIRED if not (root / item).exists()]
    metadata = json.loads((root / "models/model_metadata.json").read_text())
    assert metadata["language"] == "python"
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    if not args.ci:
        model_files = list((root / "models").glob("*.keras"))
        if not model_files:
            raise SystemExit("No .keras model checkpoint found.")
    print("Project validation passed.")


if __name__ == "__main__":
    main()
