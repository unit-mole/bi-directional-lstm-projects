"""Lightweight structural and artifact validation used locally and in CI."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED = [
    "README.md", "requirements.txt", "app/streamlit_app.py",
    "src/inference_pipeline.py", "src/crf_layer.py",
    "models/model_metadata.json", "models/word_to_index.pkl",
    "models/tag_to_index.pkl", "models/legacy_bilstm_softmax_model.h5",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()
    missing = [item for item in REQUIRED if not (PROJECT_ROOT / item).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required project files: {missing}")

    metadata = json.loads((PROJECT_ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
    with (PROJECT_ROOT / "models/word_to_index.pkl").open("rb") as handle:
        words = pickle.load(handle)
    with (PROJECT_ROOT / "models/tag_to_index.pkl").open("rb") as handle:
        tags = pickle.load(handle)
    assert words.get("<PAD>") == 0
    assert "<UNK>" in words
    assert tags.get("O") == 0
    assert len(words) == metadata["vocab_size"]
    assert len(tags) == metadata["num_tags"]

    from src.inference_pipeline import NERInferencePipeline
    pipeline = NERInferencePipeline(PROJECT_ROOT / "models")
    assert pipeline.max_length == metadata["max_sequence_length"]
    if not args.skip_model_load:
        _ = pipeline.model_kind
    print("Project validation passed.")


if __name__ == "__main__":
    main()
