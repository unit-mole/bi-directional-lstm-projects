from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import load_label_mapping, load_model_metadata
from src.config import (
    DEFAULT_LABEL_MAPPING_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate file structure without importing TensorFlow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        DEFAULT_MODEL_PATH,
        DEFAULT_TOKENIZER_PATH,
        DEFAULT_LABEL_MAPPING_PATH,
        DEFAULT_METADATA_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")

    mapping = load_label_mapping(DEFAULT_LABEL_MAPPING_PATH)
    metadata = load_model_metadata(DEFAULT_METADATA_PATH)
    tokenizer_payload = json.loads(
        DEFAULT_TOKENIZER_PATH.read_text(encoding="utf-8")
    )
    if not isinstance(tokenizer_payload, (str, dict)):
        raise ValueError("Unsupported tokenizer_config.json payload.")

    with zipfile.ZipFile(DEFAULT_MODEL_PATH) as archive:
        model_members = set(archive.namelist())
    required_members = {"metadata.json", "config.json", "model.weights.h5"}
    if not required_members.issubset(model_members):
        raise ValueError("The .keras archive is incomplete.")

    expected_labels = [mapping[index] for index in sorted(mapping)]
    if metadata["class_labels"] != expected_labels:
        raise ValueError("Metadata labels and label mapping differ.")

    if not args.metadata_only:
        from src.inference_pipeline import MedicalTextInferencePipeline

        MedicalTextInferencePipeline().load()

    print(
        {
            "status": "valid",
            "classes": expected_labels,
            "metadata_only": args.metadata_only,
        }
    )


if __name__ == "__main__":
    main()
