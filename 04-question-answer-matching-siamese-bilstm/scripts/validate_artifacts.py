from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = PROJECT_ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    model_path = MODEL_DIR / "qa_siamese_bilstm_model.keras"
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    metadata_path = MODEL_DIR / "model_metadata.json"
    for path in (model_path, tokenizer_path, metadata_path):
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty artifact: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_keys = {"max_sequence_length", "prediction_threshold", "label_mapping"}
    missing = required_keys - metadata.keys()
    if missing:
        raise SystemExit(f"Metadata missing keys: {sorted(missing)}")

    with zipfile.ZipFile(model_path) as archive:
        required_members = {"metadata.json", "config.json", "model.weights.h5"}
        if not required_members.issubset(archive.namelist()):
            raise SystemExit("Keras archive is incomplete")

    if not args.metadata_only:
        from src.inference_pipeline import QAMatcher
        matcher = QAMatcher.from_artifacts(MODEL_DIR)
        result = matcher.predict_pair("What is machine learning?", "Can you explain machine learning?")
        print(result.as_dict())
    else:
        print("Artifact metadata validation passed.")


if __name__ == "__main__":
    main()
