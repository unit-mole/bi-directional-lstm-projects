"""Lightweight CI validation without retraining or loading the neural network."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "app" / "streamlit_app.py",
    PROJECT_ROOT / "src" / "attention_layer.py",
    PROJECT_ROOT / "src" / "model_training.py",
    PROJECT_ROOT / "models" / "legacy_emotion_bilstm_model.keras",
    PROJECT_ROOT / "models" / "legacy_tokenizer_config.json",
    PROJECT_ROOT / "models" / "legacy_label_mapping.json",
]

missing = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_PATHS if not path.exists()]
if missing:
    raise SystemExit("Missing required files: " + ", ".join(missing))

mapping = json.loads((PROJECT_ROOT / "models" / "legacy_label_mapping.json").read_text())
if not mapping:
    raise SystemExit("Legacy label mapping is empty.")
print("Project structure validation passed.")
