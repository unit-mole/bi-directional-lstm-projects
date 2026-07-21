from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_METADATA_KEYS = {
    "project_name",
    "model_type",
    "class_labels",
    "max_sequence_length",
    "preprocessing_mode",
}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_label_mapping(path: str | Path) -> dict[int, str]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Label mapping must be a JSON object.")

    mapping = {int(key): str(value) for key, value in raw.items()}
    expected = list(range(len(mapping)))
    if sorted(mapping) != expected:
        raise ValueError(
            "Label mapping keys must be contiguous integers starting at zero."
        )
    return mapping


def load_model_metadata(path: str | Path) -> dict[str, Any]:
    metadata = load_json(path)
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")

    missing = REQUIRED_METADATA_KEYS.difference(metadata)
    if missing:
        raise ValueError(f"Model metadata is missing keys: {sorted(missing)}")
    return metadata
