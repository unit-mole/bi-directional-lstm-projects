from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_SAMPLE_DATA = DATA_DIR / "sample_medical_text_data.csv"
DEFAULT_MODEL_PATH = MODEL_DIR / "medical_text_bilstm_attention_model.keras"
DEFAULT_TOKENIZER_PATH = MODEL_DIR / "tokenizer_config.json"
DEFAULT_LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"
DEFAULT_METADATA_PATH = MODEL_DIR / "model_metadata.json"


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for reproducible experiments."""

    random_seed: int = 42
    max_vocabulary_size: int = 30_000
    max_sequence_length: int = 300
    embedding_dimension: int = 128
    bilstm_units: int = 64
    dense_units: int = 128
    dropout_rate: float = 0.30
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    minimum_class_count: int = 2
    preprocessing_mode: str = "clinical_safe"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
