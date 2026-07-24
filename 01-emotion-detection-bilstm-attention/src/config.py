"""Central paths and training configuration."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    max_vocab_size: int = 12000
    max_sequence_length: int = 40
    embedding_dimension: int = 96
    lstm_units: int = 64
    dense_units: int = 96
    dropout_rate: float = 0.30
    batch_size: int = 64
    epochs: int = 12
    learning_rate: float = 1e-3
    validation_size: float = 0.15
    test_size: float = 0.15
    minimum_samples_per_class: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
