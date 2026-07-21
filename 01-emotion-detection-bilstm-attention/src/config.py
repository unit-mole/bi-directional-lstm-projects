"""Central configuration for the emotion detection project."""

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
    """Hyperparameters and reproducibility settings."""

    random_seed: int = 42
    max_vocab_size: int = 30_000
    max_sequence_length: int = 60
    embedding_dimension: int = 128
    lstm_units: int = 64
    dense_units: int = 128
    dropout_rate: float = 0.30
    recurrent_dropout: float = 0.0
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-3
    validation_size: float = 0.15
    test_size: float = 0.15
    minimum_samples_per_class: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
