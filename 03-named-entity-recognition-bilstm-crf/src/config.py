"""Central paths and default training configuration."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    max_vocab_size: int = 30_000
    max_sequence_length: int = 124
    embedding_dim: int = 100
    lstm_units: int = 128
    dense_units: int = 64
    dropout_rate: float = 0.30
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 15
    patience: int = 3
    lowercase_tokens: bool = True
    min_token_frequency: int = 1
    pad_token: str = "<PAD>"
    unknown_token: str = "<UNK>"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
