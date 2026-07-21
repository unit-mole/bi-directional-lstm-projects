from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    language: str = "python"
    max_code_vocab: int = 40_000
    max_comment_vocab: int = 20_000
    max_code_len: int = 180
    max_comment_len: int = 30
    embedding_dim: int = 128
    encoder_units: int = 128
    dropout: float = 0.20
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    beam_width: int = 3
    start_token: str = "<start>"
    end_token: str = "<end>"
    oov_token: str = "<OOV>"
    seed: int = 42
    preprocessing_mode: str = "semantic"

    @property
    def decoder_units(self) -> int:
        return self.encoder_units * 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        return cls(Path(__file__).resolve().parents[1])

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "model_metadata.json"

    @property
    def code_tokenizer_path(self) -> Path:
        return self.models_dir / "code_tokenizer_config.json"

    @property
    def comment_tokenizer_path(self) -> Path:
        return self.models_dir / "comment_tokenizer_config.json"

    @property
    def attention_model_path(self) -> Path:
        return self.models_dir / "code_comment_bilstm_attention_model.keras"

    @property
    def legacy_model_path(self) -> Path:
        return self.models_dir / "code_comment_bilstm_seq2seq_model.keras"

    @property
    def encoder_model_path(self) -> Path:
        return self.models_dir / "encoder_model.keras"

    @property
    def decoder_model_path(self) -> Path:
        return self.models_dir / "decoder_model.keras"
