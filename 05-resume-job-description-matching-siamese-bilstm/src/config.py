from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    project_dir: Path = Path(__file__).resolve().parents[1]
    random_seed: int = 42
    max_vocabulary_size: int = 12_000
    max_sequence_length: int = 48
    embedding_dimension: int = 32
    bilstm_units: int = 12
    projection_dimension: int = 32
    dropout_rate: float = 0.25
    batch_size: int = 32
    epochs: int = 18
    learning_rate: float = 1e-3
    default_threshold: float = 0.50

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def models_dir(self) -> Path:
        return self.project_dir / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.project_dir / "outputs"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "resume_job_siamese_bilstm_model.keras"

    @property
    def tokenizer_path(self) -> Path:
        return self.models_dir / "tokenizer.json"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "model_metadata.json"

    @property
    def training_pairs_path(self) -> Path:
        return self.data_dir / "processed" / "resume_job_pairs.csv"

    def to_serializable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["project_dir"] = str(self.project_dir)
        return payload


CONFIG = ProjectConfig()
