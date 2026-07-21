from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from src.code_preprocessing import CodePreprocessingOptions, preprocess_code
from src.comment_generation import GenerationResult, beam_search_decode, greedy_decode
from src.config import ProjectPaths
from src.sequence_generation import prepare_single_code
from src.tokenizer_utils import load_tokenizer


class ArtifactError(RuntimeError):
    pass


class CodeCommentInferencePipeline:
    """Loads saved artifacts once and exposes a stable generation API."""

    def __init__(self, project_root: str | Path | None = None):
        self.paths = ProjectPaths(Path(project_root)) if project_root else ProjectPaths.discover()
        self.metadata: dict[str, Any] = {}
        self.code_tokenizer = None
        self.comment_tokenizer = None
        self.encoder_model = None
        self.decoder_model = None
        self.attention_enabled = False
        self.loaded = False

    def artifact_status(self) -> dict[str, bool]:
        return {
            "metadata": self.paths.metadata_path.exists(),
            "code_tokenizer": self.paths.code_tokenizer_path.exists(),
            "comment_tokenizer": self.paths.comment_tokenizer_path.exists(),
            "attention_model": self.paths.attention_model_path.exists(),
            "legacy_model": self.paths.legacy_model_path.exists(),
            "encoder_model": self.paths.encoder_model_path.exists(),
            "decoder_model": self.paths.decoder_model_path.exists(),
        }

    def healthcheck(self, *, require_model: bool = True) -> tuple[bool, list[str]]:
        status = self.artifact_status()
        missing = [name for name in ("metadata", "code_tokenizer", "comment_tokenizer") if not status[name]]
        has_model = status["attention_model"] or status["legacy_model"] or (
            status["encoder_model"] and status["decoder_model"]
        )
        if require_model and not has_model:
            missing.append("model checkpoint")
        return not missing, missing

    def load(self) -> "CodeCommentInferencePipeline":
        ok, missing = self.healthcheck()
        if not ok:
            raise ArtifactError(f"Missing required artifacts: {', '.join(missing)}")
        self.metadata = json.loads(self.paths.metadata_path.read_text(encoding="utf-8"))
        self.code_tokenizer = load_tokenizer(self.paths.code_tokenizer_path)
        self.comment_tokenizer = load_tokenizer(self.paths.comment_tokenizer_path)

        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ArtifactError("TensorFlow is required to load the neural checkpoint.") from exc

        if self.paths.encoder_model_path.exists() and self.paths.decoder_model_path.exists():
            from src.attention_layer import BahdanauAttention
            custom = {"BahdanauAttention": BahdanauAttention}
            self.encoder_model = keras.models.load_model(
                self.paths.encoder_model_path, compile=False, custom_objects=custom
            )
            self.decoder_model = keras.models.load_model(
                self.paths.decoder_model_path, compile=False, custom_objects=custom
            )
            self.attention_enabled = True
        elif self.paths.attention_model_path.exists():
            from src.config import ModelConfig
            from src.seq2seq_model import build_attention_inference_models, load_attention_model
            config_keys = set(ModelConfig.__dataclass_fields__)
            config = ModelConfig(**{
                key: value for key, value in self.metadata.get("model_config", {}).items() if key in config_keys
            })
            full_model = load_attention_model(self.paths.attention_model_path)
            self.encoder_model, self.decoder_model = build_attention_inference_models(full_model, config)
            self.attention_enabled = True
        else:
            from src.seq2seq_model import build_legacy_inference_models
            full_model = keras.models.load_model(self.paths.legacy_model_path, compile=False)
            self.encoder_model, self.decoder_model = build_legacy_inference_models(full_model)
            self.attention_enabled = False

        self.loaded = True
        return self

    def generate(
        self,
        code: str,
        *,
        method: str = "greedy",
        beam_width: int = 3,
    ) -> GenerationResult:
        if not self.loaded:
            self.load()
        mode = self.metadata.get("preprocessing_mode", "legacy")
        max_code_len = int(self.metadata.get("max_code_len", 180))
        max_comment_len = int(self.metadata.get("max_comment_len", 30))
        cleaned = preprocess_code(
            code,
            CodePreprocessingOptions(max_tokens=max_code_len),
            mode=mode,
        )
        encoder_input = prepare_single_code(cleaned, self.code_tokenizer, max_code_len=max_code_len)
        common = dict(
            encoder_model=self.encoder_model,
            decoder_model=self.decoder_model,
            encoder_input=encoder_input,
            comment_tokenizer=self.comment_tokenizer,
            max_tokens=max_comment_len - 1,
            start_token=self.metadata.get("start_token", "<start>"),
            end_token=self.metadata.get("end_token", "<end>"),
            attention_enabled=self.attention_enabled,
        )
        if method == "beam":
            return beam_search_decode(**common, beam_width=beam_width)
        return greedy_decode(**common)
