"""Load model artifacts once and provide consistent single/batch inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .emotion_prediction import EmotionPrediction
from .sequence_generation import texts_to_padded_sequences
from .text_preprocessing import normalize_text
from .tokenizer_utils import load_tokenizer


class ArtifactError(RuntimeError):
    """Raised when required inference artifacts are missing or inconsistent."""


class EmotionInferencePipeline:
    """Reusable inference service for Streamlit and scripts.

    The primary path is a newly trained attention model. When that model is not
    present, the pipeline can transparently load the supplied legacy checkpoint.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model = None
        self.tokenizer = None
        self.label_mapping: dict[str, str] = {}
        self.metadata: dict[str, object] = {}
        self.artifact_status = "unloaded"
        self.max_sequence_length = 60
        self._attention_model = None

    def _resolve_artifacts(self) -> tuple[Path, Path, Path, Path | None]:
        primary = (
            self.model_dir / "emotion_bilstm_attention_model.keras",
            self.model_dir / "tokenizer.json",
            self.model_dir / "label_mapping.json",
            self.model_dir / "model_metadata.json",
        )
        if all(path.exists() for path in primary[:3]):
            return primary

        legacy = (
            self.model_dir / "legacy_emotion_bilstm_model.keras",
            self.model_dir / "legacy_tokenizer_config.json",
            self.model_dir / "legacy_label_mapping.json",
            self.model_dir / "legacy_model_metadata.json",
        )
        if all(path.exists() for path in legacy[:3]):
            return legacy
        raise ArtifactError(
            "No complete model artifact set was found. Train the attention model with "
            "`python scripts/train_model.py` or restore the legacy demo artifacts."
        )

    def load(self) -> "EmotionInferencePipeline":
        import tensorflow as tf

        from .attention_layer import TemporalAttention

        model_path, tokenizer_path, labels_path, metadata_path = self._resolve_artifacts()
        custom_objects = {"TemporalAttention": TemporalAttention}
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.label_mapping = json.loads(labels_path.read_text(encoding="utf-8"))
        self.metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path and metadata_path.exists()
            else {}
        )
        self.max_sequence_length = int(self.metadata.get("max_sequence_length", 60))
        self.artifact_status = str(self.metadata.get("artifact_status", "unknown"))

        try:
            attention_layer = self.model.get_layer("temporal_attention")
            self._attention_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[self.model.output, attention_layer.output[1]],
            )
        except (ValueError, IndexError, TypeError):
            self._attention_model = None
        return self

    @property
    def supports_attention(self) -> bool:
        return self._attention_model is not None

    @property
    def classes(self) -> list[str]:
        return [self.label_mapping[str(index)] for index in sorted(map(int, self.label_mapping))]

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.load()

    def predict(self, text: str, top_tokens: int = 8) -> EmotionPrediction:
        results = self.predict_many([text], top_tokens=top_tokens)
        return results[0]

    def predict_many(
        self,
        texts: Iterable[str],
        top_tokens: int = 8,
        batch_size: int = 64,
    ) -> list[EmotionPrediction]:
        self._ensure_loaded()
        original_texts = [str(text) for text in texts]
        cleaned = [normalize_text(text) for text in original_texts]
        if any(not value for value in cleaned):
            raise ValueError("Every input must contain non-empty text after preprocessing.")

        padded = texts_to_padded_sequences(cleaned, self.tokenizer, self.max_sequence_length)
        if self._attention_model is not None:
            probabilities, attention_scores = self._attention_model.predict(
                padded, batch_size=batch_size, verbose=0
            )
        else:
            probabilities = self.model.predict(padded, batch_size=batch_size, verbose=0)
            attention_scores = None

        results: list[EmotionPrediction] = []
        for row_index, (raw_text, clean_text, probability_row) in enumerate(
            zip(original_texts, cleaned, np.asarray(probabilities))
        ):
            predicted_id = int(np.argmax(probability_row))
            probability_map = {
                self.label_mapping[str(index)]: float(probability)
                for index, probability in enumerate(probability_row)
            }
            important_tokens: list[tuple[str, float]] = []
            if attention_scores is not None:
                token_ids = self.tokenizer.texts_to_sequences([clean_text])[0][
                    : self.max_sequence_length
                ]
                tokens = [self.tokenizer.index_word.get(token_id, "<OOV>") for token_id in token_ids]
                scores = np.asarray(attention_scores[row_index])[: len(tokens)]
                important_tokens = sorted(
                    zip(tokens, map(float, scores)), key=lambda item: item[1], reverse=True
                )[:top_tokens]

            results.append(
                EmotionPrediction(
                    input_text=raw_text,
                    predicted_emotion=self.label_mapping[str(predicted_id)],
                    confidence=float(np.max(probability_row)),
                    probabilities=probability_map,
                    important_tokens=important_tokens,
                    artifact_status=self.artifact_status,
                )
            )
        return results
