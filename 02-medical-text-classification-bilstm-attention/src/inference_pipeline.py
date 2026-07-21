from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .artifacts import load_label_mapping, load_model_metadata
from .config import (
    DEFAULT_LABEL_MAPPING_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_PATH,
)
from .medical_text_prediction import PredictionResult, rank_probabilities
from .medical_text_preprocessing import clean_medical_text
from .sequence_generation import pad_integer_sequences


class MedicalTextInferencePipeline:
    """Load saved artifacts once and provide single/batch inference."""

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
        label_mapping_path: str | Path = DEFAULT_LABEL_MAPPING_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.label_mapping_path = Path(label_mapping_path)
        self.metadata_path = Path(metadata_path)

        self.label_mapping = load_label_mapping(self.label_mapping_path)
        self.metadata = load_model_metadata(self.metadata_path)
        self._validate_artifact_contract()

        self.model = None
        self.tokenizer = None
        self._attention_probe = None

    def _validate_artifact_contract(self) -> None:
        missing = [
            path
            for path in (
                self.model_path,
                self.tokenizer_path,
                self.label_mapping_path,
                self.metadata_path,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required model artifacts: "
                + ", ".join(str(path) for path in missing)
            )

        metadata_labels = list(self.metadata["class_labels"])
        mapping_labels = [
            self.label_mapping[index]
            for index in sorted(self.label_mapping)
        ]
        if metadata_labels != mapping_labels:
            raise ValueError(
                "Metadata class labels do not match label_mapping.json."
            )

    def load(self) -> "MedicalTextInferencePipeline":
        if self.model is not None and self.tokenizer is not None:
            return self

        import tensorflow as tf

        from .attention_layer import AttentionLayer
        from .tokenizer_utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.tokenizer_path)
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False,
        )
        return self

    @property
    def class_labels(self) -> list[str]:
        return [
            self.label_mapping[index]
            for index in sorted(self.label_mapping)
        ]

    def _prepare_texts(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        self.load()
        preprocessing_mode = str(
            self.metadata.get("preprocessing_mode", "legacy")
        )
        cleaned = [
            clean_medical_text(text, mode=preprocessing_mode)
            for text in texts
        ]
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_integer_sequences(
            sequences,
            max_length=int(self.metadata["max_sequence_length"]),
            padding="post",
            truncating="post",
        )
        return cleaned, padded

    def predict(
        self,
        text: str,
        *,
        top_k: int = 3,
        include_attention: bool = True,
    ) -> PredictionResult:
        if not str(text).strip():
            raise ValueError("Medical text must not be empty.")

        cleaned, padded = self._prepare_texts([str(text)])
        probability_vector = np.asarray(
            self.model.predict(padded, verbose=0)[0],
            dtype=float,
        )
        ranked = rank_probabilities(
            probability_vector,
            self.label_mapping,
            top_k=top_k,
        )
        predicted_label, confidence = ranked[0]

        important_terms: list[tuple[str, float]] = []
        if include_attention:
            important_terms = self._extract_attention_terms(
                cleaned_text=cleaned[0],
                padded_sequence=padded,
            )

        return PredictionResult(
            input_text=str(text),
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities={
                self.label_mapping[index]: float(probability_vector[index])
                for index in sorted(self.label_mapping)
            },
            top_probabilities=ranked,
            important_terms=important_terms,
        )

    def predict_batch(
        self,
        texts: Iterable[str],
        *,
        top_k: int = 3,
    ) -> pd.DataFrame:
        text_list = [str(text) for text in texts]
        if not text_list:
            return pd.DataFrame()

        _, padded = self._prepare_texts(text_list)
        probability_matrix = np.asarray(
            self.model.predict(padded, verbose=0),
            dtype=float,
        )

        records: list[dict[str, object]] = []
        for text, probability_vector in zip(text_list, probability_matrix):
            ranked = rank_probabilities(
                probability_vector,
                self.label_mapping,
                top_k=top_k,
            )
            result = PredictionResult(
                input_text=text,
                predicted_label=ranked[0][0],
                confidence=ranked[0][1],
                probabilities={
                    self.label_mapping[index]: float(probability_vector[index])
                    for index in sorted(self.label_mapping)
                },
                top_probabilities=ranked,
                important_terms=[],
            )
            record = result.to_record()
            for label, probability in result.probabilities.items():
                record[f"probability__{label}"] = probability
            records.append(record)

        return pd.DataFrame(records)

    def _build_attention_probe(self):
        if self._attention_probe is not None:
            return self._attention_probe

        import tensorflow as tf

        embedding_layer = self.model.get_layer("embedding")
        bilstm_layer = self.model.get_layer("bilstm")
        attention_layer = self.model.get_layer("attention")

        sequence_input = self.model.input
        embeddings = embedding_layer(sequence_input)
        sequence_states = bilstm_layer(embeddings)
        self._attention_probe = tf.keras.Model(
            inputs=sequence_input,
            outputs=sequence_states,
        )
        return self._attention_probe, attention_layer

    def _extract_attention_terms(
        self,
        *,
        cleaned_text: str,
        padded_sequence: np.ndarray,
        top_terms: int = 10,
    ) -> list[tuple[str, float]]:
        try:
            probe_result = self._build_attention_probe()
            if isinstance(probe_result, tuple):
                probe, attention_layer = probe_result
            else:
                probe = probe_result
                attention_layer = self.model.get_layer("attention")

            sequence_states = probe.predict(padded_sequence, verbose=0)
            weights = attention_layer.compute_attention_weights(
                sequence_states
            ).numpy()[0, :, 0]

            token_ids = padded_sequence[0]
            scored_tokens: list[tuple[str, float]] = []
            for token_id, weight in zip(token_ids, weights):
                if int(token_id) == 0:
                    continue
                token = self.tokenizer.index_word.get(
                    int(token_id),
                    "<OOV>",
                )
                scored_tokens.append((str(token), float(weight)))

            # Aggregate repeated tokens so the displayed list is easier to read.
            aggregate: dict[str, float] = {}
            for token, weight in scored_tokens:
                aggregate[token] = aggregate.get(token, 0.0) + weight

            return sorted(
                aggregate.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_terms]
        except Exception:
            # Explainability is optional and should never block prediction.
            return []
