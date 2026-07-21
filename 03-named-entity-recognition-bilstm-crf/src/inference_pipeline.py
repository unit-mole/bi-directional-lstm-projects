"""Artifact-aware inference pipeline for true CRF and supplied legacy models."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .crf_layer import build_bio_constraints, viterbi_decode_numpy
from .entity_extraction import (
    extract_entities,
    highlighted_text_html,
    repair_bio_tags,
)
from .sequence_generation import chunk_sequence, prepare_inference_batch
from .tokenizer_utils import tokenize_with_offsets


@dataclass
class PredictionResult:
    text: str
    tokens: list[str]
    tags: list[str]
    confidences: list[float]
    entities: list[dict[str, object]]
    decoder: str
    model_kind: str

    def token_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "token_index": range(len(self.tokens)),
                "token": self.tokens,
                "predicted_tag": self.tags,
                "confidence": self.confidences,
            }
        )

    def entity_frame(self) -> pd.DataFrame:
        columns = [
            "entity_text",
            "entity_type",
            "start_token",
            "end_token",
            "char_start",
            "char_end",
            "confidence",
        ]
        return pd.DataFrame(self.entities).reindex(columns=columns)

    def highlighted_html(self) -> str:
        return highlighted_text_html(self.text, self.entities)


class NERInferencePipeline:
    """Load NER artifacts lazily and provide single/batch inference."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.model_dir = (
            Path(model_dir) if model_dir else project_root / "models"
        )

        metadata_path = self.model_dir / "model_metadata.json"
        word_index_path = self.model_dir / "word_to_index.pkl"
        tag_index_path = self.model_dir / "tag_to_index.pkl"

        required = [metadata_path, word_index_path, tag_index_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required NER metadata/vocabulary artifacts: "
                + ", ".join(missing)
            )

        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        with word_index_path.open("rb") as handle:
            self.word_to_index: dict[str, int] = pickle.load(handle)
        with tag_index_path.open("rb") as handle:
            self.tag_to_index: dict[str, int] = pickle.load(handle)

        self.index_to_tag = {
            index: tag for tag, index in self.tag_to_index.items()
        }
        self.max_length = int(self.metadata["max_sequence_length"])
        self.lowercase = bool(
            self.metadata.get("lowercase_tokens", True)
        )
        self.pad_token_id = int(
            self.metadata.get("pad_token_id", 0)
        )

        self._model = None
        self._model_kind: str | None = None

    def _model_candidates(self) -> tuple[Path, Path]:
        true_crf_name = self.metadata["true_crf_artifact"][
            "weights_filename"
        ]
        legacy_name = self.metadata["legacy_artifact"]["filename"]
        return (
            self.model_dir / true_crf_name,
            self.model_dir / legacy_name,
        )

    def model_artifact_status(self) -> dict[str, object]:
        """Report artifact availability without loading TensorFlow."""
        crf_weights, legacy_model = self._model_candidates()

        if crf_weights.is_file():
            selected_kind = "true_bilstm_crf"
            selected_path = crf_weights
        elif legacy_model.is_file():
            selected_kind = "legacy_bilstm_softmax"
            selected_path = legacy_model
        else:
            selected_kind = "missing"
            selected_path = None

        return {
            "available": selected_path is not None,
            "selected_model_kind": selected_kind,
            "selected_model_path": (
                str(selected_path) if selected_path is not None else None
            ),
            "true_crf_weights_present": crf_weights.is_file(),
            "legacy_model_present": legacy_model.is_file(),
            "expected_true_crf_path": str(crf_weights),
            "expected_legacy_model_path": str(legacy_model),
        }

    @property
    def model_kind(self) -> str:
        self._ensure_model_loaded()
        assert self._model_kind is not None
        return self._model_kind

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        crf_weights, legacy_path = self._model_candidates()

        if crf_weights.is_file():
            from .crf_layer import BiLSTMCRFTagger

            config = self.metadata.get("trained_crf_config", {})
            model = BiLSTMCRFTagger(
                vocab_size=len(self.word_to_index),
                num_tags=len(self.tag_to_index),
                max_sequence_length=self.max_length,
                pad_token_id=self.pad_token_id,
                embedding_dim=int(config.get("embedding_dim", 100)),
                lstm_units=int(config.get("lstm_units", 128)),
                dense_units=int(config.get("dense_units", 64)),
                dropout_rate=float(config.get("dropout_rate", 0.30)),
                name="bilstm_crf_tagger",
            )
            model(
                np.zeros((1, self.max_length), dtype=np.int32),
                training=False,
            )
            model.load_weights(crf_weights)
            self._model = model
            self._model_kind = "true_bilstm_crf"
            return

        if not legacy_path.is_file():
            raise FileNotFoundError(
                "The NER model artifact is missing. Expected either "
                f"'{crf_weights}' or '{legacy_path}'. Copy the supplied "
                "legacy_bilstm_softmax_model.h5 file into the models "
                "directory and push it to GitHub."
            )

        import tensorflow as tf

        try:
            self._model = tf.keras.models.load_model(
                legacy_path,
                compile=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "The legacy NER model exists but could not be loaded from "
                f"'{legacy_path}'. Confirm that Git LFS downloaded the real "
                "binary file instead of a pointer file."
            ) from exc

        self._model_kind = "legacy_bilstm_softmax"

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values - np.max(values, axis=-1, keepdims=True)
        exp = np.exp(values)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _predict_chunk(
        self,
        tokens: list[str],
    ) -> tuple[list[str], list[float], str]:
        self._ensure_model_loaded()
        batch, lengths = prepare_inference_batch(
            [tokens],
            self.word_to_index,
            self.max_length,
            self.lowercase,
        )
        length = int(lengths[0])
        assert self._model is not None

        if self._model_kind == "true_bilstm_crf":
            emissions = self._model(batch, training=False).numpy()
            transition_params = self._model.transition_params.numpy()
            decoded = viterbi_decode_numpy(
                emissions,
                transition_params,
                lengths,
            )[0]
            probabilities = self._softmax(emissions[0, :length])
            decoder = "trained linear-chain CRF Viterbi"
        else:
            probabilities = np.asarray(
                self._model.predict(batch, verbose=0)
            )[0, :length]
            if probabilities.ndim != 2:
                raise ValueError(
                    "Legacy model returned an unexpected prediction shape."
                )

            log_emissions = np.log(
                np.clip(probabilities, 1e-9, 1.0)
            )[None, :, :]
            constraints, start_scores = build_bio_constraints(
                self.index_to_tag
            )
            decoded = viterbi_decode_numpy(
                log_emissions,
                constraints,
                [length],
                start_scores=start_scores,
            )[0]
            decoder = (
                "BIO-constrained Viterbi over legacy softmax emissions"
            )

        tags = [
            self.index_to_tag[int(tag_id)]
            for tag_id in decoded
        ]
        confidences = [
            round(float(probabilities[index, tag_id]), 4)
            for index, tag_id in enumerate(decoded)
        ]
        return tags, confidences, decoder

    def predict_text(self, text: str) -> PredictionResult:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        spans = tokenize_with_offsets(text)
        if not spans:
            raise ValueError(
                "No tokens could be extracted from the input text."
            )

        tokens = [span.text for span in spans]
        tags: list[str] = []
        confidences: list[float] = []
        decoder = ""

        for _, _, chunk in chunk_sequence(tokens, self.max_length):
            chunk_tags, chunk_confidences, decoder = self._predict_chunk(
                list(chunk)
            )
            tags.extend(chunk_tags)
            confidences.extend(chunk_confidences)

        tags = repair_bio_tags(tags)
        offsets = [(span.start, span.end) for span in spans]
        entities = extract_entities(
            tokens,
            tags,
            confidences,
            offsets,
        )

        return PredictionResult(
            text=text,
            tokens=tokens,
            tags=tags,
            confidences=confidences,
            entities=entities,
            decoder=decoder,
            model_kind=self.model_kind,
        )

    def predict_batch(
        self,
        texts: Iterable[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        document_rows: list[dict[str, object]] = []
        entity_rows: list[dict[str, object]] = []

        for document_id, text in enumerate(texts):
            result = self.predict_text(str(text))
            document_rows.append(
                {
                    "document_id": document_id,
                    "text": result.text,
                    "entity_count": len(result.entities),
                    "model_kind": result.model_kind,
                    "decoder": result.decoder,
                }
            )
            for entity in result.entities:
                entity_rows.append(
                    {"document_id": document_id, **entity}
                )

        return pd.DataFrame(document_rows), pd.DataFrame(entity_rows)

    def artifact_summary(self) -> dict[str, object]:
        """Return app metadata without forcing the model to load."""
        status = self.model_artifact_status()
        return {
            "model_artifact_available": status["available"],
            "model_kind": status["selected_model_kind"],
            "model_path": status["selected_model_path"],
            "dataset": self.metadata["dataset"],
            "tagging_scheme": self.metadata["tagging_scheme"],
            "entity_types": self.metadata["entity_types"],
            "vocabulary_size": len(self.word_to_index),
            "maximum_sequence_length": self.max_length,
        }
