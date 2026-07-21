from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sequence_generation import texts_to_padded_sequences
from .text_preprocessing import clean_text, token_overlap
from .tokenizer_utils import load_tokenizer


class ModelArtifactError(RuntimeError):
    pass


@dataclass
class PredictionResult:
    text_a: str
    text_b: str
    predicted_label: str
    predicted_class: int
    match_probability: float
    confidence: float
    threshold: float
    interpretation: str
    shared_tokens: list[str]
    lexical_jaccard: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text_a": self.text_a,
            "text_b": self.text_b,
            "predicted_label": self.predicted_label,
            "predicted_class": self.predicted_class,
            "match_probability": self.match_probability,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "interpretation": self.interpretation,
            "shared_tokens": ", ".join(self.shared_tokens),
            "lexical_jaccard": self.lexical_jaccard,
        }


class QAMatcher:
    """Inference wrapper for semantic text-pair matching.

    The committed model was trained on duplicate-question labels. Treat question-answer
    relevance scoring as a transfer/demo use case unless the model is retrained on QA data.
    """

    def __init__(self, model, tokenizer, metadata: dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.max_length = int(metadata.get("max_sequence_length", 40))
        self.threshold = float(metadata.get("prediction_threshold", 0.5))

    @staticmethod
    def _custom_objects():
        import tensorflow as tf

        @tf.keras.utils.register_keras_serializable(package="Custom", name="abs_diff_fn")
        def abs_diff_fn(tensors):
            return tf.abs(tensors[0] - tensors[1])

        @tf.keras.utils.register_keras_serializable(package="Custom", name="multiply_fn")
        def multiply_fn(tensors):
            return tensors[0] * tensors[1]

        def pairwise_output_shape(input_shapes):
            return input_shapes[0]

        return {
            "abs_diff_fn": abs_diff_fn,
            "multiply_fn": multiply_fn,
            "Custom>abs_diff_fn": abs_diff_fn,
            "Custom>multiply_fn": multiply_fn,
            "pairwise_output_shape": pairwise_output_shape,
        }

    @classmethod
    def from_artifacts(cls, model_dir: str | Path) -> "QAMatcher":
        model_dir = Path(model_dir)
        model_path = model_dir / "qa_siamese_bilstm_model.keras"
        tokenizer_path = model_dir / "tokenizer.json"
        metadata_path = model_dir / "model_metadata.json"
        missing = [str(path) for path in (model_path, tokenizer_path, metadata_path) if not path.exists()]
        if missing:
            raise ModelArtifactError(f"Missing required model artifacts: {missing}")
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=cls._custom_objects(),
                compile=False,
                safe_mode=False,
            )
            tokenizer = load_tokenizer(tokenizer_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelArtifactError(f"Unable to load model artifacts: {exc}") from exc
        return cls(model=model, tokenizer=tokenizer, metadata=metadata)

    def _probabilities(self, text_a: list[str], text_b: list[str]) -> np.ndarray:
        cleaned_a = [clean_text(text) for text in text_a]
        cleaned_b = [clean_text(text) for text in text_b]
        a_padded = texts_to_padded_sequences(self.tokenizer, cleaned_a, max_length=self.max_length)
        b_padded = texts_to_padded_sequences(self.tokenizer, cleaned_b, max_length=self.max_length)
        probabilities = self.model.predict([a_padded, b_padded], verbose=0)
        return np.asarray(probabilities, dtype=float).reshape(-1)

    def predict_pair(self, text_a: str, text_b: str) -> PredictionResult:
        if not clean_text(text_a) or not clean_text(text_b):
            raise ValueError("Both text inputs must contain meaningful text.")
        probability = float(self._probabilities([text_a], [text_b])[0])
        predicted_class = int(probability >= self.threshold)
        confidence = probability if predicted_class == 1 else 1.0 - probability
        diagnostics = token_overlap(text_a, text_b)
        if probability >= 0.75:
            interpretation = "The model found comparatively strong semantic alignment."
        elif probability >= self.threshold:
            interpretation = "The model found limited-to-moderate evidence of a semantic match."
        elif probability >= 0.35:
            interpretation = "The pair is borderline; review it manually."
        else:
            interpretation = "The model found little evidence that the texts express the same intent."
        return PredictionResult(
            text_a=text_a,
            text_b=text_b,
            predicted_label="Match" if predicted_class else "No Match",
            predicted_class=predicted_class,
            match_probability=probability,
            confidence=float(confidence),
            threshold=self.threshold,
            interpretation=interpretation,
            shared_tokens=list(diagnostics["shared_tokens"]),
            lexical_jaccard=float(diagnostics["jaccard_similarity"]),
        )

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not {"text_a", "text_b"}.issubset(frame.columns):
            raise ValueError("Input frame must contain text_a and text_b columns.")
        probabilities = self._probabilities(frame["text_a"].astype(str).tolist(), frame["text_b"].astype(str).tolist())
        output = frame.copy()
        output["match_probability"] = probabilities
        output["predicted_class"] = (probabilities >= self.threshold).astype(int)
        output["predicted_label"] = output["predicted_class"].map({0: "No Match", 1: "Match"})
        output["confidence"] = np.where(output["predicted_class"] == 1, probabilities, 1 - probabilities)
        return output

    def rank_candidates(self, question: str, candidates: list[str]) -> pd.DataFrame:
        candidates = [candidate.strip() for candidate in candidates if candidate.strip()]
        if not candidates:
            raise ValueError("Provide at least one candidate text.")
        frame = pd.DataFrame({"text_a": [question] * len(candidates), "text_b": candidates})
        scored = self.predict_frame(frame)
        scored.insert(0, "rank", scored["match_probability"].rank(method="first", ascending=False).astype(int))
        return scored.sort_values("rank").reset_index(drop=True)
