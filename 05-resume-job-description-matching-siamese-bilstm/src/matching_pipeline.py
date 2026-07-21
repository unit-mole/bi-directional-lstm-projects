from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import CONFIG, ProjectConfig
from .sequence_generation import prepare_pair_inputs
from .text_preprocessing import clean_text, compare_skills
from .tokenizer_utils import load_tokenizer


RESPONSIBLE_USE = (
    "Educational portfolio demonstration only. Do not use this output as the sole basis "
    "for hiring, rejection, promotion, compensation, immigration, or legal decisions."
)


class ResumeJobMatcher:
    def __init__(
        self,
        *,
        config: ProjectConfig = CONFIG,
        allow_fallback: bool = True,
    ) -> None:
        self.config = config
        self.allow_fallback = allow_fallback
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.metadata = self._load_metadata()
        self.load_error: str | None = None
        self._load_artifacts()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _load_metadata(self) -> dict[str, Any]:
        if self.config.metadata_path.exists():
            return json.loads(self.config.metadata_path.read_text(encoding="utf-8"))
        return {
            "prediction_threshold": self.config.default_threshold,
            "blend_weights": {"neural": 0.35, "tfidf": 0.35, "skill_overlap": 0.30},
            "tokenization": {"max_sequence_length": self.config.max_sequence_length},
        }

    def _load_artifacts(self) -> None:
        try:
            if not self.config.model_path.exists() or not self.config.tokenizer_path.exists():
                raise FileNotFoundError("Model or tokenizer artifact is missing.")
            from .siamese_model import load_siamese_model
            self.model = load_siamese_model(self.config.model_path, compile_model=False)
            self.tokenizer = load_tokenizer(self.config.tokenizer_path)
        except Exception as exc:  # Fallback keeps the public demo honest and available.
            self.load_error = f"{type(exc).__name__}: {exc}"
            if not self.allow_fallback:
                raise

    def predict(self, resume_text: object, job_description: object) -> dict[str, Any]:
        resume = clean_text(resume_text)
        job = clean_text(job_description)
        if not resume or not job:
            raise ValueError("Both resume text and job-description text are required.")

        explanations = compare_skills(resume, job)
        tfidf_score = self._tfidf_similarity(resume, job)
        skill_score = float(explanations["skill_coverage"])

        neural_probability: float | None = None
        if self.model_loaded:
            max_length = int(self.metadata.get("tokenization", {}).get("max_sequence_length", self.config.max_sequence_length))
            resume_array, job_array = prepare_pair_inputs(
                self.tokenizer,
                [resume],
                [job],
                max_length=max_length,
            )
            model_output = self.model([resume_array, job_array], training=False)
            if hasattr(model_output, "detach"):
                output_array = model_output.detach().cpu().numpy()
            elif hasattr(model_output, "numpy"):
                output_array = model_output.numpy()
            else:
                output_array = np.asarray(model_output)
            neural_probability = float(np.asarray(output_array).reshape(-1)[0])

        if neural_probability is None:
            fit_score = 0.65 * tfidf_score + 0.35 * skill_score
            inference_mode = "transparent fallback"
        else:
            weights = self.metadata.get("blend_weights", {"neural": 0.35, "tfidf": 0.35, "skill_overlap": 0.30})
            fit_score = (
                float(weights.get("neural", 0.35)) * neural_probability
                + float(weights.get("tfidf", 0.35)) * tfidf_score
                + float(weights.get("skill_overlap", 0.30)) * skill_score
            )
            inference_mode = "Siamese BiLSTM + transparent supporting signals"

        fit_score = float(np.clip(fit_score, 0.0, 1.0))
        threshold = float(self.metadata.get("prediction_threshold", self.config.default_threshold))
        predicted_label = int(fit_score >= threshold)
        score_band = self._score_band(fit_score)

        return {
            "predicted_label": predicted_label,
            "prediction": "Match" if predicted_label else "No Match",
            "match_probability": neural_probability,
            "fit_score": fit_score,
            "fit_score_percent": fit_score * 100,
            "threshold": threshold,
            "score_band": score_band,
            "confidence": abs(fit_score - threshold) / max(threshold, 1 - threshold),
            "tfidf_similarity": tfidf_score,
            "skill_coverage": skill_score,
            "overlapping_skills": explanations["overlapping_skills"],
            "missing_skills": explanations["missing_skills"],
            "resume_skills": explanations["resume_skills"],
            "job_skills": explanations["job_skills"],
            "interpretation": self._interpretation(score_band, explanations["overlapping_skills"], explanations["missing_skills"]),
            "responsible_use": RESPONSIBLE_USE,
            "inference_mode": inference_mode,
            "model_loaded": self.model_loaded,
            "load_error": self.load_error,
        }

    @staticmethod
    def _tfidf_similarity(resume: str, job: str) -> float:
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform([resume, job])
        return float(cosine_similarity(matrix[0], matrix[1])[0, 0])

    @staticmethod
    def _score_band(score: float) -> str:
        if score >= 0.70:
            return "Strong Match"
        if score >= 0.40:
            return "Moderate Match"
        return "Weak Match"

    @staticmethod
    def _interpretation(band: str, overlapping: list[str], missing: list[str]) -> str:
        overlap_text = ", ".join(overlapping[:6]) if overlapping else "no cataloged skill overlap"
        missing_text = ", ".join(missing[:6]) if missing else "no cataloged requirement gaps"
        return (
            f"{band}: the texts show {overlap_text}. The transparent skill check identified "
            f"{missing_text}. This is a semantic-text comparison, not a determination of candidate quality."
        )
