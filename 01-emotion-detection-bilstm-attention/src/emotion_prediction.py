"""Prediction data structures and plain-language interpretation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionPrediction:
    input_text: str
    predicted_emotion: str
    confidence: float
    probabilities: dict[str, float]
    important_tokens: list[tuple[str, float]]
    artifact_status: str

    @property
    def top_probabilities(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)

    def interpretation(self) -> str:
        confidence_description = (
            "high" if self.confidence >= 0.80 else "moderate" if self.confidence >= 0.55 else "low"
        )
        return (
            f"The model's highest estimated class is {self.predicted_emotion.title()} "
            f"with {confidence_description} confidence. Review the probability distribution "
            "and the original context before drawing conclusions."
        )
