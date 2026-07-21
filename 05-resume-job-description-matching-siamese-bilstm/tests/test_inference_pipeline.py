from dataclasses import replace
from pathlib import Path

from src.config import CONFIG
from src.inference_pipeline import ResumeJobMatcher


def test_fallback_inference_returns_complete_result(tmp_path: Path):
    config = replace(CONFIG, project_dir=tmp_path)
    matcher = ResumeJobMatcher(config=config, allow_fallback=True)
    result = matcher.predict(
        "Data scientist with Python SQL machine learning and NLP experience.",
        "Seeking a machine learning engineer with Python NLP SQL Docker and model deployment.",
    )
    assert result["model_loaded"] is False
    assert 0.0 <= result["fit_score"] <= 1.0
    assert result["score_band"] in {"Weak Match", "Moderate Match", "Strong Match"}
    assert "Python" in result["overlapping_skills"]
