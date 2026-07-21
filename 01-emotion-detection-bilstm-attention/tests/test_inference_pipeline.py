from pathlib import Path

import pytest

from src.inference_pipeline import ArtifactError, EmotionInferencePipeline


def test_missing_artifacts_raise_clear_error(tmp_path: Path):
    pipeline = EmotionInferencePipeline(tmp_path)
    with pytest.raises(ArtifactError, match="No complete model artifact set"):
        pipeline._resolve_artifacts()


def test_legacy_artifacts_are_detected(tmp_path: Path):
    for name in [
        "legacy_emotion_bilstm_model.keras",
        "legacy_tokenizer_config.json",
        "legacy_label_mapping.json",
    ]:
        (tmp_path / name).write_text("placeholder", encoding="utf-8")
    model_path, tokenizer_path, labels_path, _ = EmotionInferencePipeline(tmp_path)._resolve_artifacts()
    assert model_path.name.startswith("legacy_")
    assert tokenizer_path.name == "legacy_tokenizer_config.json"
    assert labels_path.name == "legacy_label_mapping.json"
