from pathlib import Path

from src.inference_pipeline import NERInferencePipeline


def test_artifact_metadata_loads_without_loading_tensorflow_model() -> None:
    model_dir = Path(__file__).resolve().parents[1] / "models"
    pipeline = NERInferencePipeline(model_dir)
    assert pipeline.max_length == 124
    assert pipeline.tag_to_index["B-PER"] == 1
    assert pipeline.word_to_index["<UNK>"] == 1
