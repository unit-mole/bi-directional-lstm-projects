from pathlib import Path

import pytest

from src.artifacts import load_label_mapping, load_model_metadata
from src.config import (
    DEFAULT_LABEL_MAPPING_PATH,
    DEFAULT_METADATA_PATH,
)
from src.medical_text_prediction import rank_probabilities


def test_label_mapping_uses_five_actual_classes() -> None:
    mapping = load_label_mapping(DEFAULT_LABEL_MAPPING_PATH)
    assert mapping == {
        0: "Cardiology",
        1: "Gastroenterology",
        2: "Neurology",
        3: "Orthopedic",
        4: "Radiology",
    }


def test_metadata_contract_matches_mapping() -> None:
    mapping = load_label_mapping(DEFAULT_LABEL_MAPPING_PATH)
    metadata = load_model_metadata(DEFAULT_METADATA_PATH)
    assert metadata["class_labels"] == [
        mapping[index] for index in sorted(mapping)
    ]
    assert metadata["max_sequence_length"] == 300


def test_probability_ranking() -> None:
    mapping = {0: "A", 1: "B", 2: "C"}
    ranked = rank_probabilities([0.1, 0.7, 0.2], mapping, top_k=2)
    assert ranked == [("B", 0.7), ("C", 0.2)]


@pytest.mark.model
def test_saved_model_can_load_when_tensorflow_is_available() -> None:
    pytest.importorskip("tensorflow")
    from src.inference_pipeline import MedicalTextInferencePipeline

    pipeline = MedicalTextInferencePipeline().load()
    assert pipeline.model is not None
    assert pipeline.tokenizer is not None
