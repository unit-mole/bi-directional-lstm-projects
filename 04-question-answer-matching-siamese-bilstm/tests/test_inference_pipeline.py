import numpy as np

from src.inference_pipeline import QAMatcher


class FakeTokenizer:
    def texts_to_sequences(self, texts):
        return [[1, 2] if text else [] for text in texts]


class FakeModel:
    def predict(self, inputs, verbose=0):
        batch_size = inputs[0].shape[0]
        return np.full((batch_size, 1), 0.8)


def test_prediction_result_with_injected_artifacts():
    matcher = QAMatcher(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        metadata={"max_sequence_length": 4, "prediction_threshold": 0.5},
    )
    result = matcher.predict_pair("learn python", "python tutorial")
    assert result.predicted_label == "Match"
    assert result.match_probability == 0.8
    assert result.confidence == 0.8


def test_rank_candidates_sorts_probability():
    matcher = QAMatcher(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        metadata={"max_sequence_length": 4, "prediction_threshold": 0.5},
    )
    ranked = matcher.rank_candidates("question", ["answer one", "answer two"])
    assert ranked["rank"].tolist() == [1, 2]
