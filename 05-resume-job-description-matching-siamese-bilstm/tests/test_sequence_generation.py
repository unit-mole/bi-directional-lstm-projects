import numpy as np

from src.sequence_generation import prepare_pair_inputs


class TinyTokenizer:
    def texts_to_sequences(self, texts):
        return [[len(token) for token in text.split()] for text in texts]


def test_prepare_pair_inputs_shapes_and_padding():
    resume, job = prepare_pair_inputs(
        TinyTokenizer(),
        ["one two", "three"],
        ["alpha", "beta gamma delta"],
        max_length=4,
    )
    assert resume.shape == (2, 4)
    assert job.shape == (2, 4)
    assert resume.dtype == np.int32
    assert resume[0, -1] == 0
