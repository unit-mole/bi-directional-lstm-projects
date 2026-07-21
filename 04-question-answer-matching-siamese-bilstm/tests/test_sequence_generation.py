import numpy as np

from src.sequence_generation import pad_token_sequences


def test_post_padding_and_truncation():
    actual = pad_token_sequences([[1, 2], [3, 4, 5, 6]], max_length=3)
    expected = np.array([[1, 2, 0], [3, 4, 5]], dtype=np.int32)
    np.testing.assert_array_equal(actual, expected)
