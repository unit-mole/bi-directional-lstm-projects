import numpy as np
import pytest

from src.sequence_generation import pad_integer_sequences


def test_post_padding_and_truncation() -> None:
    result = pad_integer_sequences(
        [[1, 2], [3, 4, 5, 6]],
        max_length=3,
        padding="post",
        truncating="post",
    )
    np.testing.assert_array_equal(
        result,
        np.array([[1, 2, 0], [3, 4, 5]], dtype=np.int32),
    )


def test_pre_padding_and_truncation() -> None:
    result = pad_integer_sequences(
        [[1, 2], [3, 4, 5, 6]],
        max_length=3,
        padding="pre",
        truncating="pre",
    )
    np.testing.assert_array_equal(
        result,
        np.array([[0, 1, 2], [4, 5, 6]], dtype=np.int32),
    )


def test_invalid_max_length_raises() -> None:
    with pytest.raises(ValueError):
        pad_integer_sequences([[1]], max_length=0)
