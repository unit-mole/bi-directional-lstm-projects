import pytest

pytest.importorskip("tensorflow")

from src.sequence_generation import texts_to_padded_sequences
from src.tokenizer_utils import create_tokenizer


def test_sequences_have_fixed_shape_and_oov_support():
    tokenizer = create_tokenizer(["joyful result", "fearful result"], max_vocab_size=20)
    matrix = texts_to_padded_sequences(
        ["joyful result", "unseen phrase"], tokenizer, max_sequence_length=5
    )
    assert matrix.shape == (2, 5)
    assert matrix[1, 0] == tokenizer.word_index["<OOV>"]
