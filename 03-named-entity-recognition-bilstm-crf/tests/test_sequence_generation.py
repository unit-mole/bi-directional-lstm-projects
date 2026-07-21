import numpy as np

from src.sequence_generation import chunk_sequence, prepare_inference_batch


def test_padding_and_unknown_token() -> None:
    vocabulary = {"<PAD>": 0, "<UNK>": 1, "apple": 2}
    batch, lengths = prepare_inference_batch(
        [["Apple", "unknown"]], vocabulary, max_length=4, lowercase=True
    )
    assert batch.tolist() == [[2, 1, 0, 0]]
    assert lengths.tolist() == [2]


def test_chunk_sequence() -> None:
    chunks = chunk_sequence(list(range(7)), max_length=3)
    assert [(start, end) for start, end, _ in chunks] == [(0, 3), (3, 6), (6, 7)]
