from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pad_token_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    max_length: int,
    padding: str = "post",
    truncating: str = "post",
    value: int = 0,
) -> np.ndarray:
    """Small NumPy equivalent of Keras pad_sequences for portable inference/tests."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if padding not in {"pre", "post"} or truncating not in {"pre", "post"}:
        raise ValueError("padding and truncating must be 'pre' or 'post'")
    result = np.full((len(sequences), max_length), value, dtype=np.int32)
    for row_index, sequence in enumerate(sequences):
        seq = list(sequence)
        seq = seq[-max_length:] if truncating == "pre" else seq[:max_length]
        if not seq:
            continue
        if padding == "post":
            result[row_index, : len(seq)] = seq
        else:
            result[row_index, -len(seq) :] = seq
    return result


def texts_to_padded_sequences(tokenizer, texts: Sequence[str], *, max_length: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(list(texts))
    return pad_token_sequences(sequences, max_length=max_length, padding="post", truncating="post")
