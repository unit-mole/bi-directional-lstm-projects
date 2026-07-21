"""Text-to-sequence conversion shared by training and inference."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def texts_to_padded_sequences(
    texts: Iterable[str],
    tokenizer: Any,
    max_sequence_length: int,
) -> np.ndarray:
    """Convert texts to fixed-length sequences with a lazy TensorFlow import."""

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be greater than zero.")
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences(list(texts))
    return pad_sequences(
        sequences,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
        value=0,
    )
