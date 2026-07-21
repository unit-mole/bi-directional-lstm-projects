from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pad_integer_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    max_length: int,
    padding: str = "post",
    truncating: str = "post",
    value: int = 0,
) -> np.ndarray:
    """Pad integer sequences without importing TensorFlow.

    This produces the same basic contract used by Keras ``pad_sequences`` for
    the supported pre/post combinations.
    """
    if max_length <= 0:
        raise ValueError("max_length must be greater than zero.")
    if padding not in {"pre", "post"}:
        raise ValueError("padding must be 'pre' or 'post'.")
    if truncating not in {"pre", "post"}:
        raise ValueError("truncating must be 'pre' or 'post'.")

    output = np.full((len(sequences), max_length), value, dtype=np.int32)

    for row_index, sequence in enumerate(sequences):
        values = list(sequence)
        if truncating == "pre":
            values = values[-max_length:]
        else:
            values = values[:max_length]

        if not values:
            continue

        if padding == "post":
            output[row_index, : len(values)] = values
        else:
            output[row_index, -len(values) :] = values

    return output
