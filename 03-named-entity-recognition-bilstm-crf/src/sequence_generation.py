"""Sequence preparation utilities shared by training and inference."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .tokenizer_utils import normalize_token


def encode_tokens(
    tokens: Sequence[str],
    word_to_index: dict[str, int],
    lowercase: bool = True,
    unknown_token: str = "<UNK>",
) -> list[int]:
    unknown_id = word_to_index[unknown_token]
    return [
        word_to_index.get(normalize_token(token, lowercase), unknown_id)
        for token in tokens
    ]


def prepare_inference_batch(
    token_sequences: Sequence[Sequence[str]],
    word_to_index: dict[str, int],
    max_length: int,
    lowercase: bool = True,
    pad_token: str = "<PAD>",
    unknown_token: str = "<UNK>",
) -> tuple[np.ndarray, np.ndarray]:
    pad_id = word_to_index[pad_token]
    batch = np.full((len(token_sequences), max_length), pad_id, dtype=np.int32)
    lengths = np.zeros(len(token_sequences), dtype=np.int32)
    for row, tokens in enumerate(token_sequences):
        ids = encode_tokens(tokens, word_to_index, lowercase, unknown_token)[:max_length]
        batch[row, : len(ids)] = ids
        lengths[row] = len(ids)
    return batch, lengths


def chunk_sequence(
    items: Sequence[object],
    max_length: int,
) -> list[tuple[int, int, Sequence[object]]]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    return [
        (start, min(start + max_length, len(items)), items[start : start + max_length])
        for start in range(0, len(items), max_length)
    ]
