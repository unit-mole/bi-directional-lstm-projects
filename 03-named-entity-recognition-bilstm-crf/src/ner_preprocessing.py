"""Vocabulary building, encoding, padding, masking, and artifact persistence."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data_preprocessing import NERSentence
from .tokenizer_utils import normalize_token


def validate_alignment(tokens: Sequence[str], tags: Sequence[str]) -> None:
    if len(tokens) != len(tags):
        raise ValueError(f"Token/tag alignment error: {len(tokens)} tokens vs {len(tags)} tags")


def build_word_vocabulary(
    sentences: Iterable[NERSentence],
    max_size: int = 30_000,
    min_frequency: int = 1,
    lowercase: bool = True,
    pad_token: str = "<PAD>",
    unknown_token: str = "<UNK>",
) -> dict[str, int]:
    counts = Counter(
        normalize_token(token, lowercase)
        for sentence in sentences
        for token in sentence.tokens
    )
    ordered = sorted(
        (item for item in counts.items() if item[1] >= min_frequency),
        key=lambda item: (-item[1], item[0]),
    )
    if max_size < 2:
        raise ValueError("max_size must reserve space for PAD and UNK.")
    vocabulary = {pad_token: 0, unknown_token: 1}
    for token, _ in ordered[: max_size - 2]:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def build_tag_vocabulary(
    sentences: Iterable[NERSentence],
    preferred_order: Sequence[str] | None = None,
) -> dict[str, int]:
    observed = {tag for sentence in sentences for tag in sentence.tags}
    if preferred_order:
        ordered = [tag for tag in preferred_order if tag in observed]
        ordered.extend(sorted(observed.difference(ordered)))
    else:
        ordered = ["O"] if "O" in observed else []
        ordered.extend(sorted(observed.difference(ordered)))
    return {tag: index for index, tag in enumerate(ordered)}


def encode_sentences(
    sentences: Iterable[NERSentence],
    word_to_index: dict[str, int],
    tag_to_index: dict[str, int],
    lowercase: bool = True,
    pad_token: str = "<PAD>",
    unknown_token: str = "<UNK>",
) -> tuple[list[list[int]], list[list[int]]]:
    if pad_token not in word_to_index or unknown_token not in word_to_index:
        raise ValueError("Vocabulary must contain PAD and UNK tokens.")
    unknown_id = word_to_index[unknown_token]
    token_ids: list[list[int]] = []
    tag_ids: list[list[int]] = []
    for sentence in sentences:
        validate_alignment(sentence.tokens, sentence.tags)
        token_ids.append([
            word_to_index.get(normalize_token(token, lowercase), unknown_id)
            for token in sentence.tokens
        ])
        try:
            tag_ids.append([tag_to_index[tag] for tag in sentence.tags])
        except KeyError as exc:
            raise ValueError(f"Unknown tag encountered during encoding: {exc.args[0]}") from exc
    return token_ids, tag_ids


def pad_token_sequences(
    sequences: Sequence[Sequence[int]],
    max_length: int,
    pad_value: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-pad token IDs and return padded IDs, lengths, and boolean mask."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    padded = np.full((len(sequences), max_length), pad_value, dtype=np.int32)
    lengths = np.zeros(len(sequences), dtype=np.int32)
    for row, sequence in enumerate(sequences):
        clipped = list(sequence[:max_length])
        padded[row, : len(clipped)] = clipped
        lengths[row] = len(clipped)
    mask = np.arange(max_length)[None, :] < lengths[:, None]
    return padded, lengths, mask


def pad_tag_sequences(
    sequences: Sequence[Sequence[int]],
    max_length: int,
    pad_value: int,
) -> np.ndarray:
    padded, _, _ = pad_token_sequences(sequences, max_length, pad_value)
    return padded


def save_pickle(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> object:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
