from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SequenceBatch:
    encoder_inputs: np.ndarray
    decoder_inputs: np.ndarray
    decoder_targets: np.ndarray


def _pad_sequences(sequences, *, maxlen: int):
    try:
        from keras.utils import pad_sequences
    except ImportError:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")


def prepare_seq2seq_arrays(
    code_texts: list[str],
    comment_sequences: list[str],
    code_tokenizer: Any,
    comment_tokenizer: Any,
    *,
    max_code_len: int,
    max_comment_len: int,
) -> SequenceBatch:
    code_ids = code_tokenizer.texts_to_sequences(code_texts)
    comment_ids = comment_tokenizer.texts_to_sequences(comment_sequences)
    encoder = _pad_sequences(code_ids, maxlen=max_code_len)
    target_full = _pad_sequences(comment_ids, maxlen=max_comment_len)
    return SequenceBatch(
        encoder_inputs=np.asarray(encoder, dtype=np.int32),
        decoder_inputs=np.asarray(target_full[:, :-1], dtype=np.int32),
        decoder_targets=np.asarray(target_full[:, 1:], dtype=np.int32),
    )


def prepare_single_code(
    cleaned_code: str,
    tokenizer: Any,
    *,
    max_code_len: int,
) -> np.ndarray:
    sequence = tokenizer.texts_to_sequences([cleaned_code])
    return np.asarray(_pad_sequences(sequence, maxlen=max_code_len), dtype=np.int32)
