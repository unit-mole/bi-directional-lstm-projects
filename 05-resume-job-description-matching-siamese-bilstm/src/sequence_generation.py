from __future__ import annotations

from typing import Any

import numpy as np

from .tokenizer_utils import _keras_text_tools


def texts_to_padded_sequences(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
) -> np.ndarray:
    _, _, pad_sequences = _keras_text_tools()
    sequences = tokenizer.texts_to_sequences(texts)
    return np.asarray(
        pad_sequences(
            sequences,
            maxlen=max_length,
            padding="post",
            truncating="post",
            value=0,
        ),
        dtype="int32",
    )


def prepare_pair_inputs(
    tokenizer: Any,
    resumes: list[str],
    job_descriptions: list[str],
    *,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(resumes) != len(job_descriptions):
        raise ValueError("Resume and job-description collections must have equal length.")
    return (
        texts_to_padded_sequences(tokenizer, resumes, max_length=max_length),
        texts_to_padded_sequences(tokenizer, job_descriptions, max_length=max_length),
    )
