from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass
class GenerationResult:
    comment: str
    token_ids: list[int]
    tokens: list[str]
    decoding_method: str
    attention: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


def _reverse_index(tokenizer: Any) -> dict[int, str]:
    return {int(index): word for word, index in tokenizer.word_index.items()}


def _special_ids(tokenizer: Any, start_token: str, end_token: str) -> tuple[int, int]:
    start_id = tokenizer.word_index.get(start_token)
    end_id = tokenizer.word_index.get(end_token)
    if start_id is None or end_id is None:
        raise ValueError("Comment tokenizer does not contain start/end tokens.")
    return int(start_id), int(end_id)


def greedy_decode(
    encoder_model,
    decoder_model,
    encoder_input: np.ndarray,
    comment_tokenizer: Any,
    *,
    max_tokens: int,
    start_token: str = "<start>",
    end_token: str = "<end>",
    attention_enabled: bool = True,
) -> GenerationResult:
    reverse = _reverse_index(comment_tokenizer)
    start_id, end_id = _special_ids(comment_tokenizer, start_token, end_token)
    encoder_state = encoder_model.predict(encoder_input, verbose=0)

    if attention_enabled:
        encoder_outputs, state_h, state_c = encoder_state
    else:
        state_h, state_c = encoder_state
        encoder_outputs = None

    current = start_id
    token_ids: list[int] = []
    attention_rows: list[np.ndarray] = []
    warnings: list[str] = []

    for _ in range(max_tokens):
        token_array = np.asarray([[current]], dtype=np.int32)
        if attention_enabled:
            probs, state_h, state_c, scores = decoder_model.predict(
                [token_array, encoder_outputs, state_h, state_c], verbose=0
            )
            attention_rows.append(np.asarray(scores[0, 0]))
        else:
            probs, state_h, state_c = decoder_model.predict(
                [token_array, state_h, state_c], verbose=0
            )
        next_id = int(np.argmax(probs[0, -1]))
        if next_id in {0, end_id}:
            break
        if next_id != start_id:
            token_ids.append(next_id)
        current = next_id

    tokens = [reverse.get(idx, "<OOV>") for idx in token_ids]
    if len(tokens) >= 4 and len(set(tokens)) <= 2:
        warnings.append("The checkpoint produced a repetitive low-diversity sequence.")
    return GenerationResult(
        comment=" ".join(tokens).strip(),
        token_ids=token_ids,
        tokens=tokens,
        decoding_method="greedy",
        attention=np.vstack(attention_rows) if attention_rows else None,
        warnings=warnings,
    )


def beam_search_decode(
    encoder_model,
    decoder_model,
    encoder_input: np.ndarray,
    comment_tokenizer: Any,
    *,
    max_tokens: int,
    beam_width: int = 3,
    length_penalty: float = 0.7,
    start_token: str = "<start>",
    end_token: str = "<end>",
    attention_enabled: bool = True,
) -> GenerationResult:
    reverse = _reverse_index(comment_tokenizer)
    start_id, end_id = _special_ids(comment_tokenizer, start_token, end_token)
    encoder_state = encoder_model.predict(encoder_input, verbose=0)
    if attention_enabled:
        encoder_outputs, state_h, state_c = encoder_state
    else:
        state_h, state_c = encoder_state
        encoder_outputs = None

    # sequence, log_probability, h, c, attention_rows, ended
    beams = [([start_id], 0.0, state_h, state_c, [], False)]
    for _ in range(max_tokens):
        candidates = []
        for sequence, score, h, c, attn_rows, ended in beams:
            if ended:
                candidates.append((sequence, score, h, c, attn_rows, True))
                continue
            token_array = np.asarray([[sequence[-1]]], dtype=np.int32)
            if attention_enabled:
                probs, next_h, next_c, scores = decoder_model.predict(
                    [token_array, encoder_outputs, h, c], verbose=0
                )
                row = np.asarray(scores[0, 0])
            else:
                probs, next_h, next_c = decoder_model.predict([token_array, h, c], verbose=0)
                row = None
            distribution = np.asarray(probs[0, -1], dtype=float)
            top = np.argpartition(distribution, -beam_width)[-beam_width:]
            for token_id in top:
                token_id = int(token_id)
                probability = max(float(distribution[token_id]), 1e-12)
                candidates.append((
                    sequence + [token_id],
                    score + math.log(probability),
                    next_h,
                    next_c,
                    attn_rows + ([row] if row is not None else []),
                    token_id == end_id,
                ))

        def normalized(item):
            sequence, score, *_ = item
            generated_length = max(1, len(sequence) - 1)
            return score / (generated_length ** length_penalty)

        beams = sorted(candidates, key=normalized, reverse=True)[:beam_width]
        if all(item[-1] for item in beams):
            break

    best = max(beams, key=lambda item: item[1] / (max(1, len(item[0]) - 1) ** length_penalty))
    sequence, _, _, _, attention_rows, _ = best
    ids = [idx for idx in sequence[1:] if idx not in {0, start_id, end_id}]
    tokens = [reverse.get(idx, "<OOV>") for idx in ids]
    warnings = []
    if len(tokens) >= 4 and len(set(tokens)) <= 2:
        warnings.append("The checkpoint produced a repetitive low-diversity sequence.")
    return GenerationResult(
        comment=" ".join(tokens).strip(),
        token_ids=ids,
        tokens=tokens,
        decoding_method=f"beam-{beam_width}",
        attention=np.vstack(attention_rows) if attention_rows else None,
        warnings=warnings,
    )
