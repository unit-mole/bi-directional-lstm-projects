"""Tokenizer creation and portable JSON serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def create_tokenizer(texts: Iterable[str], max_vocab_size: int = 30_000) -> Any:
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token="<OOV>",
        lower=False,
        filters='"#$%&()*+,./:;<=>@[\\]^_`{|}~\t\n',
    )
    tokenizer.fit_on_texts(list(texts))
    return tokenizer


def effective_vocabulary_size(tokenizer: Any, max_vocab_size: int) -> int:
    return min(max_vocab_size, len(tokenizer.word_index) + 1)


def save_tokenizer(tokenizer: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Any:
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    raw = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if isinstance(parsed, str):
        raw = parsed
    return tokenizer_from_json(raw)
