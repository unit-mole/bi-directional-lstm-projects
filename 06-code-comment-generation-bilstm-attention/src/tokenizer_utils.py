from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _keras_tokenizer_api():
    try:
        from keras.preprocessing.text import Tokenizer, tokenizer_from_json
    except ImportError:
        from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
    return Tokenizer, tokenizer_from_json


def build_tokenizers(
    code_texts: list[str],
    comment_texts: list[str],
    *,
    max_code_vocab: int,
    max_comment_vocab: int,
    oov_token: str = "<OOV>",
):
    Tokenizer, _ = _keras_tokenizer_api()
    # filters="" is intentional: lexical preprocessing already separates code
    # operators, and Keras' default filters would delete useful symbols.
    code_tokenizer = Tokenizer(num_words=max_code_vocab, oov_token=oov_token, filters="", lower=False)
    comment_tokenizer = Tokenizer(num_words=max_comment_vocab, oov_token=oov_token, filters="", lower=True)
    code_tokenizer.fit_on_texts(code_texts)
    comment_tokenizer.fit_on_texts(comment_texts)
    return code_tokenizer, comment_tokenizer


def save_tokenizer(tokenizer: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(path: str | Path):
    _, tokenizer_from_json = _keras_tokenizer_api()
    raw = Path(path).read_text(encoding="utf-8")
    # The supplied notebook double-encoded the JSON string with json.dump.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            raw = parsed
    except json.JSONDecodeError:
        pass
    return tokenizer_from_json(raw)


def effective_vocab_size(tokenizer: Any, limit: int | None = None) -> int:
    size = len(tokenizer.word_index) + 1
    return min(size, limit) if limit else size
