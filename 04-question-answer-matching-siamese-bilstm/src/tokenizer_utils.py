from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _tokenizer_api():
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
    except ImportError:
        from keras.preprocessing.text import Tokenizer, tokenizer_from_json
    return Tokenizer, tokenizer_from_json


def build_tokenizer(texts: Iterable[str], *, num_words: int = 40000, oov_token: str = "<OOV>"):
    Tokenizer, _ = _tokenizer_api()
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(list(texts))
    return tokenizer


def save_tokenizer(tokenizer, path: str | Path) -> None:
    Path(path).write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(path: str | Path):
    _, tokenizer_from_json = _tokenizer_api()
    raw = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    # Support both standard JSON and the double-encoded JSON supplied with the original notebook.
    json_string = parsed if isinstance(parsed, str) else json.dumps(parsed)
    return tokenizer_from_json(json_string)
