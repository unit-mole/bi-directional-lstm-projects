from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def build_tokenizer(
    texts: Iterable[str],
    *,
    maximum_vocabulary_size: int = 30_000,
    oov_token: str = "<OOV>",
):
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(
        num_words=maximum_vocabulary_size,
        oov_token=oov_token,
    )
    tokenizer.fit_on_texts(list(texts))
    return tokenizer


def save_tokenizer(tokenizer, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save the Tokenizer JSON object directly. The loader also supports the
    # nested JSON-string format produced by the original notebook.
    parsed = json.loads(tokenizer.to_json())
    path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")


def load_tokenizer(input_path: str | Path):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    path = Path(input_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, str):
        tokenizer_json = raw
    elif isinstance(raw, dict):
        tokenizer_json = json.dumps(raw)
    else:
        raise ValueError("Unsupported tokenizer JSON format.")

    return tokenizer_from_json(tokenizer_json)
