from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _keras_text_tools():
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return Tokenizer, tokenizer_from_json, pad_sequences
    except ImportError:
        try:
            from keras.src.legacy.preprocessing.text import Tokenizer, tokenizer_from_json
            from keras.src.utils.sequence_utils import pad_sequences
            return Tokenizer, tokenizer_from_json, pad_sequences
        except ImportError:
            import numpy as np

            def pad_sequences(sequences, maxlen, padding="pre", truncating="pre", value=0):
                result = np.full((len(sequences), maxlen), value, dtype="int32")
                for row_index, sequence in enumerate(sequences):
                    values = list(sequence)
                    if truncating == "pre":
                        values = values[-maxlen:]
                    else:
                        values = values[:maxlen]
                    if not values:
                        continue
                    if padding == "post":
                        result[row_index, : len(values)] = values
                    else:
                        result[row_index, -len(values):] = values
                return result

            return None, None, pad_sequences


def fit_shared_tokenizer(texts: list[str], *, num_words: int, oov_token: str = "<OOV>"):
    Tokenizer, _, _ = _keras_text_tools()
    if Tokenizer is None:
        raise ImportError("TensorFlow or Keras is required to fit a tokenizer.")
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, lower=False, filters="")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(path: str | Path):
    _, tokenizer_from_json, _ = _keras_text_tools()
    if tokenizer_from_json is None:
        raise ImportError("TensorFlow or Keras is required to load a tokenizer artifact.")
    raw = Path(path).read_text(encoding="utf-8")
    # Support the original notebook's double-encoded JSON as well as normal JSON.
    parsed = json.loads(raw)
    tokenizer_json = parsed if isinstance(parsed, str) else json.dumps(parsed)
    return tokenizer_from_json(tokenizer_json)


def vocabulary_size(tokenizer: Any, maximum: int | None = None) -> int:
    size = len(tokenizer.word_index) + 1
    return min(size, maximum) if maximum else size


def tokenizer_metadata(tokenizer: Any, *, maximum: int | None = None) -> dict[str, Any]:
    return {
        "document_count": int(getattr(tokenizer, "document_count", 0)),
        "observed_vocabulary_size": len(getattr(tokenizer, "word_index", {})) + 1,
        "effective_vocabulary_size": vocabulary_size(tokenizer, maximum),
        "oov_token": getattr(tokenizer, "oov_token", None),
    }
