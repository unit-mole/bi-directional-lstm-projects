"""Vocabulary creation and sequence encoding."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .text_preprocessing import tokenize

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[str]]:
        tokens = tokenize(text)[:max_length]
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        ids += [self.pad_id] * (max_length - len(ids))
        return ids, tokens

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def build_vocabulary(texts: Iterable[str], max_vocab_size: int) -> Vocabulary:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(str(text)))
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(max(0, max_vocab_size - 2)):
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    return Vocabulary(token_to_id)
