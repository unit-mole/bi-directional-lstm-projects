"""Tokenization helpers that preserve token order, punctuation, and offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\w+(?:[-’']\w+)*|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


def tokenize_with_offsets(text: str) -> list[TokenSpan]:
    """Tokenize text without deleting punctuation or changing character offsets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [TokenSpan(m.group(0), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def tokenize_text(text: str) -> list[str]:
    return [span.text for span in tokenize_with_offsets(text)]


def normalize_token(token: str, lowercase: bool = True) -> str:
    """Apply only training-compatible normalization; no destructive cleaning."""
    token = token.strip()
    return token.lower() if lowercase else token
