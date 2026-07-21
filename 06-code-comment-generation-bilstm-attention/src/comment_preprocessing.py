from __future__ import annotations

import re
import unicodedata

_MEANINGLESS = {"todo", "fixme", "none", "na", "n/a", "test", "placeholder"}


def clean_comment(comment: str, *, max_tokens: int = 28) -> str:
    text = unicodedata.normalize("NFKC", str(comment or "")).lower().strip()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[`*_#]", " ", text)
    text = re.sub(r"[^a-z0-9_+\-/.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:max_tokens])


def is_meaningful_comment(comment: str, *, min_tokens: int = 3) -> bool:
    cleaned = clean_comment(comment)
    tokens = cleaned.split()
    if len(tokens) < min_tokens:
        return False
    if cleaned in _MEANINGLESS:
        return False
    return any(token.isalpha() and len(token) > 1 for token in tokens)


def add_boundary_tokens(
    comment: str,
    *,
    start_token: str = "<start>",
    end_token: str = "<end>",
    max_content_tokens: int = 28,
) -> str:
    cleaned = clean_comment(comment, max_tokens=max_content_tokens)
    return f"{start_token} {cleaned} {end_token}".strip()
