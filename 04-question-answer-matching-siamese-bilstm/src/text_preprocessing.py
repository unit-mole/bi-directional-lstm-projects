from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any, *, lowercase: bool = True, replace_urls: bool = True) -> str:
    """Clean text conservatively without deleting negations, entities, or numbers.

    Punctuation is normalized to spaces because the supplied tokenizer was trained with
    punctuation filtering. Question words and negations are retained.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = _HTML_RE.sub(" ", text)
    if replace_urls:
        text = _URL_RE.sub(" <URL> ", text)
    if lowercase:
        text = text.lower()
    text = re.sub(r"[^\w\s<>']", " ", text, flags=re.UNICODE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def token_overlap(text_a: str, text_b: str) -> dict[str, object]:
    """Return transparent lexical-overlap diagnostics for UI interpretation."""
    a_tokens = set(clean_text(text_a).split())
    b_tokens = set(clean_text(text_b).split())
    union = a_tokens | b_tokens
    shared = sorted(a_tokens & b_tokens)
    jaccard = len(shared) / len(union) if union else 0.0
    return {"shared_tokens": shared, "jaccard_similarity": float(jaccard)}
