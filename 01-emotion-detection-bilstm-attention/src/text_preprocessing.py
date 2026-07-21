"""Emotion-aware text normalization.

The preprocessing intentionally preserves signals that can matter for emotion:
emojis are converted to aliases, hashtags retain their words, and repeated
exclamation/question marks are represented with explicit tokens.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Final

try:
    import emoji
except ImportError:  # pragma: no cover - graceful local fallback
    emoji = None

URL_PATTERN: Final = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_PATTERN: Final = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_PATTERN: Final = re.compile(r"#([A-Za-z0-9_]+)")
HTML_TAG_PATTERN: Final = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN: Final = re.compile(r"\s+")
UPPERCASE_WORD_PATTERN: Final = re.compile(r"\b[A-Z]{3,}\b")


def _replace_intensity_marks(text: str) -> str:
    """Convert punctuation intensity to stable tokens before tokenization."""

    text = re.sub(
        r"!+",
        lambda match: " " + " ".join(["exclamationtoken"] * min(len(match.group()), 3)) + " ",
        text,
    )
    text = re.sub(
        r"\?+",
        lambda match: " " + " ".join(["questiontoken"] * min(len(match.group()), 3)) + " ",
        text,
    )
    return text


def normalize_text(text: object) -> str:
    """Normalize text without blindly deleting emotion-bearing information.

    Args:
        text: Any value that should represent a text sample.

    Returns:
        A deterministic, lowercase string suitable for Keras tokenization.
    """

    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", html.unescape(str(text)))
    uppercase_count = len(UPPERCASE_WORD_PATTERN.findall(normalized))
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = URL_PATTERN.sub(" urltoken ", normalized)
    normalized = MENTION_PATTERN.sub(" usertoken ", normalized)
    normalized = HASHTAG_PATTERN.sub(lambda match: f" hashtagtoken {match.group(1)} ", normalized)

    if emoji is not None:
        normalized = emoji.demojize(normalized, delimiters=(" emojitoken_", " "))

    normalized = _replace_intensity_marks(normalized)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^\w\s'\-]", " ", normalized, flags=re.UNICODE)

    if uppercase_count:
        normalized += " " + " ".join(["allcapstoken"] * min(uppercase_count, 3))

    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip().lower()
    return normalized


def batch_normalize_text(texts: list[object]) -> list[str]:
    """Normalize a list of text values."""

    return [normalize_text(text) for text in texts]
