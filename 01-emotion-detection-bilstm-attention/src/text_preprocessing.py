\
"""Text cleaning and tokenization utilities."""
from __future__ import annotations
import html
import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?]+", flags=re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    value = html.unescape(str(text))
    value = unicodedata.normalize("NFKC", value)
    value = URL_PATTERN.sub(" <url> ", value)
    value = MENTION_PATTERN.sub(" <user> ", value)
    value = value.replace("#", " ")
    value = SPACE_PATTERN.sub(" ", value).strip().lower()
    return value


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(clean_text(text))
