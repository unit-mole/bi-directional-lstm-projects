from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_LEGACY_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_CLINICAL_UNSAFE_RE = re.compile(r"[^\w\s%./:+\-]")

# Conservative expansions. These are disabled by default because medical
# abbreviations can be ambiguous across specialties and organizations.
SAFE_ABBREVIATIONS: dict[str, str] = {
    "sob": "shortness of breath",
    "htn": "hypertension",
    "hr": "heart rate",
    "bp": "blood pressure",
}


def normalize_unicode(text: str) -> str:
    """Normalize visually equivalent Unicode characters."""
    return unicodedata.normalize("NFKC", text)


def remove_html(text: str) -> str:
    """Remove simple HTML tags and decode HTML entities."""
    return _HTML_TAG_RE.sub(" ", html.unescape(text))


def expand_abbreviations(text: str, mapping: dict[str, str] | None = None) -> str:
    """Expand only explicitly supplied abbreviations using word boundaries."""
    mapping = mapping or {}
    output = text
    for short, expanded in mapping.items():
        output = re.sub(
            rf"\b{re.escape(short)}\b",
            expanded,
            output,
            flags=re.IGNORECASE,
        )
    return output


def clean_medical_text(
    text: object,
    *,
    mode: str = "clinical_safe",
    lowercase: bool = True,
    abbreviation_mapping: dict[str, str] | None = None,
) -> str:
    """Clean medical text without removing clinically important meaning.

    Modes
    -----
    clinical_safe:
        Preserves numbers, decimals, percentages, slashes, colons, plus/minus
        signs, and hyphenated terms. This is recommended for retraining.
    legacy:
        Reproduces the cleaner used by the supplied notebook/model artifact.
        It lowercases text and removes all characters except a-z, 0-9, and
        whitespace. Use this mode when scoring with the supplied model.
    """
    value = "" if text is None else str(text)
    value = normalize_unicode(value)
    value = remove_html(value)
    value = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    value = expand_abbreviations(value, abbreviation_mapping)

    if lowercase:
        value = value.lower()

    if mode == "legacy":
        value = _LEGACY_NON_ALNUM_RE.sub(" ", value)
    elif mode == "clinical_safe":
        value = _CLINICAL_UNSAFE_RE.sub(" ", value)
    else:
        raise ValueError("mode must be either 'clinical_safe' or 'legacy'")

    return _WHITESPACE_RE.sub(" ", value).strip()


def clean_texts(
    texts: Iterable[object],
    *,
    mode: str = "clinical_safe",
    lowercase: bool = True,
    abbreviation_mapping: dict[str, str] | None = None,
) -> list[str]:
    return [
        clean_medical_text(
            text,
            mode=mode,
            lowercase=lowercase,
            abbreviation_mapping=abbreviation_mapping,
        )
        for text in texts
    ]
