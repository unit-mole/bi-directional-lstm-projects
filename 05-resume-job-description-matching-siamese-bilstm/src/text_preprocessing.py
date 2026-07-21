from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable

from .skills import SKILL_CATALOG

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_HTML_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


TECHNICAL_NORMALIZATIONS = {
    "c++": "cplusplus",
    "c#": "csharp",
    ".net": "dotnet",
    "node.js": "nodejs",
    "ci/cd": "ci cd",
    "powerbi": "power bi",
    "scikit learn": "scikit-learn",
}


def mask_personal_identifiers(text: str) -> str:
    """Mask common contact identifiers before public/demo processing."""
    text = _EMAIL_RE.sub(" <EMAIL> ", text)
    text = _URL_RE.sub(" <URL> ", text)
    text = _PHONE_RE.sub(" <PHONE> ", text)
    return text


def clean_text(text: object, *, mask_pii: bool = True, lowercase: bool = True) -> str:
    """Normalize text while preserving skills, numbers, and useful hiring context.

    The function intentionally avoids stop-word removal and stemming. Requirements,
    negations, years of experience, section headings, and technical terms can carry
    important meaning in resume–job matching.
    """
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", html.unescape(str(text)))
    value = _HTML_RE.sub(" ", value)
    value = value.replace("•", " ").replace("·", " ").replace("▪", " ")
    if mask_pii:
        value = mask_personal_identifiers(value)
    if lowercase:
        value = value.lower()

    for source, target in TECHNICAL_NORMALIZATIONS.items():
        value = value.replace(source, target)

    # Keep letters, numbers, common skill separators, and masked placeholders.
    value = re.sub(r"[^a-zA-Z0-9+#./<>\-\s]", " ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def tokenize_words(text: object) -> list[str]:
    cleaned = clean_text(text)
    return [token for token in cleaned.split() if token]


def extract_skills(text: object) -> list[str]:
    cleaned = f" {clean_text(text)} "
    found: list[str] = []
    for display_name, aliases in SKILL_CATALOG.items():
        if any(f" {alias.lower()} " in cleaned for alias in aliases):
            found.append(display_name)
    return sorted(found)


def compare_skills(resume_text: object, job_description: object) -> dict[str, list[str] | float]:
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))
    overlap = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    denominator = max(len(job_skills), 1)
    coverage = len(overlap) / denominator
    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "overlapping_skills": overlap,
        "missing_skills": missing,
        "skill_coverage": float(coverage),
    }


def join_nonempty(parts: Iterable[object]) -> str:
    return " ".join(clean_text(part) for part in parts if clean_text(part))
