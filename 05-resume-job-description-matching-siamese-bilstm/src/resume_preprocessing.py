from __future__ import annotations

from .text_preprocessing import clean_text, extract_skills, mask_personal_identifiers


def preprocess_resume(text: object) -> str:
    """Prepare resume text for model inference with basic PII masking."""
    return clean_text(text, mask_pii=True)


__all__ = ["preprocess_resume", "extract_skills", "mask_personal_identifiers"]
