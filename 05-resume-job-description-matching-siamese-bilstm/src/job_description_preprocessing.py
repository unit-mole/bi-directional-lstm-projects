from __future__ import annotations

from .text_preprocessing import clean_text, extract_skills


def preprocess_job_description(text: object) -> str:
    """Prepare job-description text without aggressive context removal."""
    return clean_text(text, mask_pii=True)


__all__ = ["preprocess_job_description", "extract_skills"]
