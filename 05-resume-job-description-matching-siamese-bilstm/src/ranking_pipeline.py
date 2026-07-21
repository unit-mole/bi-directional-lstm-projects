from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .matching_pipeline import ResumeJobMatcher


def rank_resumes(
    job_description: object,
    resumes: Iterable[dict[str, Any]],
    *,
    matcher: ResumeJobMatcher | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    scorer = matcher or ResumeJobMatcher()
    rows: list[dict[str, Any]] = []
    for index, resume in enumerate(resumes, start=1):
        text = resume.get("resume_text", "")
        result = scorer.predict(text, job_description)
        rows.append({
            "resume_id": resume.get("resume_id", f"resume_{index}"),
            "fit_score": result["fit_score"],
            "fit_score_percent": result["fit_score_percent"],
            "score_band": result["score_band"],
            "prediction": result["prediction"],
            "overlapping_skills": ", ".join(result["overlapping_skills"]),
            "missing_skills": ", ".join(result["missing_skills"]),
        })
    frame = pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame.head(top_k) if top_k else frame
