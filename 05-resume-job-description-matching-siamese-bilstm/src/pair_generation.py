from __future__ import annotations

import random
from collections.abc import Sequence

import pandas as pd

from .text_preprocessing import clean_text


def generate_balanced_pairs(
    resumes: pd.DataFrame,
    job_descriptions: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Create one negative pair for every positive pair.

    Expected columns:
    - resumes: resume_id, category, resume_text
    - jobs: job_id, category, job_description, split
    """
    required_resume = {"resume_id", "category", "resume_text"}
    required_job = {"job_id", "category", "job_description"}
    if not required_resume.issubset(resumes.columns):
        raise ValueError(f"Resume data must contain {sorted(required_resume)}")
    if not required_job.issubset(job_descriptions.columns):
        raise ValueError(f"Job data must contain {sorted(required_job)}")

    rng = random.Random(seed)
    job_rows = job_descriptions.reset_index(drop=True)
    records: list[dict[str, object]] = []

    for resume_index, resume in resumes.reset_index(drop=True).iterrows():
        same_jobs = job_rows[job_rows["category"] == resume["category"]]
        for _, positive_job in same_jobs.iterrows():
            records.append(_record(resume, positive_job, label=1))

            negative_pool = job_rows[
                (job_rows["category"] != resume["category"])
                & (job_rows.get("split", "train") == positive_job.get("split", "train"))
            ]
            if negative_pool.empty:
                negative_pool = job_rows[job_rows["category"] != resume["category"]]
            # Deterministic but varied negative selection.
            candidates = list(negative_pool.index)
            selected_index = candidates[(resume_index + int(positive_job.get("template_index", 0))) % len(candidates)]
            negative_job = job_rows.loc[selected_index]
            records.append(_record(resume, negative_job, label=0))

    pairs = pd.DataFrame(records)
    pairs["resume_text"] = pairs["resume_text"].map(clean_text)
    pairs["job_description"] = pairs["job_description"].map(clean_text)
    pairs = pairs.drop_duplicates(subset=["resume_id", "job_id", "label"])
    pairs = pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pairs


def _record(resume: pd.Series, job: pd.Series, *, label: int) -> dict[str, object]:
    return {
        "resume_id": resume["resume_id"],
        "job_id": job["job_id"],
        "resume_category": resume["category"],
        "job_category": job["category"],
        "resume_text": resume["resume_text"],
        "job_description": job["job_description"],
        "label": int(label),
        "split": job.get("split", "train"),
        "template_index": int(job.get("template_index", 0)),
    }


def sample_negative_categories(categories: Sequence[str], current: str, *, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    choices = [category for category in categories if category != current]
    rng.shuffle(choices)
    return choices
