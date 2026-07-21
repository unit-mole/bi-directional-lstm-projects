from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .text_preprocessing import clean_text

RESUME_CANDIDATES = (
    "resume_text", "resume", "resume_str", "candidate_profile", "candidate_summary"
)
JOB_CANDIDATES = (
    "job_description", "job_text", "jd_text", "description", "requirements"
)
LABEL_CANDIDATES = (
    "label", "target", "is_match", "match", "relevance", "match_score", "score"
)


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return str(normalized[candidate])
    for column in columns:
        lower = str(column).lower()
        if any(candidate in lower for candidate in candidates):
            return str(column)
    return None


def detect_pair_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    return {
        "resume": _find_column(frame.columns, RESUME_CANDIDATES),
        "job_description": _find_column(frame.columns, JOB_CANDIDATES),
        "label": _find_column(frame.columns, LABEL_CANDIDATES),
    }


def standardize_pair_dataframe(frame: pd.DataFrame, *, require_label: bool = False) -> pd.DataFrame:
    detected = detect_pair_columns(frame)
    if not detected["resume"] or not detected["job_description"]:
        raise ValueError(
            "CSV must contain resume and job-description columns. "
            f"Detected columns: {detected}. Received: {list(frame.columns)}"
        )
    if require_label and not detected["label"]:
        raise ValueError("A label/target column is required for training or evaluation.")

    rename_map = {
        detected["resume"]: "resume_text",
        detected["job_description"]: "job_description",
    }
    if detected["label"]:
        rename_map[detected["label"]] = "label"

    result = frame.rename(columns=rename_map).copy()
    result["resume_text"] = result["resume_text"].fillna("").map(clean_text)
    result["job_description"] = result["job_description"].fillna("").map(clean_text)
    result = result[(result["resume_text"] != "") & (result["job_description"] != "")]
    result = result.drop_duplicates(subset=["resume_text", "job_description"]).reset_index(drop=True)
    if "label" in result:
        result["label"] = normalize_binary_labels(result["label"])
    return result


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    positive = {"1", "true", "yes", "match", "matched", "relevant", "good fit", "strong match"}
    negative = {"0", "false", "no", "no match", "unmatched", "irrelevant", "poor fit", "weak match"}

    def convert(value: object) -> int:
        if pd.isna(value):
            raise ValueError("Target contains missing values.")
        if isinstance(value, (int, float)) and float(value) in {0.0, 1.0}:
            return int(value)
        normalized = str(value).strip().lower()
        if normalized in positive:
            return 1
        if normalized in negative:
            return 0
        raise ValueError(f"Unsupported binary label: {value!r}")

    return series.map(convert).astype(int)


def load_pair_csv(path: str | Path, *, require_label: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return standardize_pair_dataframe(frame, require_label=require_label)
