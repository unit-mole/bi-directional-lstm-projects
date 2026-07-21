from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .medical_text_preprocessing import clean_medical_text

TEXT_COLUMN_CANDIDATES = (
    "transcription",
    "clinical_text",
    "medical_text",
    "clinical_note",
    "note",
    "abstract",
    "description",
    "symptoms",
    "text",
)

LABEL_COLUMN_CANDIDATES = (
    "medical_specialty",
    "specialty",
    "category",
    "diagnosis",
    "condition",
    "label",
    "target",
    "class",
)


@dataclass(frozen=True)
class DataAudit:
    original_rows: int
    final_rows: int
    missing_text_rows: int
    missing_label_rows: int
    empty_text_rows: int
    duplicate_text_rows: int
    removed_rare_class_rows: int
    class_count: int
    text_column: str
    label_column: str

    def to_dict(self) -> dict[str, int | str]:
        return self.__dict__.copy()


def _find_candidate(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    for col in columns:
        lower = str(col).strip().lower()
        if any(candidate in lower for candidate in candidates):
            return str(col)
    return None


def infer_text_and_label_columns(
    dataframe: pd.DataFrame,
    text_column: str | None = None,
    label_column: str | None = None,
) -> tuple[str, str]:
    if text_column is not None and text_column not in dataframe.columns:
        raise KeyError(f"Text column '{text_column}' is not present.")
    if label_column is not None and label_column not in dataframe.columns:
        raise KeyError(f"Label column '{label_column}' is not present.")

    resolved_text = text_column or _find_candidate(dataframe.columns, TEXT_COLUMN_CANDIDATES)
    resolved_label = label_column or _find_candidate(dataframe.columns, LABEL_COLUMN_CANDIDATES)

    if resolved_text is None or resolved_label is None:
        raise ValueError(
            "Unable to infer text/label columns. "
            f"Available columns: {list(dataframe.columns)}"
        )
    if resolved_text == resolved_label:
        raise ValueError("Text and label columns resolved to the same column.")

    return resolved_text, resolved_label


def standardize_label(value: object) -> str:
    label = "" if value is None else str(value)
    return " ".join(label.strip().split())


def load_and_prepare_dataset(
    csv_path: str | Path,
    *,
    text_column: str | None = None,
    label_column: str | None = None,
    minimum_class_count: int = 2,
    preprocessing_mode: str = "clinical_safe",
    remove_duplicate_text: bool = True,
) -> tuple[pd.DataFrame, DataAudit]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path)
    original_rows = len(raw)
    resolved_text, resolved_label = infer_text_and_label_columns(
        raw, text_column=text_column, label_column=label_column
    )

    working = raw[[resolved_text, resolved_label]].copy()
    missing_text_rows = int(working[resolved_text].isna().sum())
    missing_label_rows = int(working[resolved_label].isna().sum())

    working = working.dropna(subset=[resolved_text, resolved_label])
    working["clinical_text"] = working[resolved_text].astype(str).str.strip()
    working["label"] = working[resolved_label].map(standardize_label)

    empty_mask = working["clinical_text"].eq("") | working["label"].eq("")
    empty_text_rows = int(empty_mask.sum())
    working = working.loc[~empty_mask].copy()

    duplicate_text_rows = int(
        working.duplicated(subset=["clinical_text"], keep="first").sum()
    )
    if remove_duplicate_text:
        working = working.drop_duplicates(subset=["clinical_text"], keep="first")

    working["text_clean"] = working["clinical_text"].map(
        lambda value: clean_medical_text(value, mode=preprocessing_mode)
    )
    working = working.loc[working["text_clean"].str.len() > 0].copy()

    class_counts = working["label"].value_counts()
    valid_labels = class_counts[class_counts >= minimum_class_count].index
    rare_mask = ~working["label"].isin(valid_labels)
    removed_rare_class_rows = int(rare_mask.sum())
    working = working.loc[~rare_mask].copy()

    working["text_char_len"] = working["clinical_text"].str.len()
    working["text_word_len"] = working["clinical_text"].str.split().str.len()
    working = working.reset_index(drop=True)

    audit = DataAudit(
        original_rows=original_rows,
        final_rows=len(working),
        missing_text_rows=missing_text_rows,
        missing_label_rows=missing_label_rows,
        empty_text_rows=empty_text_rows,
        duplicate_text_rows=duplicate_text_rows,
        removed_rare_class_rows=removed_rare_class_rows,
        class_count=int(working["label"].nunique()),
        text_column=resolved_text,
        label_column=resolved_label,
    )
    return working, audit
