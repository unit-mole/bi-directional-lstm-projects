"""Dataset loading, validation, cleaning, and leakage-safe splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .text_preprocessing import normalize_text

TEXT_CANDIDATES = ("text", "sentence", "statement", "message", "tweet", "content", "comment")
LABEL_CANDIDATES = ("emotion", "label", "target", "class", "sentiment", "category")


@dataclass(frozen=True)
class DatasetAudit:
    source_rows: int
    cleaned_rows: int
    missing_text_rows: int
    missing_label_rows: int
    duplicate_text_rows: int
    number_of_classes: int
    text_column: str
    label_column: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _identify_column(columns: Iterable[str], candidates: tuple[str, ...], kind: str) -> str:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for column in columns:
        lower = str(column).strip().lower()
        if any(candidate in lower for candidate in candidates):
            return str(column)
    raise ValueError(f"Could not identify the {kind} column. Available columns: {list(columns)}")


def standardize_label(label: object) -> str:
    """Return a clean, human-readable emotion label."""

    value = str(label).strip().replace("_", " ").replace("-", " ")
    value = " ".join(value.split()).lower()
    aliases = {
        "sad": "sadness",
        "happy": "joy",
        "fearful": "fear",
        "surprised": "surprise",
        "angry": "anger",
    }
    return aliases.get(value, value)


def load_and_clean_dataset(
    csv_path: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, DatasetAudit]:
    """Load a CSV and return standardized `text`, `text_clean`, and `emotion` columns."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("The dataset is empty.")

    text_column = text_column or _identify_column(raw.columns, TEXT_CANDIDATES, "text")
    label_column = label_column or _identify_column(raw.columns, LABEL_CANDIDATES, "label")

    missing_text = int(raw[text_column].isna().sum())
    missing_label = int(raw[label_column].isna().sum())

    frame = raw[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "emotion"}
    )
    frame = frame.dropna(subset=["text", "emotion"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["emotion"] = frame["emotion"].map(standardize_label)
    frame["text_clean"] = frame["text"].map(normalize_text)
    frame = frame[(frame["text_clean"] != "") & (frame["emotion"] != "")].copy()

    duplicate_count = int(frame.duplicated(subset=["text_clean"]).sum())
    if drop_duplicates:
        frame = frame.drop_duplicates(subset=["text_clean"], keep="first")

    frame = frame.reset_index(drop=True)
    audit = DatasetAudit(
        source_rows=len(raw),
        cleaned_rows=len(frame),
        missing_text_rows=missing_text,
        missing_label_rows=missing_label,
        duplicate_text_rows=duplicate_count,
        number_of_classes=int(frame["emotion"].nunique()),
        text_column=text_column,
        label_column=label_column,
    )
    return frame, audit


def validate_class_support(frame: pd.DataFrame, minimum_samples_per_class: int = 3) -> None:
    """Raise a helpful error when class support is too small for evaluation."""

    counts = frame["emotion"].value_counts()
    insufficient = counts[counts < minimum_samples_per_class]
    if not insufficient.empty:
        details = ", ".join(f"{label}={count}" for label, count in insufficient.items())
        raise ValueError(
            "Each class needs enough examples for train/validation/test splitting. "
            f"Minimum requested: {minimum_samples_per_class}. Insufficient classes: {details}"
        )


def split_dataframe(
    frame: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe once so text, labels, and metadata stay aligned.

    Stratification is used whenever class counts permit it. The tokenizer must
    be fitted only after this function, using `train_df` only.
    """

    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")

    temporary_size = validation_size + test_size
    train_df, temporary_df = train_test_split(
        frame,
        test_size=temporary_size,
        random_state=random_seed,
        stratify=frame["emotion"],
    )
    relative_test_size = test_size / temporary_size
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=relative_test_size,
        random_state=random_seed,
        stratify=temporary_df["emotion"],
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
