"""Dataset loading, validation, and deterministic splitting."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .text_preprocessing import clean_text

TEXT_ALIASES = ("text", "sentence", "message", "content", "comment")
LABEL_ALIASES = ("emotion", "label", "target", "class")

@dataclass(frozen=True)
class DatasetAudit:
    input_rows: int
    cleaned_rows: int
    duplicate_rows_removed: int
    text_column: str
    label_column: str
    class_counts: dict[str, int]
    def to_dict(self):
        return asdict(self)


def _resolve(columns, aliases):
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    raise ValueError(f"Required column not found. Accepted names: {aliases}")


def load_and_clean_dataset(path: str | Path):
    frame = pd.read_csv(path)
    text_col = _resolve(frame.columns, TEXT_ALIASES)
    label_col = _resolve(frame.columns, LABEL_ALIASES)
    input_rows = len(frame)
    work = frame[[text_col, label_col]].rename(columns={text_col: "text", label_col: "emotion"}).copy()
    work = work.dropna(subset=["text", "emotion"])
    work["text"] = work["text"].astype(str).str.strip()
    work["emotion"] = work["emotion"].astype(str).str.strip().str.lower()
    work = work[work["text"].ne("") & work["emotion"].ne("")]
    work["text_clean"] = work["text"].map(clean_text)
    before = len(work)
    work = work.drop_duplicates(subset=["text_clean", "emotion"]).reset_index(drop=True)
    audit = DatasetAudit(
        input_rows=input_rows,
        cleaned_rows=len(work),
        duplicate_rows_removed=before-len(work),
        text_column=text_col,
        label_column=label_col,
        class_counts={str(k): int(v) for k, v in work["emotion"].value_counts().sort_index().items()},
    )
    return work, audit


def validate_class_support(frame: pd.DataFrame, minimum_samples: int) -> None:
    counts = frame["emotion"].value_counts()
    weak = counts[counts < minimum_samples]
    if not weak.empty:
        raise ValueError(f"Classes below minimum support ({minimum_samples}): {weak.to_dict()}")


def split_dataframe(frame, validation_size=0.15, test_size=0.15, random_seed=42):
    train, temp = train_test_split(
        frame, test_size=validation_size+test_size, random_state=random_seed,
        stratify=frame["emotion"]
    )
    relative_test = test_size / (validation_size + test_size)
    validation, test = train_test_split(
        temp, test_size=relative_test, random_state=random_seed,
        stratify=temp["emotion"]
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
