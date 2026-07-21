from pathlib import Path

import pandas as pd

from src.data_preprocessing import (
    infer_text_and_label_columns,
    load_and_prepare_dataset,
)


def test_infer_actual_uploaded_dataset_columns() -> None:
    frame = pd.DataFrame(
        {
            "medical_specialty": ["Cardiology"],
            "transcription": ["Chest pain"],
        }
    )
    assert infer_text_and_label_columns(frame) == (
        "transcription",
        "medical_specialty",
    )


def test_prepare_dataset_removes_duplicate_text(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "medical_specialty": ["A", "A", "B", "B", "B"],
            "transcription": ["x", "x", "y", "z", "w"],
        }
    ).to_csv(source, index=False)

    prepared, audit = load_and_prepare_dataset(
        source,
        minimum_class_count=1,
    )
    assert len(prepared) == 4
    assert audit.duplicate_text_rows == 1
