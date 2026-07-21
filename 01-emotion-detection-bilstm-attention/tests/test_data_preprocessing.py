from pathlib import Path

import pandas as pd

from src.data_preprocessing import load_and_clean_dataset


def test_dataset_columns_are_detected_and_duplicates_removed(tmp_path: Path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "sentence": ["I am happy!", "I am happy!", "I am afraid"],
            "label": ["happy", "happy", "fearful"],
        }
    ).to_csv(path, index=False)

    frame, audit = load_and_clean_dataset(path)
    assert list(frame.columns) == ["text", "emotion", "text_clean"]
    assert len(frame) == 2
    assert set(frame["emotion"]) == {"joy", "fear"}
    assert audit.duplicate_text_rows == 1
