import pandas as pd

from src.data_preprocessing import normalize_binary_label, resolve_pair_columns


def test_resolve_actual_attached_columns():
    frame = pd.DataFrame(columns=["question1", "question2", "is_duplicate"])
    columns = resolve_pair_columns(frame)
    assert columns.text_a == "question1"
    assert columns.text_b == "question2"
    assert columns.label == "is_duplicate"


def test_normalize_label_strings():
    assert normalize_binary_label("match") == 1
    assert normalize_binary_label("not duplicate") == 0
