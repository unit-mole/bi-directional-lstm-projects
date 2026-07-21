import pandas as pd
import pytest

from src.data_preprocessing import infer_schema, standardize_dataframe


def test_schema_inference_for_codesearchnet_columns():
    frame = pd.DataFrame({"func_code_string": ["def f(): return 1"], "func_documentation_string": ["returns one"]})
    schema = infer_schema(frame)
    assert schema.code_column == "func_code_string"
    assert schema.comment_column == "func_documentation_string"


def test_standardization_drops_duplicates():
    frame = pd.DataFrame({
        "code": ["def add(a,b): return a+b"] * 2,
        "comment": ["returns the sum of values"] * 2,
    })
    result = standardize_dataframe(frame)
    assert len(result) == 1
    assert "<start>" in result.loc[0, "comment_sequence"]


def test_missing_columns_raise_clear_error():
    with pytest.raises(ValueError):
        infer_schema(pd.DataFrame({"x": [1]}))
