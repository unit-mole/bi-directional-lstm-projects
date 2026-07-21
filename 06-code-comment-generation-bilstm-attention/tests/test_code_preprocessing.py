from src.code_preprocessing import preprocess_code, split_identifier, strip_python_docstrings, validate_python_code


def test_docstring_is_removed_but_return_is_preserved():
    code = 'def add(a, b):\n    """Return the sum."""\n    return a + b'
    cleaned = strip_python_docstrings(code)
    assert "Return the sum" not in cleaned
    assert "return a + b" in cleaned


def test_semantic_preprocessing_preserves_operator():
    cleaned = preprocess_code("def add(a, b):\n    return a + b")
    assert "+" in cleaned.split()
    assert "return" in cleaned.split()


def test_legacy_mode_matches_original_whitespace_cleanup():
    assert preprocess_code("def f():\n    return 1", mode="legacy") == "def f(): return 1"


def test_identifier_splitting():
    assert split_identifier("calculateAverage_value") == ["calculate", "average", "value"]


def test_validation_reports_invalid_code():
    valid, error = validate_python_code("def broken(:")
    assert valid is False
    assert error
