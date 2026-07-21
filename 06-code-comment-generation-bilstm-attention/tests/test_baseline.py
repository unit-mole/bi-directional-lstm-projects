from src.baseline import identifier_baseline


def test_identifier_baseline_uses_function_name():
    result = identifier_baseline("def calculate_average(values):\n    return sum(values) / len(values)")
    assert "calculate average" in result.lower()


def test_identifier_baseline_detects_addition():
    result = identifier_baseline("def add(a, b):\n    return a + b")
    assert "sum" in result.lower()
