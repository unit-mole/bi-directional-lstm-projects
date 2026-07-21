from src.model_evaluation import token_f1


def test_token_f1_is_one_for_identical_text():
    assert token_f1("returns the value", "returns the value") == 1.0


def test_token_f1_is_zero_without_overlap():
    assert token_f1("returns value", "opens file") == 0.0
