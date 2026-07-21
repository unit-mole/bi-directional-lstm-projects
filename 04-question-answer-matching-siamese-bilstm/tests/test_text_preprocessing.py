from src.text_preprocessing import clean_text, token_overlap


def test_clean_text_preserves_negation_and_numbers():
    cleaned = clean_text("Why is this NOT working for model 4500?")
    assert "not" in cleaned
    assert "4500" in cleaned


def test_clean_text_removes_html_and_normalizes_spaces():
    assert clean_text("<b>Hello</b>   world") == "hello world"


def test_token_overlap_is_transparent():
    result = token_overlap("learn python fast", "fast python tutorial")
    assert result["shared_tokens"] == ["fast", "python"]
    assert 0 < result["jaccard_similarity"] < 1
