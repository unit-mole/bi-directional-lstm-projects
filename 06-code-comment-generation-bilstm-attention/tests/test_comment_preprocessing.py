from src.comment_preprocessing import add_boundary_tokens, clean_comment, is_meaningful_comment


def test_comment_cleaning_and_boundary_tokens():
    assert clean_comment(" Returns   the SUM! ") == "returns the sum"
    assert add_boundary_tokens("Returns the sum!") == "<start> returns the sum <end>"


def test_meaningless_comments_are_rejected():
    assert not is_meaningful_comment("TODO")
    assert is_meaningful_comment("returns the computed value")
