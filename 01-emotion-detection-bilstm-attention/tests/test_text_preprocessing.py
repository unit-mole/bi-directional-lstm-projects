from src.text_preprocessing import normalize_text


def test_normalize_text_preserves_emotion_signals():
    result = normalize_text("WOW!!! I am #Excited 😄 https://example.com @friend")
    assert "exclamationtoken" in result
    assert "hashtagtoken" in result
    assert "excited" in result
    assert "urltoken" in result
    assert "usertoken" in result
    assert "allcapstoken" in result


def test_normalize_text_handles_none_and_spacing():
    assert normalize_text(None) == ""
    assert normalize_text("  I   feel   calm  ") == "i feel calm"
