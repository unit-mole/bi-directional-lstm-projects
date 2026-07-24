from src.text_preprocessing import clean_text, tokenize
def test_cleaning_and_tokens():
    assert clean_text("  HAPPY!!!  ")=="happy!!!"
    assert tokenize("I'm very happy!")==["i'm","very","happy","!"]
