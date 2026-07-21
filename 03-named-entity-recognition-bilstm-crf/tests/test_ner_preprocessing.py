from src.data_preprocessing import NERSentence, validate_bio_sequence
from src.ner_preprocessing import build_word_vocabulary, encode_sentences


def test_alignment_and_vocabulary() -> None:
    sentences = [NERSentence(("Apple", "hired", "Maya"), ("B-ORG", "O", "B-PER"))]
    vocabulary = build_word_vocabulary(sentences, lowercase=True)
    tags = {"O": 0, "B-ORG": 1, "B-PER": 2}
    token_ids, tag_ids = encode_sentences(sentences, vocabulary, tags, lowercase=True)
    assert token_ids[0][0] == vocabulary["apple"]
    assert tag_ids == [[1, 0, 2]]


def test_invalid_bio_transition_is_reported() -> None:
    errors = validate_bio_sequence(["O", "I-PER"])
    assert errors
