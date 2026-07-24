from src.tokenizer_utils import build_vocabulary
from src.sequence_generation import encode_texts
def test_sequence_shape():
    vocab=build_vocabulary(["happy joy","sad fear"],100); tensor=encode_texts(["happy","fear"],vocab,5); assert tuple(tensor.shape)==(2,5)
