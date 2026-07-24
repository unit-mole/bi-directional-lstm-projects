"""Sequence helper functions."""
from __future__ import annotations
import torch
from .tokenizer_utils import Vocabulary

def encode_texts(texts, vocabulary: Vocabulary, max_length: int) -> torch.Tensor:
    return torch.tensor([vocabulary.encode(text,max_length)[0] for text in texts],dtype=torch.long)
