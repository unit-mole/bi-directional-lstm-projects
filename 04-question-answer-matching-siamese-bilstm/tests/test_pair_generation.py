from src.pair_generation import canonical_pair_key, deduplicate_pairs


def test_canonical_key_is_order_invariant():
    assert canonical_pair_key("a", "b") == canonical_pair_key("b", "a")


def test_deduplicate_pairs_removes_reversed_duplicate():
    rows = [("a", "b"), ("b", "a"), ("a", "c")]
    assert deduplicate_pairs(rows) == [("a", "b"), ("a", "c")]
