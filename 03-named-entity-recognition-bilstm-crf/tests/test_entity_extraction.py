from src.entity_extraction import extract_entities, repair_bio_tags


def test_bio_repair_and_entity_extraction() -> None:
    tokens = ["New", "York", "hosts", "Apple"]
    tags = ["I-LOC", "I-LOC", "O", "B-ORG"]
    repaired = repair_bio_tags(tags)
    assert repaired[0] == "B-LOC"
    entities = extract_entities(tokens, repaired, [0.9, 0.8, 0.99, 0.95])
    assert entities[0]["entity_text"] == "New York"
    assert entities[0]["entity_type"] == "LOC"
    assert entities[1]["entity_type"] == "ORG"
