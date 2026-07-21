from src.medical_text_preprocessing import clean_medical_text


def test_legacy_cleaner_matches_supplied_notebook_contract() -> None:
    text = "Patient's BP: 140/90. No chest-pain!"
    cleaned = clean_medical_text(text, mode="legacy")
    assert cleaned == "patient s bp 140 90 no chest pain"


def test_clinical_safe_cleaner_preserves_numeric_context_and_negation() -> None:
    text = "<b>No fever</b>; BP 140/90, O2 98%, dose 2.5-mg."
    cleaned = clean_medical_text(text, mode="clinical_safe")
    assert "no fever" in cleaned
    assert "140/90" in cleaned
    assert "98%" in cleaned
    assert "2.5-mg" in cleaned


def test_cleaner_normalizes_whitespace() -> None:
    assert clean_medical_text("  chest\n\n pain  ") == "chest pain"
