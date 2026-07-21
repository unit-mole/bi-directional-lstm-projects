from src.text_preprocessing import clean_text, compare_skills, extract_skills


def test_clean_text_masks_common_identifiers_and_preserves_skills():
    text = "Email Jane@example.com; call +1 (970) 555-1234. Skills: C++, Python, Power BI."
    cleaned = clean_text(text)
    assert "jane@example.com" not in cleaned
    assert "555" not in cleaned
    assert "cplusplus" in cleaned
    assert "python" in cleaned
    assert "power bi" in cleaned


def test_skill_comparison_returns_overlap_and_gaps():
    result = compare_skills(
        "Python SQL machine learning experience",
        "Need Python SQL machine learning Docker and AWS",
    )
    assert "Python" in result["overlapping_skills"]
    assert "Docker" in result["missing_skills"]
    assert 0.0 <= result["skill_coverage"] <= 1.0
