import pandas as pd

from src.pair_generation import generate_balanced_pairs


def test_generate_balanced_pairs_is_balanced():
    resumes = pd.DataFrame([
        {"resume_id": "r1", "category": "A", "resume_text": "alpha skills"},
        {"resume_id": "r2", "category": "B", "resume_text": "beta skills"},
    ])
    jobs = pd.DataFrame([
        {"job_id": "j1", "category": "A", "job_description": "alpha role", "split": "train", "template_index": 0},
        {"job_id": "j2", "category": "B", "job_description": "beta role", "split": "train", "template_index": 0},
    ])
    pairs = generate_balanced_pairs(resumes, jobs)
    assert pairs["label"].value_counts().to_dict() == {1: 2, 0: 2}
    assert set(pairs["split"]) == {"train"}
