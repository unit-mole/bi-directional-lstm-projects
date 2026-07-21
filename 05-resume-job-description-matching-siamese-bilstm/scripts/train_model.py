from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG
from src.model_training import train_from_pairs
from src.pair_generation import generate_balanced_pairs
from src.ranking_pipeline import rank_resumes
from src.visualization import save_architecture_diagram


def main() -> None:
    resumes = pd.read_csv(CONFIG.data_dir / "sample" / "sample_resumes.csv")
    jobs = pd.read_csv(CONFIG.data_dir / "sample" / "sample_job_descriptions.csv")
    pairs = generate_balanced_pairs(resumes, jobs, seed=CONFIG.random_seed)
    CONFIG.training_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(CONFIG.training_pairs_path, index=False)

    result = train_from_pairs(pairs, config=CONFIG)
    save_architecture_diagram(CONFIG.project_dir / "images" / "architecture.png")

    # Produce a portfolio ranking example using the trained artifacts.
    from src.inference_pipeline import ResumeJobMatcher
    matcher = ResumeJobMatcher(config=CONFIG, allow_fallback=False)
    sample_job = jobs[jobs["category"] == "Data Science"].iloc[-1]["job_description"]
    candidates = resumes[["resume_id", "resume_text"]].to_dict(orient="records")
    ranking = rank_resumes(sample_job, candidates, matcher=matcher)
    ranking.to_csv(CONFIG.outputs_dir / "predictions" / "ranking_examples.csv", index=False)

    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved model: {CONFIG.model_path}")


if __name__ == "__main__":
    main()
