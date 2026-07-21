# Monorepo Integration

Copy the entire `04-question-answer-matching-siamese-bilstm/` folder into the root of `bi-directional-lstm-projects` and copy the workflow file to `.github/workflows/`.

Expected locations:

```text
bi-directional-lstm-projects/
├── .github/workflows/04-question-answer-matching-siamese-bilstm.yml
└── 04-question-answer-matching-siamese-bilstm/
```

Run from the project folder:

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/validate_artifacts.py --metadata-only
streamlit run app/streamlit_app.py
```
