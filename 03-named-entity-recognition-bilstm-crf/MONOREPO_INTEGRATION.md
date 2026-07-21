# Monorepo integration

Copy the following paths into the root of `bi-directional-lstm-projects`:

```text
.github/workflows/03-named-entity-recognition-bilstm-crf.yml
.streamlit/config.toml
03-named-entity-recognition-bilstm-crf/
```

Merge the supplied root `README.md`, `.gitignore`, and `LICENSE` with your existing root files rather than deleting content for Projects 01 and 02.

Run from the repository root:

```bash
pip install -r 03-named-entity-recognition-bilstm-crf/requirements-dev.txt
pytest 03-named-entity-recognition-bilstm-crf/tests -q
streamlit run 03-named-entity-recognition-bilstm-crf/app/streamlit_app.py
```

The workflow triggers only when Project 03 or its YAML file changes.
