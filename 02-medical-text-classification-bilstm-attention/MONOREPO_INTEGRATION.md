# Monorepo Integration Guide

## Target repository

```text
bi-directional-lstm-projects/
```

## Copy these items

From the generated package, copy:

```text
02-medical-text-classification-bilstm-attention/
.github/workflows/02-medical-text-classification-bilstm-attention.yml
```

Review and merge—rather than blindly overwrite—the generated root files:

```text
README.md
.gitignore
LICENSE
.streamlit/config.toml
```

This protects any Project 01 content already present in the repository.

## Expected final placement

```text
bi-directional-lstm-projects/
├── .github/
│   └── workflows/
│       ├── 01-emotion-detection-bilstm-attention.yml
│       └── 02-medical-text-classification-bilstm-attention.yml
├── .streamlit/
│   └── config.toml
├── 01-emotion-detection-bilstm-attention/
├── 02-medical-text-classification-bilstm-attention/
├── .gitignore
├── LICENSE
└── README.md
```

## Git commands

```bash
git checkout -b project-02-medical-text-bilstm
git add .
git status
git commit -m "Add medical text classification BiLSTM attention project"
git push -u origin project-02-medical-text-bilstm
```

Open a pull request and confirm that the Project 02 workflow passes before merging.

## Model artifact policy

The included `.keras` file is approximately 1.6 MB, so normal Git storage is sufficient. Git LFS is not required for this artifact. Use Git LFS later if retrained artifacts become large, and confirm that the selected hosting platform can retrieve them.

## Do not commit

- protected health information,
- employer-confidential text,
- private datasets,
- real patient records,
- secrets,
- environment folders,
- temporary uploads,
- training checkpoints not intended for release.
