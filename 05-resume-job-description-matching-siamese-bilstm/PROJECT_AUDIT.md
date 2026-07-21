# Project Audit

## Current objective

Estimate semantic alignment between a resume and a job description, then support top-k ranking of anonymized resumes for one role.

## Data quality conclusion

The supplied eight-row file is sufficient for demonstrating pipeline mechanics but far too small for a credible hiring model. It contains category labels rather than independently authored resume–job relevance labels. Synthetic job descriptions are therefore used only to create an executable portfolio demonstration.

## Model conclusion

The original uploaded model is not weight-shared and collapses to all-negative predictions on its four-row test set. It is retained as a legacy artifact, while the deployable project uses a corrected shared Siamese BiLSTM.

## Deployment readiness

The generated project is runnable and hostable. The model is intentionally small, artifacts are pre-generated, the app has a fallback scoring mode, and CI avoids expensive training. Deployment readiness does not imply production hiring readiness.

## Go/no-go recommendation

- **Portfolio demonstration:** Go.
- **Recruiter-facing live demo using synthetic data:** Go with disclaimers.
- **Real candidate screening or automated hiring decisions:** No-go without representative labeled data, legal review, fairness testing, security controls, governance, and human oversight.
