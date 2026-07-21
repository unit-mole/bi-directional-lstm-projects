from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_preprocessing import standardize_pair_dataframe
from src.inference_pipeline import RESPONSIBLE_USE, ResumeJobMatcher, rank_resumes

st.set_page_config(
    page_title="Resume–JD Siamese BiLSTM Matcher",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_matcher() -> ResumeJobMatcher:
    return ResumeJobMatcher(allow_fallback=True)


@st.cache_data
def load_samples() -> pd.DataFrame:
    return pd.read_csv(PROJECT_DIR / "data" / "sample" / "sample_resume_job_pairs.csv")


def score_dataframe(frame: pd.DataFrame, matcher: ResumeJobMatcher) -> pd.DataFrame:
    standardized = standardize_pair_dataframe(frame, require_label=False)
    results = []
    for _, row in standardized.iterrows():
        prediction = matcher.predict(row["resume_text"], row["job_description"])
        results.append({
            **row.to_dict(),
            "prediction": prediction["prediction"],
            "fit_score": round(prediction["fit_score"], 4),
            "fit_score_percent": round(prediction["fit_score_percent"], 2),
            "score_band": prediction["score_band"],
            "overlapping_skills": ", ".join(prediction["overlapping_skills"]),
            "missing_skills": ", ".join(prediction["missing_skills"]),
        })
    return pd.DataFrame(results)


matcher = load_matcher()
samples = load_samples()

st.title("Resume–Job Description Matching with a Shared Siamese BiLSTM")
st.caption(
    "Semantic text-pair scoring, transparent skill signals, batch inference, and top-k resume ranking."
)

st.warning(
    "Educational demonstration only. Do not upload private, sensitive, confidential, or personally "
    "identifiable resume data. This app must not be used as the sole basis for any employment decision."
)
st.info(RESPONSIBLE_USE)

with st.sidebar:
    st.header("Model status")
    st.metric("Neural artifact", "Loaded" if matcher.model_loaded else "Fallback mode")
    st.write(f"**Inference mode:** {('Siamese BiLSTM' if matcher.model_loaded else 'TF-IDF + skill overlap')} ")
    st.write(f"**Decision threshold:** {float(matcher.metadata.get('prediction_threshold', 0.5)):.2f}")
    if matcher.load_error:
        st.caption(f"Artifact note: {matcher.load_error}")
    st.divider()
    st.markdown("**Privacy rule:** Use synthetic or anonymized text only.")

single_tab, batch_tab, ranking_tab, details_tab = st.tabs(
    ["Single Match", "Batch CSV", "Rank Resumes", "Model & Limitations"]
)

with single_tab:
    sample_names = ["Custom input"] + samples["sample_name"].tolist()
    selected = st.selectbox("Choose a safe example or enter custom text", sample_names)
    default_resume = ""
    default_job = ""
    if selected != "Custom input":
        row = samples[samples["sample_name"] == selected].iloc[0]
        default_resume = row["resume_text"]
        default_job = row["job_description"]

    left, right = st.columns(2)
    with left:
        resume_text = st.text_area("Resume text", value=default_resume, height=260, placeholder="Paste anonymized resume text…")
    with right:
        job_text = st.text_area("Job description", value=default_job, height=260, placeholder="Paste a job description…")

    if st.button("Analyze Match", type="primary", use_container_width=True):
        try:
            result = matcher.predict(resume_text, job_text)
        except ValueError as exc:
            st.error(str(exc))
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prediction", result["prediction"])
            m2.metric("Fit score", f"{result['fit_score_percent']:.1f}%")
            neural = result["match_probability"]
            m3.metric("Neural probability", "N/A" if neural is None else f"{neural * 100:.1f}%")
            m4.metric("Score band", result["score_band"])
            st.progress(float(result["fit_score"]))
            st.write(result["interpretation"])

            exp_left, exp_right = st.columns(2)
            with exp_left:
                st.subheader("Overlapping skills")
                st.write(result["overlapping_skills"] or "No cataloged overlap detected.")
            with exp_right:
                st.subheader("Potential requirement gaps")
                st.write(result["missing_skills"] or "No cataloged gaps detected.")

            with st.expander("Supporting signals"):
                st.json({
                    "tfidf_similarity": round(result["tfidf_similarity"], 4),
                    "skill_coverage": round(result["skill_coverage"], 4),
                    "confidence_around_threshold": round(result["confidence"], 4),
                    "inference_mode": result["inference_mode"],
                })

with batch_tab:
    st.write("Upload a CSV containing resume and job-description columns. Common column names are detected automatically.")
    template = samples[["resume_text", "job_description"]].head(4)
    st.download_button(
        "Download batch template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="resume_job_pairs_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Batch CSV", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            scored = score_dataframe(pd.read_csv(uploaded), matcher)
        except Exception as exc:
            st.error(f"Unable to score the file: {exc}")
        else:
            st.dataframe(scored, use_container_width=True)
            chart = px.histogram(scored, x="fit_score_percent", color="score_band", nbins=10,
                                 title="Prediction Distribution")
            st.plotly_chart(chart, use_container_width=True)
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="resume_job_predictions.csv",
                mime="text/csv",
            )

with ranking_tab:
    st.write("Rank multiple anonymized resumes against one job description.")
    ranking_job = st.text_area("Job description for ranking", height=180, key="ranking_job")
    ranking_file = st.file_uploader(
        "Upload resumes CSV with columns resume_id and resume_text",
        type=["csv"],
        key="ranking_csv",
    )
    if st.button("Rank Resumes", use_container_width=True):
        if not ranking_job.strip() or ranking_file is None:
            st.error("Provide a job description and a resumes CSV.")
        else:
            frame = pd.read_csv(ranking_file)
            if "resume_text" not in frame.columns:
                st.error("The resumes CSV must contain a resume_text column.")
            else:
                if "resume_id" not in frame.columns:
                    frame["resume_id"] = [f"resume_{index + 1}" for index in range(len(frame))]
                ranked = rank_resumes(
                    ranking_job,
                    frame[["resume_id", "resume_text"]].to_dict(orient="records"),
                    matcher=matcher,
                )
                st.dataframe(ranked, use_container_width=True)
                st.download_button(
                    "Download ranking",
                    ranked.to_csv(index=False).encode("utf-8"),
                    file_name="ranked_resumes.csv",
                    mime="text/csv",
                )

with details_tab:
    st.subheader("Architecture")
    st.markdown(
        "The same embedding and Bidirectional LSTM encoder processes both inputs. "
        "The classifier compares the resulting vectors using absolute difference, element-wise product, "
        "cosine similarity, and the original embeddings."
    )
    architecture_path = PROJECT_DIR / "images" / "architecture.png"
    if architecture_path.exists():
        st.image(str(architecture_path), use_container_width=True)

    st.subheader("Limitations")
    st.markdown(
        """
- The supplied dataset contains only eight short example resumes; the included model is therefore a demonstration artifact, not a validated hiring model.
- Synthetic job descriptions and pair labels simplify real recruiting language and cannot establish generalization.
- Skill extraction uses a small transparent catalog and will miss synonyms and domain-specific requirements.
- Semantic similarity does not verify truthfulness, seniority, work authorization, soft skills, interview performance, or legal eligibility.
- Historical hiring data can encode bias even when protected attributes are removed.
        """
    )

    st.subheader("Fairness and privacy")
    st.markdown(
        "Protected attributes must not be used or inferred for ranking. Real deployments require bias testing, "
        "representative evaluation, human review, audit logs, data minimization, retention controls, candidate notice, "
        "access controls, and applicable legal review."
    )
