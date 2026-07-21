from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import standardize_inference_frame
from src.inference_pipeline import ModelArtifactError, QAMatcher

st.set_page_config(
    page_title="Siamese BiLSTM Semantic Matching",
    page_icon="🔎",
    layout="wide",
)

SAMPLE_PAIRS = {
    "Python learning — likely match": (
        "How can I learn Python quickly?",
        "What is the fastest way to learn Python?",
    ),
    "Unrelated topics — likely no match": (
        "How can I learn Python quickly?",
        "What is the capital of France?",
    ),
    "Machine learning paraphrase": (
        "What is machine learning?",
        "Can you explain the meaning of machine learning?",
    ),
}


@st.cache_resource(show_spinner="Loading the Siamese BiLSTM model...")
def load_matcher() -> QAMatcher:
    return QAMatcher.from_artifacts(PROJECT_ROOT / "models")


st.title("Siamese BiLSTM Semantic Text Matching")
st.caption(
    "A shared-encoder Bidirectional LSTM that estimates whether two texts express the same intent."
)

st.warning(
    "Responsible use: this educational model estimates semantic similarity only. It does not verify factual "
    "correctness, completeness, safety, or currency. Do not upload confidential, sensitive, private, or "
    "personally identifiable information. Do not use the output as the sole basis for legal, medical, "
    "financial, safety-critical, customer-support, or compliance decisions."
)

st.info(
    "Important scope note: the supplied model was trained on a very small synthetic duplicate-question "
    "dataset. Calling the second text an 'answer' is a transfer/demo use case, not evidence of true "
    "question-answer relevance performance. Retrain on a large QA-labelled dataset for that objective."
)

try:
    matcher = load_matcher()
except ModelArtifactError as exc:
    st.error(str(exc))
    st.stop()

manual_tab, batch_tab, ranking_tab, details_tab = st.tabs(
    ["Single Pair", "Batch CSV", "Rank Candidates", "Model Details"]
)

with manual_tab:
    sample_name = st.selectbox("Load a sample pair", ["Custom input", *SAMPLE_PAIRS.keys()])
    default_a, default_b = ("", "") if sample_name == "Custom input" else SAMPLE_PAIRS[sample_name]
    left, right = st.columns(2)
    with left:
        text_a = st.text_area("Question / Text A", value=default_a, height=150)
    with right:
        text_b = st.text_area("Candidate answer / Text B", value=default_b, height=150)

    if st.button("Evaluate semantic match", type="primary"):
        try:
            result = matcher.predict_pair(text_a, text_b)
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Prediction", result.predicted_label)
            metric_b.metric("Match probability", f"{result.match_probability:.1%}")
            metric_c.metric("Decision confidence", f"{result.confidence:.1%}")
            st.progress(min(max(result.match_probability, 0.0), 1.0))
            st.write(result.interpretation)
            st.caption(f"Decision threshold: {result.threshold:.2f}")
            if result.shared_tokens:
                st.write("Shared tokens:", ", ".join(result.shared_tokens))
            st.write(f"Lexical Jaccard overlap: {result.lexical_jaccard:.3f}")
        except ValueError as exc:
            st.error(str(exc))

with batch_tab:
    st.write("Upload a CSV with `question1`/`question2`, `question`/`answer`, or `text_a`/`text_b` columns.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            raw_frame = pd.read_csv(uploaded)
            frame = standardize_inference_frame(raw_frame)
            scored = matcher.predict_frame(frame)
            st.dataframe(scored, use_container_width=True)
            counts = scored["predicted_label"].value_counts().rename_axis("label").reset_index(name="rows")
            st.plotly_chart(px.bar(counts, x="label", y="rows", title="Prediction Distribution"), use_container_width=True)
            st.download_button(
                "Download scored CSV",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="semantic_match_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Unable to score the uploaded file: {exc}")

with ranking_tab:
    ranking_question = st.text_area(
        "Question / query",
        value="What is overfitting in machine learning?",
        height=100,
        key="ranking_question",
    )
    candidate_block = st.text_area(
        "Candidate texts — one per line",
        value=(
            "Overfitting happens when a model memorizes training data and performs poorly on unseen data.\n"
            "A confusion matrix summarizes classification outcomes.\n"
            "SQL is used to query relational databases."
        ),
        height=180,
    )
    if st.button("Rank candidates"):
        try:
            ranked = matcher.rank_candidates(ranking_question, candidate_block.splitlines())
            st.dataframe(
                ranked[["rank", "text_b", "match_probability", "predicted_label"]],
                use_container_width=True,
            )
        except ValueError as exc:
            st.error(str(exc))

with details_tab:
    metadata = matcher.metadata
    st.subheader("Architecture")
    st.image(PROJECT_ROOT / "images" / "siamese_bilstm_architecture.png")
    st.json(
        {
            "actual_training_task": metadata.get("actual_training_task"),
            "embedding_dimension": metadata.get("embedding_dimension"),
            "bilstm_units_per_direction": metadata.get("bilstm_units_per_direction"),
            "max_sequence_length": metadata.get("max_sequence_length"),
            "prediction_threshold": metadata.get("prediction_threshold"),
            "similarity_features": metadata.get("similarity_features"),
        }
    )
    st.subheader("Known limitations")
    st.markdown(
        "- The attached model was trained on only 15 synthetic pairs.\n"
        "- Its test split contains only 3 rows, so the saved metrics are not statistically meaningful.\n"
        "- Saved probabilities are clustered near 0.51, indicating weak calibration and undertraining.\n"
        "- Duplicate-question training does not establish factual answer relevance.\n"
        "- Domain shift, negation, long answers, and unseen terminology may produce unreliable scores."
    )
    st.caption("GitHub repository: replace this text with your public repository URL after publishing.")
