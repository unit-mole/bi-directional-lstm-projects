from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_pipeline import MedicalTextInferencePipeline  # noqa: E402

DISCLAIMER = (
    "This project is for educational and portfolio demonstration purposes only. "
    "It is not a medical diagnostic tool and must not be used to diagnose, "
    "treat, prevent, or manage any medical condition. Do not enter protected, "
    "private, confidential, or personally identifiable health information."
)

SAFE_EXAMPLES = {
    "Cardiology-style example": (
        "Patient reports exertional chest discomfort, palpitations, and a "
        "history of hypertension. ECG review is recommended."
    ),
    "Gastroenterology-style example": (
        "Persistent upper abdominal discomfort, reflux after meals, nausea, "
        "and endoscopy findings consistent with gastritis."
    ),
    "Neurology-style example": (
        "Recurrent headache with dizziness, sensory changes, and weakness in "
        "the right upper limb on neurological examination."
    ),
    "Orthopedic-style example": (
        "Knee pain increases with movement. MRI findings suggest a meniscal "
        "injury and possible ligament strain."
    ),
    "Radiology-style example": (
        "CT imaging demonstrates no acute intracranial hemorrhage and stable "
        "ventricular size."
    ),
}


@st.cache_resource(show_spinner=False)
def load_pipeline() -> MedicalTextInferencePipeline:
    pipeline = MedicalTextInferencePipeline()
    pipeline.load()
    return pipeline


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        .hero {
            padding: 1.3rem 1.5rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #f3f8fd 0%, #ffffff 100%);
            border: 1px solid #d9e6f2;
            margin-bottom: 1rem;
        }
        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid #d9e6f2;
            background: #ffffff;
        }
        .small-note {font-size: 0.9rem; color: #52606d;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_probability_chart(top_probabilities: list[tuple[str, float]]) -> None:
    frame = pd.DataFrame(
        top_probabilities,
        columns=["Medical category", "Probability"],
    ).set_index("Medical category")
    st.bar_chart(frame, horizontal=True)


def _render_attention_terms(terms: list[tuple[str, float]]) -> None:
    if not terms:
        st.info(
            "Attention terms could not be extracted from this model runtime. "
            "The category probabilities remain available."
        )
        return

    frame = pd.DataFrame(terms, columns=["Term", "Attention weight"])
    st.dataframe(
        frame.style.format({"Attention weight": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Attention weights are model internals, not validated clinical "
        "explanations or evidence of causality."
    )


def _manual_prediction_tab(pipeline: MedicalTextInferencePipeline) -> None:
    st.subheader("Single-text prediction")
    selected_example = st.selectbox(
        "Start with a safe synthetic example",
        ["Write my own text", *SAFE_EXAMPLES.keys()],
    )
    default_text = (
        ""
        if selected_example == "Write my own text"
        else SAFE_EXAMPLES[selected_example]
    )
    medical_text = st.text_area(
        "Medical text",
        value=default_text,
        height=180,
        placeholder=(
            "Enter a synthetic or non-sensitive clinical-style passage. "
            "Do not enter real patient identifiers or confidential records."
        ),
    )

    if st.button("Classify medical text", type="primary"):
        if not medical_text.strip():
            st.warning("Enter medical text before running a prediction.")
            return

        with st.spinner("Loading model artifacts and generating prediction..."):
            result = pipeline.predict(
                medical_text,
                top_k=3,
                include_attention=True,
            )

        st.markdown(
            f"""
            <div class="result-card">
              <div class="small-note">Predicted medical category</div>
              <h2>{result.predicted_label}</h2>
              <div><strong>Confidence:</strong> {result.confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("#### Top category probabilities")
            _render_probability_chart(result.top_probabilities)
        with right:
            st.markdown("#### Important terms from attention")
            _render_attention_terms(result.important_terms)

        if result.confidence < 0.50:
            st.warning(
                "The confidence is low. The supplied portfolio artifact was "
                "trained on only ten demonstration rows and is not suitable "
                "for real-world classification."
            )

        st.markdown("#### Interpretation")
        st.write(
            "The model assigns probabilities across the five learned medical "
            "specialty labels and selects the highest-scoring category. The "
            "attention section, when available, shows tokens that received "
            "larger internal weights. This output is a machine-learning result, "
            "not medical advice."
        )


def _batch_prediction_tab(pipeline: MedicalTextInferencePipeline) -> None:
    st.subheader("Batch CSV prediction")
    st.write(
        "Upload a CSV containing one text column. The file is processed only "
        "for the current app session. Do not upload protected health information."
    )
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        st.info(
            "Expected format: a CSV with a column such as `transcription`, "
            "`clinical_text`, `medical_text`, `note`, or `text`."
        )
        return

    try:
        dataframe = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Unable to read the CSV: {exc}")
        return

    st.dataframe(dataframe.head(10), use_container_width=True)
    if dataframe.shape[1] == 0:
        st.error("The uploaded CSV does not contain any columns.")
        return

    text_candidates = [
        column
        for column in dataframe.columns
        if any(
            token in column.lower()
            for token in (
                "text",
                "transcription",
                "note",
                "abstract",
                "description",
            )
        )
    ]
    default_index = (
        dataframe.columns.get_loc(text_candidates[0])
        if text_candidates
        else 0
    )
    text_column = st.selectbox(
        "Select the medical text column",
        dataframe.columns.tolist(),
        index=int(default_index),
    )

    valid_texts = dataframe[text_column].dropna().astype(str)
    if valid_texts.empty:
        st.warning("The selected text column does not contain any non-empty values.")
        return

    if len(valid_texts) > 2_000:
        st.warning(
            "For a public demo, batch scoring is limited to the first 2,000 "
            "non-empty rows."
        )
        valid_texts = valid_texts.head(2_000)

    if st.button("Generate batch predictions", type="primary"):
        with st.spinner(f"Scoring {len(valid_texts):,} rows..."):
            prediction_frame = pipeline.predict_batch(valid_texts, top_k=3)

        st.success(f"Generated {len(prediction_frame):,} predictions.")
        st.dataframe(prediction_frame, use_container_width=True)

        st.markdown("#### Predicted-category distribution")
        distribution = (
            prediction_frame["predicted_label"]
            .value_counts()
            .rename_axis("Medical category")
            .to_frame("Predictions")
        )
        st.bar_chart(distribution, horizontal=True)

        output_buffer = io.StringIO()
        prediction_frame.to_csv(output_buffer, index=False)
        st.download_button(
            "Download scored CSV",
            data=output_buffer.getvalue().encode("utf-8"),
            file_name="medical_text_batch_predictions.csv",
            mime="text/csv",
        )


def _render_model_details(pipeline: MedicalTextInferencePipeline) -> None:
    metadata = pipeline.metadata
    with st.expander("Model details and limitations"):
        st.markdown(
            f"""
            - **Architecture:** {metadata.get("model_type")}
            - **Classes:** {", ".join(metadata.get("class_labels", []))}
            - **Maximum sequence length:** {metadata.get("max_sequence_length")}
            - **Embedding dimension:** {metadata.get("embedding_dimension")}
            - **BiLSTM units per direction:** {metadata.get("bilstm_units")}
            - **Preprocessing compatibility mode:** {metadata.get("preprocessing_mode")}
            - **Artifact status:** {metadata.get("artifact_status")}
            """
        )
        st.warning(
            "The bundled model is a pipeline demonstration trained on ten "
            "synthetic rows—two per class. Its holdout consisted of only three "
            "rows, so the included metrics are not statistically meaningful. "
            "Replace it with a model trained and validated on an appropriately "
            "licensed, de-identified, representative dataset before making "
            "performance claims."
        )


def main() -> None:
    st.set_page_config(
        page_title="Medical Text Classification | BiLSTM + Attention",
        page_icon="🩺",
        layout="wide",
    )
    _apply_page_style()

    st.markdown(
        """
        <div class="hero">
          <h1>Medical Text Classification with BiLSTM + Attention</h1>
          <p>
            A portfolio-ready healthcare NLP demo that classifies text into
            Cardiology, Gastroenterology, Neurology, Orthopedic, or Radiology.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(DISCLAIMER)

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(
            "The saved model artifacts could not be loaded. Install the pinned "
            "requirements and confirm that all files under `models/` are "
            f"present. Technical details: {exc}"
        )
        st.stop()

    manual_tab, batch_tab, about_tab = st.tabs(
        ["Single prediction", "Batch CSV", "About the project"]
    )

    with manual_tab:
        _manual_prediction_tab(pipeline)
    with batch_tab:
        _batch_prediction_tab(pipeline)
    with about_tab:
        st.markdown(
            """
            ### What this project demonstrates

            - Bidirectional LSTM sequence modeling
            - A custom temporal attention layer
            - Multi-class probability output
            - Consistent training/inference preprocessing
            - Saved model, tokenizer, metadata, and label mapping artifacts
            - Batch scoring and downloadable results
            - Privacy-aware and responsible AI communication

            ### Portfolio connection

            The same technical pattern can support complaint-text routing,
            quality-event categorization, failure-description classification,
            customer-comment analysis, and root-cause narrative triage.
            """
        )
        _render_model_details(pipeline)

    st.caption(
        "Educational portfolio demonstration only · Not a diagnostic tool · "
        "Do not enter private or identifiable health information."
    )


if __name__ == "__main__":
    main()
