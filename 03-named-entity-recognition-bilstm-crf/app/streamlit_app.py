"""Interactive portfolio demo for BiLSTM NER with CRF-aware decoding."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_conll  # noqa: E402
from src.inference_pipeline import NERInferencePipeline  # noqa: E402

SAMPLES = {
    "Technology and leadership": "Apple CEO Tim Cook visited India for a technology conference.",
    "Organization, person, and location": "Microsoft hired Priya Shah to lead its Seattle research team.",
    "International affairs": "Barack Obama spoke at the United Nations in New York.",
    "Quality analytics scenario": "AquaSense engineer Maya Chen reviewed sensor failures reported in Colorado.",
}

DISCLAIMER = (
    "This educational portfolio demo can miss entities, assign incorrect labels, or produce "
    "incomplete spans. Do not upload private, sensitive, confidential, personal, medical, "
    "legal, or proprietary text. Do not use predictions as the sole basis for legal, medical, "
    "financial, hiring, compliance, surveillance, safety-critical, or other consequential decisions."
)


@st.cache_resource(show_spinner="Loading NER model artifacts...")
def load_pipeline() -> NERInferencePipeline:
    return NERInferencePipeline(PROJECT_ROOT / "models")


def render_prediction(text: str, pipeline: NERInferencePipeline) -> None:
    result = pipeline.predict_text(text)
    if result.model_kind == "legacy_bilstm_softmax":
        st.info(
            "The supplied pretrained artifact is a BiLSTM with independent softmax token outputs, "
            "not a CRF-trained model. This demo applies BIO-constrained Viterbi decoding. Run the "
            "training script to generate true CRF weights; the app will use them automatically."
        )
    left, middle, right = st.columns(3)
    left.metric("Tokens", len(result.tokens))
    middle.metric("Entities", len(result.entities))
    right.metric("Entity types", len({e["entity_type"] for e in result.entities}))

    st.subheader("Highlighted entities")
    st.markdown(
        f'<div style="line-height:2;font-size:1.05rem;padding:1rem;border:1px solid #ddd;border-radius:0.5rem;">{result.highlighted_html()}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Extracted entity spans")
    entity_frame = result.entity_frame()
    if entity_frame.empty:
        st.warning("No entity span was extracted from this input.")
    else:
        st.dataframe(entity_frame, use_container_width=True, hide_index=True)
        chart_data = entity_frame["entity_type"].value_counts().rename_axis("entity_type").reset_index(name="count")
        st.plotly_chart(
            px.bar(chart_data, x="entity_type", y="count", text="count", title="Entity type distribution"),
            use_container_width=True,
        )
        st.download_button(
            "Download entities as CSV",
            entity_frame.to_csv(index=False).encode("utf-8"),
            file_name="extracted_entities.csv",
            mime="text/csv",
        )

    with st.expander("Token-level BIO predictions", expanded=False):
        st.caption(f"Decoder: {result.decoder}. Confidence is the selected token emission probability, not calibrated sequence certainty.")
        st.dataframe(result.token_frame(), use_container_width=True, hide_index=True)


def single_text_tab(pipeline: NERInferencePipeline) -> None:
    sample_name = st.selectbox("Try a sample", list(SAMPLES), index=0)
    text = st.text_area("Enter text", value=SAMPLES[sample_name], height=150, max_chars=10_000)
    if st.button("Extract named entities", type="primary", use_container_width=True):
        try:
            render_prediction(text, pipeline)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def batch_csv_tab(pipeline: NERInferencePipeline) -> None:
    st.write("Upload a CSV containing one text example per row. Processing is limited to 500 rows per run.")
    uploaded = st.file_uploader("CSV file", type=["csv"], key="batch_csv")
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    if frame.empty:
        st.warning("The uploaded CSV is empty.")
        return
    text_column = st.selectbox("Text column", frame.columns.tolist())
    st.dataframe(frame.head(10), use_container_width=True, hide_index=True)
    if st.button("Run batch extraction", type="primary", use_container_width=True):
        texts = frame[text_column].dropna().astype(str).head(500).tolist()
        try:
            with st.spinner("Extracting entities..."):
                documents, entities = pipeline.predict_batch(texts)
            st.subheader("Document summary")
            st.dataframe(documents, use_container_width=True, hide_index=True)
            st.subheader("All extracted entities")
            st.dataframe(entities, use_container_width=True, hide_index=True)
            st.download_button(
                "Download batch entities",
                entities.to_csv(index=False).encode("utf-8"),
                file_name="batch_extracted_entities.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")


def conll_tab() -> None:
    st.write("Validate a blank-line-separated CoNLL file where the first column is the token and the last column is the BIO tag.")
    uploaded = st.file_uploader("CoNLL file", type=["conll", "txt"], key="conll")
    if uploaded is None:
        return
    temporary = PROJECT_ROOT / "outputs" / "tmp" / uploaded.name
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(uploaded.getvalue())
    try:
        sentences = load_conll(temporary)
        st.success(f"Validated {len(sentences):,} sentences and {sum(len(s.tokens) for s in sentences):,} tokens.")
        preview = [
            {"sentence_id": i, "tokens": " ".join(s.tokens), "tags": " ".join(s.tags)}
            for i, s in enumerate(sentences[:20])
        ]
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"CoNLL validation failed: {exc}")
    finally:
        temporary.unlink(missing_ok=True)


def about_tab(pipeline: NERInferencePipeline) -> None:
    st.subheader("How the model works")
    st.markdown(
        """
1. Tokens are mapped to training vocabulary IDs, with `<UNK>` for unseen words.
2. An embedding layer creates dense token representations.
3. A Bidirectional LSTM reads the sequence from both directions.
4. The emission layer scores every BIO tag at every token.
5. A trained CRF learns tag-transition scores and Viterbi selects the highest-scoring valid sequence.

The supplied notebook artifact is retained as a transparent **BiLSTM-softmax baseline**. The repository adds a true linear-chain CRF training implementation without relying on TensorFlow Addons.
        """
    )
    st.subheader("Loaded artifact")
    artifact_summary = pipeline.artifact_summary()
    if artifact_summary.get("model_artifact_available"):
        st.success(
            "A deployable NER model artifact is available in the models directory."
        )
    else:
        st.error(
            "The NER model file is missing. Add "
            "`models/legacy_bilstm_softmax_model.h5` to the repository. "
            "The app can display documentation, but predictions require the model."
        )
    st.json(artifact_summary)
    st.subheader("Supported labels")
    st.code("O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC")
    st.subheader("Limitations")
    st.markdown(
        "- Trained on newswire-style CoNLL-2003 labels, not medical, legal, resume, or quality-specific entities.\n"
        "- The label set does not contain DATE, PRODUCT, SYMPTOM, MEDICATION, SKILL, or FAILURE_MODE.\n"
        "- Lowercasing in the original training pipeline removes useful capitalization signals.\n"
        "- Long documents are processed in fixed-length chunks, which can split an entity at a chunk boundary.\n"
        "- Confidence values are not calibrated probabilities of full entity correctness."
    )


def main() -> None:
    st.set_page_config(
        page_title="BiLSTM-CRF Named Entity Recognition",
        page_icon="🏷️",
        layout="wide",
    )
    st.title("Named Entity Recognition with BiLSTM + CRF")
    st.caption("Project 03 of the Bi-Directional LSTM portfolio · CoNLL-2003 BIO sequence tagging")
    st.warning(DISCLAIMER)

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(f"Model artifacts could not be loaded: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Project information")
        st.write("**Entity types:** PER, ORG, LOC, MISC")
        st.write("**Dataset:** CoNLL-2003")
        st.write("**Task:** token-level BIO sequence tagging")
        st.write("**GitHub:** add repository URL")
        st.divider()
        st.caption("Text is processed in memory by the app. Do not submit sensitive content.")

    tab_single, tab_batch, tab_conll, tab_about = st.tabs([
        "Single text", "Batch CSV", "CoNLL validation", "Model & limitations"
    ])
    with tab_single:
        single_text_tab(pipeline)
    with tab_batch:
        batch_csv_tab(pipeline)
    with tab_conll:
        conll_tab()
    with tab_about:
        about_tab(pipeline)


if __name__ == "__main__":
    main()
