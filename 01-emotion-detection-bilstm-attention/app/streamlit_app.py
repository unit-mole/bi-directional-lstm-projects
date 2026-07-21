"""Interactive Streamlit demo for single and batch emotion prediction."""

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

from src.inference_pipeline import ArtifactError, EmotionInferencePipeline

RESPONSIBLE_USE = (
    "This educational portfolio model may misinterpret tone, sarcasm, cultural context, "
    "mixed emotions, or ambiguous language. Do not submit private, sensitive, confidential, "
    "or personally identifiable text. Do not use predictions as the sole basis for mental-health, "
    "hiring, insurance, legal, surveillance, or other high-stakes decisions."
)

SAMPLES = {
    "Joy": "I am genuinely excited and grateful for this new opportunity!",
    "Fear": "I feel nervous and afraid about what might happen tomorrow.",
    "Sadness": "I feel lonely and deeply disappointed after everything that happened.",
    "Anger": "I am furious that the same problem keeps happening again!",
    "Surprise": "I cannot believe the unexpected news I just received!",
    "Calm": "I feel peaceful, relaxed, and comfortable this morning.",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Emotion Detection | BiLSTM + Attention",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_pipeline() -> EmotionInferencePipeline:
    return EmotionInferencePipeline(PROJECT_ROOT / "models").load()


def render_header() -> None:
    st.title("🧠 Emotion Detection using BiLSTM with Attention")
    st.caption(
        "Multi-class NLP classification with bidirectional sequence modeling, "
        "probability estimates, and optional token-level attention visualization."
    )
    st.warning(RESPONSIBLE_USE, icon="⚠️")


def render_sidebar(pipeline: EmotionInferencePipeline) -> None:
    with st.sidebar:
        st.header("Model Snapshot")
        st.write(f"**Available classes:** {', '.join(label.title() for label in pipeline.classes)}")
        st.write(f"**Max sequence length:** {pipeline.max_sequence_length}")
        st.write(f"**Attention visualization:** {'Available' if pipeline.supports_attention else 'Unavailable'}")
        st.write(f"**Artifact mode:** `{pipeline.artifact_status}`")
        st.divider()
        st.markdown("**Portfolio links**")
        st.markdown("[GitHub repository](https://github.com/USERNAME/bi-directional-lstm-projects)")
        st.caption("Replace `USERNAME` after publishing the repository.")


def probability_chart(probabilities: dict[str, float]):
    frame = pd.DataFrame(
        sorted(probabilities.items(), key=lambda item: item[1], reverse=True),
        columns=["Emotion", "Probability"],
    )
    frame["Emotion"] = frame["Emotion"].str.title()
    figure = px.bar(
        frame,
        x="Probability",
        y="Emotion",
        orientation="h",
        text=frame["Probability"].map(lambda value: f"{value:.1%}"),
        range_x=[0, 1],
    )
    figure.update_layout(yaxis={"categoryorder": "total ascending"}, height=320)
    return figure


def render_single_prediction(pipeline: EmotionInferencePipeline) -> None:
    st.subheader("Single-text prediction")
    selected_sample = st.selectbox("Load a sample", ["Custom text", *SAMPLES.keys()])
    default_text = "" if selected_sample == "Custom text" else SAMPLES[selected_sample]
    text = st.text_area(
        "Enter a sentence, message, review, or feedback comment",
        value=default_text,
        height=150,
        placeholder="Example: I am so excited about starting this new opportunity!",
    )

    if st.button("Predict emotion", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Enter some text before requesting a prediction.")
            return
        with st.spinner("Analyzing text..."):
            prediction = pipeline.predict(text)

        first, second = st.columns([1, 2])
        with first:
            st.metric("Predicted emotion", prediction.predicted_emotion.title())
            st.metric("Confidence", f"{prediction.confidence:.1%}")
            if prediction.confidence < 0.55:
                st.info("Confidence is low. Treat the result as uncertain and review competing classes.")
        with second:
            st.plotly_chart(
                probability_chart(dict(prediction.top_probabilities[: max(3, len(prediction.probabilities))])),
                use_container_width=True,
            )

        st.markdown(f"**Interpretation:** {prediction.interpretation()}")
        if prediction.important_tokens:
            attention_frame = pd.DataFrame(prediction.important_tokens, columns=["Token", "Attention"])
            st.markdown("#### Most influential tokens")
            st.dataframe(
                attention_frame.style.format({"Attention": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(
                "Token-level attention is unavailable for the supplied legacy checkpoint. "
                "Train the upgraded attention architecture to enable this section."
            )


def _detect_text_column(frame: pd.DataFrame) -> str | None:
    candidates = ["text", "sentence", "message", "tweet", "content", "comment"]
    normalized = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def render_batch_prediction(pipeline: EmotionInferencePipeline) -> None:
    st.subheader("Batch CSV prediction")
    st.write("Upload a CSV containing a text column. The app adds predicted emotion and class probabilities.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if uploaded is None:
        st.download_button(
            "Download CSV template",
            data="text\nI am excited about this result!\nI feel worried about tomorrow.\n",
            file_name="emotion_prediction_template.csv",
            mime="text/csv",
        )
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    if frame.empty:
        st.error("The uploaded CSV has no rows.")
        return

    detected = _detect_text_column(frame)
    selected = st.selectbox(
        "Text column",
        options=list(frame.columns),
        index=list(frame.columns).index(detected) if detected in frame.columns else 0,
    )
    valid_mask = frame[selected].notna() & frame[selected].astype(str).str.strip().ne("")
    st.caption(f"{int(valid_mask.sum())} valid text rows detected out of {len(frame)} rows.")

    if st.button("Run batch prediction", type="primary", use_container_width=True):
        if not valid_mask.any():
            st.error("No non-empty text rows were found.")
            return
        results = pipeline.predict_many(frame.loc[valid_mask, selected].astype(str).tolist())
        scored = frame.loc[valid_mask].copy().reset_index(drop=True)
        scored["predicted_emotion"] = [result.predicted_emotion for result in results]
        scored["confidence"] = [result.confidence for result in results]
        for label in pipeline.classes:
            scored[f"probability_{label}"] = [result.probabilities[label] for result in results]

        st.dataframe(scored, use_container_width=True, hide_index=True)
        distribution = scored["predicted_emotion"].value_counts().rename_axis("Emotion").reset_index(name="Count")
        st.plotly_chart(px.bar(distribution, x="Emotion", y="Count", text="Count"), use_container_width=True)
        st.download_button(
            "Download scored CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="emotion_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_model_notes(pipeline: EmotionInferencePipeline) -> None:
    st.subheader("Architecture and limitations")
    st.markdown(
        """
        **Intended architecture**

        `Text → emotion-aware normalization → tokenizer → padded sequence → embedding → BiLSTM → temporal attention → dense classifier → softmax probabilities`

        **Why attention?** The attention layer assigns a normalized weight to each BiLSTM timestep, allowing the model to aggregate context from emotionally informative words instead of relying only on one final hidden state.

        **Evaluation priorities**

        Accuracy is reported alongside macro F1, weighted F1, per-class recall, confusion matrices, and error examples. Macro F1 is especially important when minority emotion classes would otherwise be hidden by overall accuracy.
        """
    )
    if pipeline.artifact_status == "legacy_bilstm_without_attention":
        st.error(
            "The currently bundled checkpoint is the original limited artifact: 7 retained training rows, "
            "3 output classes, no attention layer, and unreliable evaluation. It is included only so the "
            "app can demonstrate artifact loading. Retrain on the complete licensed dataset before publishing "
            "performance claims."
        )
    st.markdown("#### Responsible-use reminder")
    st.write(RESPONSIBLE_USE)


def main() -> None:
    configure_page()
    render_header()
    try:
        pipeline = load_pipeline()
    except (ArtifactError, ImportError, OSError, ValueError) as exc:
        st.error("The model artifacts could not be loaded.")
        st.code(str(exc))
        st.info("Run `python scripts/train_model.py --data data/your_dataset.csv` and restart the app.")
        st.stop()

    render_sidebar(pipeline)
    if pipeline.artifact_status == "legacy_bilstm_without_attention":
        st.warning(
            "Legacy demonstration mode is active. The supplied checkpoint supports only Fear, Joy, and Sadness "
            "and does not contain attention. See the Model & Responsible Use tab for details.",
            icon="🧪",
        )

    single_tab, batch_tab, notes_tab = st.tabs(
        ["Single Text", "Batch CSV", "Model & Responsible Use"]
    )
    with single_tab:
        render_single_prediction(pipeline)
    with batch_tab:
        render_batch_prediction(pipeline)
    with notes_tab:
        render_model_notes(pipeline)


if __name__ == "__main__":
    main()
