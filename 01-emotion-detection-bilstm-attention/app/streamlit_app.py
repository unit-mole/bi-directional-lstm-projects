"""Interactive Streamlit app for BiLSTM-attention emotion prediction."""

from __future__ import annotations

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
    "This educational portfolio model can misinterpret sarcasm, mixed emotions, "
    "cultural context, and ambiguous language. Do not submit private or sensitive "
    "text, and do not use predictions for diagnosis or other high-stakes decisions."
)

SAMPLES = {
    "Joy": "I am extremely happy and excited today.",
    "Fear": "I feel worried and anxious about the upcoming examination.",
    "Sadness": "I feel lonely and heartbroken tonight.",
    "Anger": "I am furious about the unfair decision.",
    "Love": "I adore my family and feel so close to them.",
    "Surprise": "The unexpected announcement left me stunned.",
}


st.set_page_config(
    page_title="Emotion Detection | BiLSTM + Attention",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained BiLSTM-attention model...")
def load_pipeline() -> EmotionInferencePipeline:
    """Load and cache the trained inference pipeline."""
    return EmotionInferencePipeline(PROJECT_ROOT / "models").load()


def probability_chart(probabilities: dict[str, float]):
    """Build a horizontal class-probability chart."""
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
    figure.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return figure


def attention_chart(items: list[tuple[str, float]]):
    """Build a token-level attention chart."""
    frame = pd.DataFrame(items, columns=["Token", "Attention"]).sort_values(
        "Attention"
    )

    return px.bar(
        frame,
        x="Attention",
        y="Token",
        orientation="h",
        text=frame["Attention"].map(lambda value: f"{value:.3f}"),
        title="Most influential tokens",
    )


try:
    pipeline = load_pipeline()
except ArtifactError as exc:
    st.error(str(exc))
    st.stop()


st.title("🧠 Emotion Detection using BiLSTM with Attention")
st.caption(
    "Six-class text emotion classification with a trained bidirectional LSTM, "
    "temporal attention, probability estimates, and batch scoring."
)
st.warning(RESPONSIBLE_USE, icon="⚠️")


with st.sidebar:
    st.header("Model Snapshot")

    st.markdown(
        f"""
**Classes:** {", ".join(label.title() for label in pipeline.classes)}

**Architecture:** BiLSTM + temporal attention

**Maximum length:** {pipeline.max_sequence_length} tokens

**Model status:** Educational demonstration checkpoint

**Training data:** Balanced template-augmented dataset

**Evaluation:** Synthetic holdout only — not a real-world benchmark
"""
    )

    st.info(
        "The bundled checkpoint is provided to demonstrate the complete "
        "training, inference, attention, and Streamlit workflow. Its synthetic "
        "holdout results should not be interpreted as real-world performance."
    )

    st.markdown(
        "[GitHub repository]"
        "(https://github.com/unit-mole/bi-directional-lstm-projects)"
    )


tab_single, tab_batch, tab_model = st.tabs(
    ["Single Text", "Batch CSV", "Model & Responsible Use"]
)


with tab_single:
    st.subheader("Single-text prediction")

    sample = st.selectbox(
        "Load a sample",
        ["Custom text", *SAMPLES.keys()],
    )
    default_text = "" if sample == "Custom text" else SAMPLES[sample]

    text = st.text_area(
        "Enter a sentence, message, review, or feedback comment",
        value=default_text,
        height=140,
    )

    if st.button(
        "Predict emotion",
        type="primary",
        use_container_width=True,
    ):
        if not text.strip():
            st.error("Enter text before requesting a prediction.")
        else:
            result = pipeline.predict(text)
            left_column, right_column = st.columns([1, 2])

            with left_column:
                st.metric(
                    "Predicted emotion",
                    result.predicted_emotion.title(),
                )
                st.metric(
                    "Confidence",
                    f"{result.confidence:.1%}",
                )

                if result.confidence < 0.55:
                    st.info(
                        "Confidence is low. Treat the result as uncertain "
                        "and review competing classes."
                    )
                elif result.confidence >= 0.80:
                    st.success(
                        "The model produced a strong class probability "
                        "for this input."
                    )

            with right_column:
                st.plotly_chart(
                    probability_chart(result.probabilities),
                    use_container_width=True,
                )

            st.markdown(f"**Interpretation:** {result.interpretation()}")

            if result.important_tokens:
                st.plotly_chart(
                    attention_chart(result.important_tokens),
                    use_container_width=True,
                )


with tab_batch:
    st.subheader("Batch CSV prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV with a text column",
        type=["csv"],
    )

    if uploaded_file is None:
        st.download_button(
            "Download CSV template",
            "text\n"
            "I am happy about the result.\n"
            "I feel nervous about tomorrow.\n",
            file_name="emotion_template.csv",
            mime="text/csv",
        )
    else:
        input_frame = pd.read_csv(uploaded_file)
        candidate_columns = [
            column
            for column in input_frame.columns
            if str(column).lower()
            in {"text", "sentence", "message", "comment", "content"}
        ]

        default_index = (
            list(input_frame.columns).index(candidate_columns[0])
            if candidate_columns
            else 0
        )

        text_column = st.selectbox(
            "Text column",
            list(input_frame.columns),
            index=default_index,
        )

        if st.button(
            "Run batch prediction",
            type="primary",
            use_container_width=True,
        ):
            valid_rows = (
                input_frame[text_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
            )

            results = pipeline.predict_many(
                input_frame.loc[valid_rows, text_column].astype(str)
            )
            scored_frame = input_frame.loc[valid_rows].copy()

            scored_frame["predicted_emotion"] = [
                result.predicted_emotion for result in results
            ]
            scored_frame["confidence"] = [
                result.confidence for result in results
            ]

            for label in pipeline.classes:
                scored_frame[f"probability_{label}"] = [
                    result.probabilities[label] for result in results
                ]

            st.dataframe(
                scored_frame,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download scored CSV",
                scored_frame.to_csv(index=False).encode("utf-8"),
                file_name="emotion_predictions.csv",
                mime="text/csv",
            )


with tab_model:
    st.subheader("Architecture and artifact details")

    st.json(
        {
            "model_type": pipeline.metadata.get("model_type"),
            "framework": pipeline.metadata.get("framework"),
            "classes": pipeline.classes,
            "vocabulary_size": pipeline.metadata.get("vocabulary_size"),
            "max_sequence_length": pipeline.max_sequence_length,
            "dataset_note": pipeline.metadata.get("dataset_note"),
        }
    )

    metrics = pipeline.metadata.get("evaluation_metrics", {})

    with st.expander("Synthetic holdout results"):
        st.warning(
            "These values were calculated on a balanced, template-augmented "
            "synthetic holdout set. They are useful for validating the packaged "
            "workflow, but they are not a real-world benchmark."
        )

        if metrics:
            metric_column_1, metric_column_2 = st.columns(2)

            with metric_column_1:
                st.metric(
                    "Synthetic holdout accuracy",
                    f"{metrics.get('accuracy', 0):.1%}",
                )

            with metric_column_2:
                st.metric(
                    "Synthetic holdout macro F1",
                    f"{metrics.get('macro_f1', 0):.1%}",
                )
        else:
            st.info("No packaged holdout metrics were found.")

    st.markdown("### Responsible use")
    st.write(RESPONSIBLE_USE)

    st.markdown("### Recommended next step")
    st.write(
        "Retrain and evaluate the model on a licensed, independently collected "
        "real-world emotion corpus before making research or performance claims."
    )
