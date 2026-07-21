from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import identifier_baseline
from src.code_preprocessing import validate_python_code
from src.inference_pipeline import ArtifactError, CodeCommentInferencePipeline
from src.model_evaluation import evaluate_pair

st.set_page_config(
    page_title="Code Comment Generation | BiLSTM",
    page_icon="🧠",
    layout="wide",
)

SAMPLES = {
    "Addition": "def add_numbers(a, b):\n    return a + b",
    "Average": "def calculate_average(values):\n    if not values:\n        return 0\n    return sum(values) / len(values)",
    "Even filter": "def keep_even_numbers(values):\n    return [value for value in values if value % 2 == 0]",
    "File extension": "def get_extension(path):\n    return path.rsplit('.', 1)[-1].lower()",
}


@st.cache_resource(show_spinner=False)
def get_pipeline() -> CodeCommentInferencePipeline:
    return CodeCommentInferencePipeline(PROJECT_ROOT).load()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def sidebar(metadata: dict) -> None:
    st.sidebar.title("Project information")
    st.sidebar.markdown("**Architecture**")
    st.sidebar.write(metadata.get("architecture_display", "BiLSTM encoder-decoder"))
    st.sidebar.markdown("**Dataset**")
    st.sidebar.write("CodeSearchNet — Python functions and documentation strings")
    st.sidebar.markdown("**Checkpoint status**")
    if metadata.get("attention_available"):
        st.sidebar.success("True attention checkpoint")
    else:
        st.sidebar.warning("Legacy checkpoint: no attention layer")
    st.sidebar.markdown("**Known limitations**")
    for item in metadata.get("known_limitations", [])[:5]:
        st.sidebar.caption(f"• {item}")


def render_attention(attention, generated_tokens):
    if attention is None or not generated_tokens:
        st.info("Attention visualization is unavailable for the supplied legacy checkpoint.")
        return
    frame = pd.DataFrame(attention[: len(generated_tokens), :])
    frame.index = generated_tokens[: len(frame)]
    frame.columns = [f"src_{i+1}" for i in range(frame.shape[1])]
    figure = px.imshow(frame, aspect="auto", labels={"x": "Source token position", "y": "Generated token"})
    st.plotly_chart(figure, use_container_width=True)


def main() -> None:
    metadata = load_json(PROJECT_ROOT / "models" / "model_metadata.json")
    sidebar(metadata)

    st.title("Code Comment Generation using BiLSTM with Attention")
    st.caption("A code-to-text sequence generation portfolio project for Python functions.")

    st.error(
        "Responsible use: Generated comments may be incomplete, inaccurate, or misleading. "
        "The model does not verify correctness, security, performance, licensing, or production readiness. "
        "Do not paste proprietary, confidential, private, or copyrighted source code into this public demo. "
        "A developer must review every output before use."
    )

    if metadata.get("checkpoint_kind") == "legacy_final_state_seq2seq":
        st.warning(
            "Technical audit notice: the supplied trained checkpoint is a BiLSTM encoder-decoder without an "
            "attention layer, despite the original notebook title. This app loads it for reproducibility. "
            "The repository includes a corrected Bahdanau-attention training pipeline; retrain before presenting "
            "attention visualizations or claiming attention-model results."
        )

    tab_demo, tab_batch, tab_results, tab_about = st.tabs(
        ["Interactive demo", "Batch demo", "Training results", "Architecture & limitations"]
    )

    with tab_demo:
        col_input, col_controls = st.columns([3, 1])
        with col_controls:
            sample_name = st.selectbox("Safe sample", list(SAMPLES))
            method = st.radio("Decoding", ["Greedy", "Beam search"], horizontal=False)
            beam_width = st.slider("Beam width", 2, 5, 3, disabled=method == "Greedy")
            reference = st.text_input("Optional reference comment")
        with col_input:
            code = st.text_area("Python function", value=SAMPLES[sample_name], height=260)

        valid, syntax_error = validate_python_code(code)
        if not valid:
            st.warning(f"Python syntax check: {syntax_error}. Generation can still be attempted.")

        if st.button("Generate comment", type="primary", use_container_width=True):
            if not code.strip():
                st.warning("Enter a code snippet first.")
            else:
                baseline = identifier_baseline(code)
                try:
                    started = time.perf_counter()
                    result = get_pipeline().generate(
                        code,
                        method="beam" if method == "Beam search" else "greedy",
                        beam_width=beam_width,
                    )
                    latency = time.perf_counter() - started
                    generated = result.comment or "No comment token was generated."
                    st.subheader("Generated comment")
                    st.success(generated)
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Comment words", len(generated.split()))
                    metric_cols[1].metric("Latency", f"{latency:.2f} s")
                    metric_cols[2].metric("Decoder", result.decoding_method)
                    metric_cols[3].metric("Attention", "Available" if result.attention is not None else "Unavailable")
                    for warning in result.warnings:
                        st.warning(warning)
                    with st.expander("Transparent identifier baseline"):
                        st.write(baseline)
                        st.caption("This rule-based baseline is displayed for comparison and is not neural output.")
                    if reference.strip():
                        st.json(evaluate_pair(reference.lower().strip(), result.comment))
                    render_attention(result.attention, result.tokens)
                except ArtifactError as exc:
                    st.error(str(exc))
                    st.info(f"Baseline result: {baseline}")
                except Exception as exc:
                    st.exception(exc)

    with tab_batch:
        st.write("Upload a CSV containing a `code` column. Processing is limited to 20 rows in the public demo.")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            frame = pd.read_csv(uploaded)
            if "code" not in frame.columns:
                st.error("The uploaded CSV must contain a `code` column.")
            else:
                preview = frame.head(20).copy()
                if st.button("Generate batch comments"):
                    try:
                        pipeline = get_pipeline()
                        preview["generated_comment"] = [
                            pipeline.generate(code, method="greedy").comment for code in preview["code"].astype(str)
                        ]
                        st.dataframe(preview, use_container_width=True)
                        st.download_button(
                            "Download batch results",
                            preview.to_csv(index=False).encode("utf-8"),
                            file_name="code_comment_batch_results.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.exception(exc)

    with tab_results:
        history_path = PROJECT_ROOT / "outputs" / "code_comment_training_history.csv"
        metrics_path = PROJECT_ROOT / "outputs" / "model_metrics.json"
        if history_path.exists():
            history = pd.read_csv(history_path)
            st.plotly_chart(
                px.line(history.reset_index(names="epoch"), x="epoch", y=["loss", "val_loss"], markers=True),
                use_container_width=True,
            )
            st.dataframe(history, use_container_width=True)
        st.json(load_json(metrics_path))
        st.caption(
            "The original token accuracy includes padding positions and must not be interpreted as semantic "
            "comment-generation accuracy. BLEU and qualitative error analysis are more informative here."
        )

    with tab_about:
        st.markdown(
            """
            ### Corrected architecture
            `Python code → semantic lexical tokens → embedding → bidirectional LSTM encoder → Bahdanau attention → LSTM decoder → token softmax`

            ### Why the audit matters
            The original notebook used the target docstring inside `func_code_string`, removed code operators through the default Keras tokenizer filters, measured unmasked token accuracy, and did not include an attention layer. The modular retraining path fixes those issues.

            ### Appropriate use
            This project is an educational demonstration of recurrent code-to-text generation. It is not a code reviewer, security scanner, compiler, or production documentation system.
            """
        )


if __name__ == "__main__":
    main()
