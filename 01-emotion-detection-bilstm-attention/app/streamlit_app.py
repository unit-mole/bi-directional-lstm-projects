"""Interactive Streamlit app for BiLSTM-attention emotion prediction."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from src.inference_pipeline import ArtifactError, EmotionInferencePipeline

RESPONSIBLE_USE="This educational portfolio model can misinterpret sarcasm, mixed emotions, cultural context, and ambiguous language. Do not submit private or sensitive text, and do not use predictions for diagnosis or other high-stakes decisions."
SAMPLES={"Joy":"I am extremely happy and excited today.","Fear":"I feel worried and anxious about the upcoming examination.","Sadness":"I feel lonely and heartbroken tonight.","Anger":"I am furious about the unfair decision.","Love":"I adore my family and feel so close to them.","Surprise":"The unexpected announcement left me stunned."}

st.set_page_config(page_title="Emotion Detection | BiLSTM + Attention",page_icon="🧠",layout="wide")

@st.cache_resource(show_spinner="Loading trained BiLSTM-attention model...")
def load_pipeline(): return EmotionInferencePipeline(PROJECT_ROOT/"models").load()

def probability_chart(probabilities):
    frame=pd.DataFrame(sorted(probabilities.items(),key=lambda x:x[1],reverse=True),columns=["Emotion","Probability"]); frame["Emotion"]=frame["Emotion"].str.title()
    fig=px.bar(frame,x="Probability",y="Emotion",orientation="h",text=frame["Probability"].map(lambda x:f"{x:.1%}"),range_x=[0,1]); fig.update_layout(yaxis={"categoryorder":"total ascending"},height=360,margin=dict(l=10,r=10,t=20,b=10)); return fig

def attention_chart(items):
    frame=pd.DataFrame(items,columns=["Token","Attention"]).sort_values("Attention")
    return px.bar(frame,x="Attention",y="Token",orientation="h",text=frame["Attention"].map(lambda x:f"{x:.3f}"),title="Most influential tokens")

try: pipeline=load_pipeline()
except ArtifactError as exc: st.error(str(exc)); st.stop()

st.title("🧠 Emotion Detection using BiLSTM with Attention")
st.caption("Six-class text emotion classification with a trained bidirectional LSTM, temporal attention, probability estimates, and batch scoring.")
st.warning(RESPONSIBLE_USE,icon="⚠️")

with st.sidebar:
    st.header("Model Snapshot")
    st.write(f"**Classes:** {', '.join(x.title() for x in pipeline.classes)}")
    st.write(f"**Architecture:** BiLSTM + temporal attention")
    st.write(f"**Maximum length:** {pipeline.max_sequence_length} tokens")
    metrics=pipeline.metadata.get("evaluation_metrics",{})
    if metrics:
        st.metric("Bundled test accuracy",f"{metrics.get('accuracy',0):.1%}")
        st.metric("Bundled macro F1",f"{metrics.get('macro_f1',0):.1%}")
    st.caption("Bundled checkpoint uses a balanced template-augmented educational dataset. Replace it with a licensed real-world corpus for research conclusions.")
    st.markdown("[GitHub repository](https://github.com/unit-mole/bi-directional-lstm-projects)")

tab_single,tab_batch,tab_model=st.tabs(["Single Text","Batch CSV","Model & Responsible Use"])
with tab_single:
    st.subheader("Single-text prediction")
    sample=st.selectbox("Load a sample",["Custom text",*SAMPLES.keys()]); default="" if sample=="Custom text" else SAMPLES[sample]
    text=st.text_area("Enter a sentence, message, review, or feedback comment",value=default,height=140)
    if st.button("Predict emotion",type="primary",use_container_width=True):
        if not text.strip(): st.error("Enter text before requesting a prediction.")
        else:
            result=pipeline.predict(text)
            left,right=st.columns([1,2])
            with left:
                st.metric("Predicted emotion",result.predicted_emotion.title()); st.metric("Confidence",f"{result.confidence:.1%}")
                if result.confidence<0.55: st.info("Confidence is low. Treat the result as uncertain and review competing classes.")
                elif result.confidence>=0.80: st.success("The model produced a strong class probability for this input.")
            with right: st.plotly_chart(probability_chart(result.probabilities),use_container_width=True)
            st.markdown(f"**Interpretation:** {result.interpretation()}")
            if result.important_tokens: st.plotly_chart(attention_chart(result.important_tokens),use_container_width=True)
with tab_batch:
    st.subheader("Batch CSV prediction")
    uploaded=st.file_uploader("Upload a CSV with a text column",type=["csv"])
    if uploaded is None:
        st.download_button("Download CSV template", "text\nI am happy about the result.\nI feel nervous about tomorrow.\n", file_name="emotion_template.csv", mime="text/csv")
    else:
        frame=pd.read_csv(uploaded); candidates=[c for c in frame.columns if str(c).lower() in {"text","sentence","message","comment","content"}]; column=st.selectbox("Text column",list(frame.columns),index=list(frame.columns).index(candidates[0]) if candidates else 0)
        if st.button("Run batch prediction",type="primary",use_container_width=True):
            valid=frame[column].fillna("").astype(str).str.strip().ne(""); results=pipeline.predict_many(frame.loc[valid,column].astype(str)); scored=frame.loc[valid].copy(); scored["predicted_emotion"]=[r.predicted_emotion for r in results]; scored["confidence"]=[r.confidence for r in results]
            for label in pipeline.classes: scored[f"probability_{label}"]=[r.probabilities[label] for r in results]
            st.dataframe(scored,use_container_width=True,hide_index=True); st.download_button("Download scored CSV",scored.to_csv(index=False).encode(),file_name="emotion_predictions.csv",mime="text/csv")
with tab_model:
    st.subheader("Architecture and artifact details")
    st.json({"model_type":pipeline.metadata.get("model_type"),"framework":pipeline.metadata.get("framework"),"classes":pipeline.classes,"vocabulary_size":pipeline.metadata.get("vocabulary_size"),"max_sequence_length":pipeline.max_sequence_length,"evaluation_metrics":pipeline.metadata.get("evaluation_metrics"),"dataset_note":pipeline.metadata.get("dataset_note")})
    st.markdown("### Responsible use")
    st.write(RESPONSIBLE_USE)
