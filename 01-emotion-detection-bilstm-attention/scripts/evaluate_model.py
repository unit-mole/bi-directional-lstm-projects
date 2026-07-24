"""Run smoke predictions against the saved model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.inference_pipeline import EmotionInferencePipeline
samples=["I am extremely happy and excited today.","I feel worried and anxious about the upcoming examination.","I am furious about the unfair decision.","I adore my family and feel so close to them.","The unexpected announcement left me stunned.","I feel lonely and heartbroken tonight."]
pipeline=EmotionInferencePipeline(ROOT/"models").load()
for text in samples:
    result=pipeline.predict(text); print(f"{result.predicted_emotion:>8} {result.confidence:.1%} | {text}")
