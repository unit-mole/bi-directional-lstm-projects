from pathlib import Path
from src.inference_pipeline import EmotionInferencePipeline
ROOT=Path(__file__).resolve().parents[1]
def test_smoke_predictions():
    pipe=EmotionInferencePipeline(ROOT/"models").load()
    joy=pipe.predict("I am extremely happy and excited today.")
    fear=pipe.predict("I feel worried and anxious about the upcoming examination.")
    assert joy.predicted_emotion=="joy"; assert fear.predicted_emotion=="fear"; assert joy.important_tokens
