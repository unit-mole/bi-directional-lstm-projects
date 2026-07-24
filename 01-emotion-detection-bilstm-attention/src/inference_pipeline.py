"""Artifact loading and emotion inference with token-level attention."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from .model_training import EmotionBiLSTMAttention
from .tokenizer_utils import Vocabulary

class ArtifactError(RuntimeError): pass

@dataclass(frozen=True)
class EmotionPrediction:
    text: str
    predicted_emotion: str
    confidence: float
    probabilities: dict[str,float]
    important_tokens: list[tuple[str,float]]
    @property
    def top_probabilities(self): return sorted(self.probabilities.items(),key=lambda item:item[1],reverse=True)
    def interpretation(self):
        level="high" if self.confidence>=0.75 else "moderate" if self.confidence>=0.55 else "low"
        return f"The highest estimated class is {self.predicted_emotion.title()} with {level} confidence. Review the probability distribution and context before drawing conclusions."

class EmotionInferencePipeline:
    def __init__(self, model_dir: str | Path):
        self.model_dir=Path(model_dir); self.model=None; self.vocabulary=None; self.metadata={}; self.classes=[]; self.max_sequence_length=40; self.artifact_status="not_loaded"; self.supports_attention=True
    def load(self):
        try:
            self.metadata=json.loads((self.model_dir/"model_metadata.json").read_text(encoding="utf-8"))
            self.vocabulary=Vocabulary.load(self.model_dir/"vocabulary.json")
            mapping=json.loads((self.model_dir/"label_mapping.json").read_text(encoding="utf-8")); self.classes=[mapping[str(i)] for i in range(len(mapping))]
            checkpoint=torch.load(self.model_dir/"emotion_bilstm_attention.pt",map_location="cpu",weights_only=True)
            self.model=EmotionBiLSTMAttention(**checkpoint["model_config"]); self.model.load_state_dict(checkpoint["state_dict"]); self.model.eval()
            self.max_sequence_length=int(self.metadata["max_sequence_length"]); self.artifact_status=str(self.metadata.get("artifact_status","trained_attention_model")); return self
        except Exception as exc:
            raise ArtifactError(f"Could not load trained Project 01 artifacts from {self.model_dir}: {exc}") from exc
    def predict(self,text:str):
        if self.model is None or self.vocabulary is None: raise ArtifactError("Pipeline is not loaded.")
        if not str(text).strip(): raise ValueError("Input text cannot be empty.")
        ids,tokens=self.vocabulary.encode(text,self.max_sequence_length); tensor=torch.tensor([ids],dtype=torch.long)
        with torch.no_grad(): logits,attention=self.model(tensor); probs=torch.softmax(logits,dim=1)[0].numpy(); weights=attention[0,:len(tokens)].numpy()
        predicted=int(np.argmax(probs)); ranked=sorted(zip(tokens,weights.tolist()),key=lambda item:item[1],reverse=True)[:min(10,len(tokens))]
        return EmotionPrediction(str(text),self.classes[predicted],float(probs[predicted]),{label:float(probs[i]) for i,label in enumerate(self.classes)},[(token,float(score)) for token,score in ranked])
    def predict_many(self,texts): return [self.predict(str(text)) for text in texts]
