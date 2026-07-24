"""Validate Project 01 structure and model artifacts."""
from pathlib import Path
import json, sys, torch
ROOT=Path(__file__).resolve().parents[1]
required=["app/streamlit_app.py","app/requirements.txt","data/emotion_dataset_full.csv","models/emotion_bilstm_attention.pt","models/vocabulary.json","models/label_mapping.json","models/model_metadata.json","src/model_training.py","src/inference_pipeline.py"]
missing=[p for p in required if not (ROOT/p).is_file()]
if missing: raise SystemExit("Missing required files: "+", ".join(missing))
checkpoint=torch.load(ROOT/"models/emotion_bilstm_attention.pt",map_location="cpu",weights_only=True)
assert "state_dict" in checkpoint and "model_config" in checkpoint
metadata=json.loads((ROOT/"models/model_metadata.json").read_text(encoding="utf-8")); assert metadata["artifact_status"]=="bundled_trained_attention_model"
print("Project 01 validation passed.")
