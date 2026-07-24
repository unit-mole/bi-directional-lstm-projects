"""CLI for training the Project 01 BiLSTM-attention model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from src.config import TrainingConfig
from src.model_training import train_pipeline

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data",default="data/emotion_dataset_full.csv"); parser.add_argument("--epochs",type=int,default=12); parser.add_argument("--batch-size",type=int,default=64); args=parser.parse_args()
    config=TrainingConfig(epochs=args.epochs,batch_size=args.batch_size); metadata=train_pipeline(PROJECT_ROOT/args.data,config=config); print(json.dumps(metadata["evaluation_metrics"],indent=2))
if __name__=="__main__": main()
