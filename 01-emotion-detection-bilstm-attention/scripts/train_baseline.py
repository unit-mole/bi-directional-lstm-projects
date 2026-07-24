from pathlib import Path
import sys, joblib
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_preprocessing import load_and_clean_dataset, split_dataframe
from src.baseline_model import build_baseline
frame,_=load_and_clean_dataset(ROOT/"data/emotion_dataset_full.csv"); train,_,test=split_dataframe(frame)
model=build_baseline(); model.fit(train["text_clean"],train["emotion"]); print("Baseline accuracy:",model.score(test["text_clean"],test["emotion"])); joblib.dump(model,ROOT/"models/tfidf_logistic_baseline.joblib")
