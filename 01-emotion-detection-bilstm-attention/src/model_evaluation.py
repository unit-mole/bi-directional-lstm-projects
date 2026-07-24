"""Small evaluation helpers used by scripts and notebooks."""
from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score

def summary_metrics(y_true,y_pred):
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"macro_f1":float(f1_score(y_true,y_pred,average="macro")),"weighted_f1":float(f1_score(y_true,y_pred,average="weighted"))}
