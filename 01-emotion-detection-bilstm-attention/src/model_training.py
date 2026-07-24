"""BiLSTM + temporal attention model and end-to-end training pipeline."""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .attention_layer import TemporalAttention
from .config import MODEL_DIR, OUTPUT_DIR, TrainingConfig
from .data_preprocessing import load_and_clean_dataset, split_dataframe, validate_class_support
from .tokenizer_utils import Vocabulary, build_vocabulary

class EmotionDataset(Dataset):
    def __init__(self, frame, vocabulary, label_to_id, max_length):
        self.frame = frame.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.label_to_id = label_to_id
        self.max_length = max_length
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        ids, _ = self.vocabulary.encode(row["text_clean"], self.max_length)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.label_to_id[row["emotion"]], dtype=torch.long)

class EmotionBiLSTMAttention(nn.Module):
    def __init__(self, vocabulary_size, number_of_classes, embedding_dimension=96, lstm_units=64, dense_units=96, dropout_rate=0.30, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(embedding_dimension, lstm_units, batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(lstm_units * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_units * 2, dense_units), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(dense_units, number_of_classes)
        )
    def forward(self, token_ids):
        mask = token_ids.ne(self.pad_id)
        embedded = self.embedding_dropout(self.embedding(token_ids))
        sequence, _ = self.bilstm(embedded)
        context, attention = self.attention(sequence, mask)
        return self.classifier(context), attention


def set_reproducibility(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _evaluate(model, loader, device):
    model.eval(); labels=[]; predictions=[]; probabilities=[]
    with torch.no_grad():
        for token_ids, target in loader:
            logits, _ = model(token_ids.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels.extend(target.numpy().tolist())
            predictions.extend(probs.argmax(axis=1).tolist())
            probabilities.extend(probs.tolist())
    return np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)


def train_pipeline(data_path: str | Path, config: TrainingConfig | None=None, model_dir: str | Path=MODEL_DIR, output_dir: str | Path=OUTPUT_DIR) -> dict[str, Any]:
    config = config or TrainingConfig(); set_reproducibility(config.random_seed)
    model_dir=Path(model_dir); output_dir=Path(output_dir); model_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    frame, audit = load_and_clean_dataset(data_path); validate_class_support(frame, config.minimum_samples_per_class)
    train_df, validation_df, test_df = split_dataframe(frame, config.validation_size, config.test_size, config.random_seed)
    encoder=LabelEncoder(); encoder.fit(train_df["emotion"])
    classes=[str(x) for x in encoder.classes_]; label_to_id={label:i for i,label in enumerate(classes)}
    vocabulary=build_vocabulary(train_df["text_clean"], config.max_vocab_size)
    train_loader=DataLoader(EmotionDataset(train_df,vocabulary,label_to_id,config.max_sequence_length), batch_size=config.batch_size, shuffle=True)
    validation_loader=DataLoader(EmotionDataset(validation_df,vocabulary,label_to_id,config.max_sequence_length), batch_size=config.batch_size)
    test_loader=DataLoader(EmotionDataset(test_df,vocabulary,label_to_id,config.max_sequence_length), batch_size=config.batch_size)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=EmotionBiLSTMAttention(len(vocabulary.token_to_id),len(classes),config.embedding_dimension,config.lstm_units,config.dense_units,config.dropout_rate,vocabulary.pad_id).to(device)
    counts=train_df["emotion"].value_counts(); weights=torch.tensor([len(train_df)/(len(classes)*counts[label]) for label in classes],dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights); optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=1e-4)
    best_loss=float("inf"); best_state=None; history=[]; patience=3; remaining=patience
    for epoch in range(1,config.epochs+1):
        model.train(); total_loss=0.0; correct=0; seen=0
        for token_ids,target in train_loader:
            token_ids=token_ids.to(device); target=target.to(device); optimizer.zero_grad()
            logits,_=model(token_ids); loss=criterion(logits,target); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total_loss += float(loss.item())*len(target); correct += int((logits.argmax(1)==target).sum().item()); seen += len(target)
        y_val,p_val,_=_evaluate(model,validation_loader,device)
        val_loss=0.0
        model.eval()
        with torch.no_grad():
            for token_ids,target in validation_loader:
                logits,_=model(token_ids.to(device)); val_loss += float(criterion(logits,target.to(device)).item())*len(target)
        val_loss/=len(validation_df)
        row={"epoch":epoch,"train_loss":total_loss/seen,"train_accuracy":correct/seen,"validation_loss":val_loss,"validation_accuracy":accuracy_score(y_val,p_val),"validation_macro_f1":f1_score(y_val,p_val,average="macro")}
        history.append(row); print(row)
        if val_loss < best_loss-1e-4:
            best_loss=val_loss; remaining=patience; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            remaining-=1
            if remaining<=0: break
    if best_state is not None: model.load_state_dict(best_state)
    model=model.to(device); y_test,p_test,probs=_evaluate(model,test_loader,device)
    report=classification_report(y_test,p_test,target_names=classes,output_dict=True,zero_division=0)
    metrics={"accuracy":float(accuracy_score(y_test,p_test)),"macro_f1":float(f1_score(y_test,p_test,average="macro")),"weighted_f1":float(f1_score(y_test,p_test,average="weighted")),"test_rows":int(len(test_df))}
    torch.save({"state_dict":model.cpu().state_dict(),"model_config":{"vocabulary_size":len(vocabulary.token_to_id),"number_of_classes":len(classes),"embedding_dimension":config.embedding_dimension,"lstm_units":config.lstm_units,"dense_units":config.dense_units,"dropout_rate":config.dropout_rate,"pad_id":vocabulary.pad_id}}, model_dir/"emotion_bilstm_attention.pt")
    vocabulary.save(model_dir/"vocabulary.json")
    (model_dir/"label_mapping.json").write_text(json.dumps({str(i):label for i,label in enumerate(classes)},indent=2),encoding="utf-8")
    pd.DataFrame(history).to_csv(output_dir/"training_history.csv",index=False)
    pd.DataFrame(report).transpose().to_csv(output_dir/"classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test,p_test),index=classes,columns=classes).to_csv(output_dir/"confusion_matrix.csv")
    pred_frame=test_df[["text","emotion"]].copy(); pred_frame["predicted_emotion"]=[classes[i] for i in p_test]; pred_frame["confidence"]=probs.max(axis=1); pred_frame.to_csv(output_dir/"test_predictions.csv",index=False)
    metadata={"project":"01-emotion-detection-bilstm-attention","framework":"PyTorch","model_type":"Bidirectional LSTM with Temporal Attention","artifact_status":"bundled_trained_attention_model","model_path":"emotion_bilstm_attention.pt","vocabulary_path":"vocabulary.json","label_mapping_path":"label_mapping.json","classes":classes,"number_of_classes":len(classes),"vocabulary_size":len(vocabulary.token_to_id),"max_sequence_length":config.max_sequence_length,"training_config":config.to_dict(),"dataset_audit":audit.to_dict(),"split_rows":{"train":len(train_df),"validation":len(validation_df),"test":len(test_df)},"evaluation_metrics":metrics,"dataset_note":"Bundled checkpoint trained on a deterministic template-augmented educational dataset. Replace with a licensed real-world corpus for research conclusions.","responsible_use":"Educational portfolio demonstration only; not for mental-health diagnosis or high-stakes decisions."}
    (model_dir/"model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata
