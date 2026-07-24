import pandas as pd
from src.data_preprocessing import load_and_clean_dataset
def test_dataset_loading(tmp_path):
    path=tmp_path/"data.csv"; pd.DataFrame({"text":["Happy","Happy","Sad"],"emotion":["joy","joy","sadness"]}).to_csv(path,index=False)
    frame,audit=load_and_clean_dataset(path); assert len(frame)==2; assert audit.duplicate_rows_removed==1
