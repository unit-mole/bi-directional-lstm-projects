"""Reusable Plotly visualizations."""
from __future__ import annotations
import pandas as pd
import plotly.express as px

def probability_figure(probabilities):
    frame=pd.DataFrame(sorted(probabilities.items(),key=lambda x:x[1],reverse=True),columns=["Emotion","Probability"]); frame["Emotion"]=frame["Emotion"].str.title()
    fig=px.bar(frame,x="Probability",y="Emotion",orientation="h",text=frame["Probability"].map(lambda x:f"{x:.1%}"),range_x=[0,1]); fig.update_layout(yaxis={"categoryorder":"total ascending"},height=360); return fig
