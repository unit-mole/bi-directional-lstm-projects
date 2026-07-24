"""Generate the deterministic, balanced educational emotion dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import pandas as pd

LEXICON={
"sadness":["sad","unhappy","lonely","heartbroken","disappointed","miserable","gloomy","sorrowful","tearful","down"],
"joy":["happy","excited","delighted","cheerful","thrilled","joyful","pleased","grateful","optimistic","wonderful"],
"love":["loving","affectionate","devoted","fond","caring","adored","close","tender","warm","romantic"],
"anger":["angry","furious","irritated","annoyed","outraged","mad","frustrated","resentful","enraged","cross"],
"fear":["afraid","anxious","worried","nervous","scared","terrified","uneasy","panicked","fearful","frightened"],
"surprise":["surprised","amazed","astonished","shocked","stunned","speechless","startled","unexpectedly amazed","in disbelief","taken aback"],
}
EVENTS={
"sadness":["the disappointing news","missing my friends","the difficult goodbye","losing the opportunity","a lonely evening","the failed plan"],
"joy":["the new opportunity","my great result","meeting my friends","the wonderful celebration","finishing the project","today's good news"],
"love":["my family","my closest friend","the person I care about","our time together","a thoughtful message","the kindness I received"],
"anger":["the repeated mistake","the unfair decision","being ignored","the broken promise","the unnecessary delay","the rude response"],
"fear":["the upcoming examination","what might happen tomorrow","the uncertain situation","speaking in public","the sudden noise","the risky journey"],
"surprise":["the unexpected announcement","the sudden gift","the unbelievable result","the last-minute change","the surprise visitor","the shocking update"],
}
TEMPLATES=[
"I feel {modifier} {word} about {event}.","I am {word} because of {event}.","Honestly, {event} makes me feel {word}.","My mood is {word} after {event}.","Right now I am {modifier} {word}.","Thinking about {event} leaves me {word}.","It is hard to hide how {word} I feel about {event}.","Today I feel {word}; it is connected to {event}.","The way I feel can only be described as {word}.","I keep feeling {word} whenever I remember {event}."
]
MODIFIERS=["really","extremely","deeply","quite","genuinely","so","incredibly","very","truly","completely"]
PREFIXES=["","To be honest, ","At this moment, ","After thinking about it, ","I have to admit, "]
SUFFIXES=[""," It has stayed on my mind."," I can feel it strongly."," That is my honest reaction."," The feeling is difficult to ignore."]

def generate(rows_per_class=1200,seed=42):
    rng=random.Random(seed); rows=[]
    for label,words in LEXICON.items():
        seen=set()
        while len(seen)<rows_per_class:
            template=rng.choice(TEMPLATES); text=rng.choice(PREFIXES)+template.format(modifier=rng.choice(MODIFIERS),word=rng.choice(words),event=rng.choice(EVENTS[label]))+rng.choice(SUFFIXES)
            text=" ".join(text.split())
            if text not in seen: seen.add(text); rows.append({"text":text,"emotion":label})
    rng.shuffle(rows); return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default="data/emotion_dataset_full.csv"); parser.add_argument("--rows-per-class",type=int,default=1200); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args()
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); frame=generate(args.rows_per_class,args.seed); frame.to_csv(path,index=False); print(f"Saved {len(frame):,} rows to {path}"); print(frame["emotion"].value_counts().sort_index())
if __name__=="__main__": main()
