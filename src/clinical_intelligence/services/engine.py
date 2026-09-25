from functools import lru_cache
import os
import torch
from transformers import pipeline

NER_MODEL = os.getenv("NER_MODEL", "d4data/biomedical-ner-all")

@lru_cache(maxsize=1)
def ner_pipeline():
    return pipeline(
        "token-classification",
        model=NER_MODEL,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

@lru_cache(maxsize=1)
def structure_pipeline():
    path = os.getenv("STRUCTURE_MODEL_PATH")
    if not path:
        return None
    return pipeline(
        "text-classification",
        model=path,
        tokenizer=path,
        device=0 if torch.cuda.is_available() else -1,
    )

def extract_entities(text: str):
    return [
        {
            "text": x.get("word", ""),
            "label": x.get("entity_group", x.get("entity", "")),
            "score": round(float(x.get("score", 0.0)), 4),
            "start": x.get("start"),
            "end": x.get("end"),
        }
        for x in ner_pipeline()(text)
    ]

def classify_structure(text: str):
    p = structure_pipeline()
    if p is None:
        return None
    x = p(text, truncation=True)[0]
    return {"label": x["label"], "score": round(float(x["score"]), 4)}

def analyze(text: str):
    return {
        "text": text,
        "structure": classify_structure(text),
        "entities": extract_entities(text),
    }
