import pandas as pd
from .parser import VALID_LABELS

REQUIRED_COLUMNS = {"abstract_id", "sentence_id", "text", "label"}

def validate_split(df: pd.DataFrame, split_name: str) -> dict:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{split_name}: missing columns {sorted(missing)}")
    if df["text"].isna().any() or (df["text"].str.strip() == "").any():
        raise ValueError(f"{split_name}: empty text found")
    bad_labels = set(df["label"].unique()) - VALID_LABELS
    if bad_labels:
        raise ValueError(f"{split_name}: invalid labels {sorted(bad_labels)}")

    duplicate_rows = int(df.duplicated(["abstract_id", "sentence_id"]).sum())
    if duplicate_rows:
        raise ValueError(f"{split_name}: duplicate sentence keys: {duplicate_rows}")

    return {
        "split": split_name,
        "rows": int(len(df)),
        "abstracts": int(df["abstract_id"].nunique()),
        "labels": df["label"].value_counts().to_dict(),
    }

def validate_no_abstract_leakage(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> None:
    sets = {
        "train": set(train["abstract_id"]),
        "dev": set(dev["abstract_id"]),
        "test": set(test["abstract_id"]),
    }
    pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for a, b in pairs:
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"Abstract leakage between {a} and {b}: {len(overlap)} IDs")
