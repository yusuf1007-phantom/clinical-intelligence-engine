import pandas as pd
from clinical_intelligence.data.validation import validate_split, validate_no_abstract_leakage

def frame(aid):
    return pd.DataFrame([{
        "abstract_id": aid, "sentence_id": 0,
        "text": "Example sentence.", "label": "BACKGROUND"
    }])

def test_valid_split():
    stats = validate_split(frame("a"), "train")
    assert stats["rows"] == 1

def test_no_leakage():
    validate_no_abstract_leakage(frame("a"), frame("b"), frame("c"))
