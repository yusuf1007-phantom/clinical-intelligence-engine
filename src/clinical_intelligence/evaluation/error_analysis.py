import pandas as pd

def disagreement_table(texts, y_true, baseline_pred, transformer_pred):
    df = pd.DataFrame({
        "text": list(texts),
        "true_label": list(y_true),
        "baseline_prediction": list(baseline_pred),
        "transformer_prediction": list(transformer_pred),
    })
    df["baseline_correct"] = df["true_label"] == df["baseline_prediction"]
    df["transformer_correct"] = df["true_label"] == df["transformer_prediction"]
    df["outcome"] = "same_result"
    df.loc[(~df["baseline_correct"]) & df["transformer_correct"], "outcome"] = "transformer_fixed"
    df.loc[df["baseline_correct"] & (~df["transformer_correct"]), "outcome"] = "baseline_was_better"
    return df
