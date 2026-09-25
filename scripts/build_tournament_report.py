import argparse, json, sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clinical_intelligence.evaluation.error_analysis import disagreement_table
from clinical_intelligence.utils.config import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transformer_tournament.yaml")
    ap.add_argument("--baseline-dir", default="artifacts/baselines")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output"]["root_dir"])

    rows = []
    for name in cfg["models"]:
        p = root / name / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            rows.append({"model": name, **m})
    if not rows:
        raise SystemExit("No completed transformer runs found.")

    leaderboard = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    leaderboard.to_csv(root / "leaderboard.csv", index=False)

    # Error comparison uses the strongest completed transformer.
    winner = leaderboard.iloc[0]["model"]
    tpred = pd.read_csv(root / winner / "test_predictions.csv", keep_default_na=False)
    bdir = Path(args.baseline_dir)
    vec = joblib.load(bdir / "tfidf_vectorizer.joblib")

    # Find strongest baseline from existing leaderboard if available.
    blb = json.loads((bdir / "leaderboard.json").read_text())
    best_baseline = blb[0]["model"]
    bmodel = joblib.load(bdir / best_baseline / "model.joblib")
    bpred = bmodel.predict(vec.transform(tpred["text"]))

    errors = disagreement_table(
        tpred["text"], tpred["label"], bpred, tpred["prediction"]
    )
    errors.to_csv(root / "baseline_vs_transformer_disagreements.csv", index=False)

    print("\nTRANSFORMER TOURNAMENT")
    print(leaderboard.to_string(index=False))
    print(f"\nWinner: {winner}")
    print("\nDisagreement analysis:")
    print(errors["outcome"].value_counts().to_string())

if __name__ == "__main__":
    main()
