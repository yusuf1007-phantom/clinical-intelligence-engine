import argparse, json, os, sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_intelligence.evaluation.metrics import evaluate_predictions
from clinical_intelligence.models.baselines import build_model
from clinical_intelligence.utils.config import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    processed = Path(cfg["data"]["processed_dir"])
    train = pd.read_csv(processed / "train.csv", keep_default_na=False)
    dev = pd.read_csv(processed / "dev.csv", keep_default_na=False)
    test = pd.read_csv(processed / "test.csv", keep_default_na=False)

    feat = cfg["features"]
    vectorizer = TfidfVectorizer(
        max_features=feat["max_features"],
        ngram_range=(feat["ngram_min"], feat["ngram_max"]),
        min_df=feat["min_df"],
        sublinear_tf=feat["sublinear_tf"],
    )
    X_train = vectorizer.fit_transform(train["text"])
    X_dev = vectorizer.transform(dev["text"])
    X_test = vectorizer.transform(test["text"])

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["experiment"]["name"])

    out = Path(cfg["output"]["artifact_dir"])
    out.mkdir(parents=True, exist_ok=True)
    summary = []

    for name, params in cfg["models"].items():
        model = build_model(name, params, cfg["training"]["seed"])
        with mlflow.start_run(run_name=name):
            mlflow.log_params({f"model.{k}": v for k, v in params.items()})
            mlflow.log_params({
                "tfidf.max_features": feat["max_features"],
                "tfidf.ngram": f'{feat["ngram_min"]}-{feat["ngram_max"]}',
                "seed": cfg["training"]["seed"],
            })

            model.fit(X_train, train["label"])
            dev_metrics = evaluate_predictions(dev["label"], model.predict(X_dev))
            test_metrics = evaluate_predictions(test["label"], model.predict(X_test))

            mlflow.log_metrics({
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
            })

            model_dir = out / name
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / "model.joblib")
            if name == list(cfg["models"].keys())[0]:
                joblib.dump(vectorizer, out / "tfidf_vectorizer.joblib")

            report_path = model_dir / "test_metrics.json"
            report_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path))
            mlflow.sklearn.log_model(model, name="model")

            summary.append({
                "model": name,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "historical_f1_gap": test_metrics["macro_f1"] - cfg["experiment"]["historical_macro_f1"],
            })

    summary = sorted(summary, key=lambda x: x["test_macro_f1"], reverse=True)
    (out / "leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBASELINE LEADERBOARD")
    print("-" * 72)
    for row in summary:
        print(f'{row["model"]:22s} accuracy={row["test_accuracy"]:.4f} '
              f'macro_f1={row["test_macro_f1"]:.4f} '
              f'gap_vs_2025={row["historical_f1_gap"]:+.4f}')
    print("-" * 72)
    print("Historical 2025 paper F1 reference: "
          f'{cfg["experiment"]["historical_macro_f1"]:.4f}')

if __name__ == "__main__":
    main()
