import argparse, json, os, random, sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clinical_intelligence.models.transformer import load_sequence_classifier
from clinical_intelligence.utils.config import load_config

LABELS = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dataset(df, tokenizer, label2id, max_length):
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    ds = ds.map(lambda x: {"labels": label2id[x["label"]]})
    ds = ds.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=max_length
        ),
        batched=True,
    )
    return ds

def metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "macro_f1": f1_score(labels, pred, average="macro"),
        "weighted_f1": f1_score(labels, pred, average="weighted"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transformer_tournament.yaml")
    ap.add_argument("--model", required=True, choices=["modernbert", "biomedbert"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["training"]["seed"])

    d = cfg["data"]
    base = Path(d["processed_dir"])
    train = pd.read_csv(base / d["train_split"], keep_default_na=False)
    dev = pd.read_csv(base / d["dev_split"], keep_default_na=False)
    test = pd.read_csv(base / d["test_split"], keep_default_na=False)

    if d.get("max_train_samples"):
        train = train.sample(min(len(train), d["max_train_samples"]), random_state=cfg["training"]["seed"])
    if d.get("max_eval_samples"):
        n = d["max_eval_samples"]
        dev, test = dev.head(n), test.head(n)

    checkpoint = cfg["models"][args.model]["checkpoint"]
    tokenizer, model, label2id, id2label = load_sequence_classifier(checkpoint, LABELS)
    train_ds = make_dataset(train, tokenizer, label2id, cfg["training"]["max_length"])
    dev_ds = make_dataset(dev, tokenizer, label2id, cfg["training"]["max_length"])
    test_ds = make_dataset(test, tokenizer, label2id, cfg["training"]["max_length"])

    out = Path(cfg["output"]["root_dir"]) / args.model
    out.mkdir(parents=True, exist_ok=True)

    use_fp16 = bool(cfg["training"]["fp16"] and torch.cuda.is_available())
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        greater_is_better=True,
        fp16=use_fp16,
        report_to=[],
        seed=cfg["training"]["seed"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg["training"]["early_stopping_patience"]
        )],
    )

    tracking = os.getenv("MLFLOW_TRACKING_URI")
    if tracking:
        mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(cfg["experiment"]["name"])

    with mlflow.start_run(run_name=args.model):
        mlflow.log_params({
            "checkpoint": checkpoint,
            "epochs": cfg["training"]["epochs"],
            "learning_rate": cfg["training"]["learning_rate"],
            "max_length": cfg["training"]["max_length"],
            "train_rows": len(train),
        })
        trainer.train()
        result = trainer.predict(test_ds)
        scores = metrics((result.predictions, result.label_ids))
        scores["historical_f1_gap"] = scores["macro_f1"] - cfg["experiment"]["historical_f1"]
        mlflow.log_metrics(scores)

        pred_ids = np.argmax(result.predictions, axis=-1)
        predictions = test.copy()
        predictions["prediction"] = [id2label[int(i)] for i in pred_ids]
        predictions["correct"] = predictions["label"] == predictions["prediction"]
        predictions.to_csv(out / "test_predictions.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        trainer.save_model(out / "best_model")
        tokenizer.save_pretrained(out / "best_model")
        mlflow.log_artifacts(str(out))

    print(json.dumps({"model": args.model, "checkpoint": checkpoint, **scores}, indent=2))

if __name__ == "__main__":
    main()
