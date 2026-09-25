import argparse, json
from pathlib import Path
from .parser import parse_pubmed_rct
from .validation import validate_split, validate_no_abstract_leakage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}
    dfs, stats = {}, {}
    for split, filename in mapping.items():
        df = parse_pubmed_rct(raw / filename)
        stats[split] = validate_split(df, split)
        dfs[split] = df

    validate_no_abstract_leakage(dfs["train"], dfs["dev"], dfs["test"])

    for split, df in dfs.items():
        df.to_csv(out / f"{split}.csv", index=False)

    (out / "dataset_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nValidated processed data written to: {out}")

if __name__ == "__main__":
    main()
