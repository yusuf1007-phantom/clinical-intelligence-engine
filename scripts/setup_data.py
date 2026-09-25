import argparse
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Stage PubMed RCT data from the official repository checkout/archive.")
    ap.add_argument("--source", required=True, help="Extracted pubmed-rct repository root")
    ap.add_argument("--project-data", default="data/raw")
    ap.add_argument("--variant", choices=["20k", "200k-at"], default="20k")
    args = ap.parse_args()

    src = Path(args.source)
    dst_root = Path(args.project_data)
    if args.variant == "20k":
        src_dir = src / "PubMed_20k_RCT"
        dst = dst_root / "PubMed_20k_RCT"
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("train.txt", "dev.txt", "test.txt"):
            shutil.copy2(src_dir / name, dst / name)
    else:
        src_dir = src / "PubMed_200k_RCT_numbers_replaced_with_at_sign"
        dst = dst_root / "PubMed_200k_RCT_numbers_replaced_with_at_sign"
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("dev.txt", "test.txt"):
            shutil.copy2(src_dir / name, dst / name)
        with zipfile.ZipFile(src_dir / "train.zip") as z:
            z.extract("train.txt", dst)
    print(f"Staged {args.variant} dataset at {dst}")

if __name__ == "__main__":
    main()
