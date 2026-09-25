import argparse, subprocess, sys

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transformer_tournament.yaml")
    args = ap.parse_args()

    for model in ("modernbert", "biomedbert"):
        run([sys.executable, "scripts/train_transformer.py",
             "--config", args.config, "--model", model])

    run([sys.executable, "scripts/build_tournament_report.py",
         "--config", args.config])

if __name__ == "__main__":
    main()
