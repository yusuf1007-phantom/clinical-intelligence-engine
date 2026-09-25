from pathlib import Path
import pandas as pd

VALID_LABELS = {"BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"}

def parse_pubmed_rct(path: str | Path) -> pd.DataFrame:
    """Parse a PubMed RCT split into one row per labeled sentence."""
    rows = []
    abstract_id = None
    sentence_id = 0

    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("###"):
                abstract_id = line[3:].strip()
                sentence_id = 0
                continue

            if "\t" not in line:
                raise ValueError(f"Malformed labeled sentence: {line[:100]}")

            label, text = line.split("\t", 1)
            label, text = label.strip(), text.strip()

            if label not in VALID_LABELS:
                raise ValueError(f"Unexpected label: {label}")
            if not text:
                raise ValueError("Empty sentence encountered")
            if abstract_id is None:
                raise ValueError("Sentence encountered before abstract identifier")

            rows.append({
                "abstract_id": abstract_id,
                "sentence_id": sentence_id,
                "text": text,
                "label": label,
            })
            sentence_id += 1

    if not rows:
        raise ValueError(f"No labeled sentences found in {path}")
    return pd.DataFrame(rows)
