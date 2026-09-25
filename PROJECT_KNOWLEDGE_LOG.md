# Project Knowledge Log

## Verified results

| Model | Scope | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| TF-IDF linear baseline | verified baseline run | 0.8005 | 0.7343 | 0.7964 |
| ModernBERT | 10k smoke test | 0.8400 | 0.7767 | 0.8408 |
| BiomedBERT | 10k smoke test | 0.8665 | 0.8095 | 0.8669 |
| Historical Hybrid BiLSTM | thesis reference | 0.8430 | 0.8404 | — |

The transformer results are deliberately small smoke tests and are not presented as a strict reproduction of the historical thesis experiment.

## Engineering problems solved
1. Preserved literal `nan` text when reading CSV.
2. Removed raw string `label` and `text` columns before Hugging Face batching.
3. Reinstalled the editable package after Colab runtime state changed.

## Modernization
Reproducible configuration, transformer tournament, MLflow-compatible tracking, biomedical NER, FastAPI, Streamlit, Docker, tests, and GitHub Actions CI.
