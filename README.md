# Clinical Intelligence Engine

A biomedical NLP project modernized from a master's research prototype into a reproducible, API-driven and containerized AI application.

## What it demonstrates
- classical ML baseline and transformer fine-tuning
- ModernBERT vs BiomedBERT automated experiments
- biomedical named-entity extraction
- optional saved fine-tuned scientific sentence classifier
- FastAPI inference service
- Streamlit interactive application
- MLflow-compatible experiment tracking
- Docker packaging
- pytest + GitHub Actions CI

> Research/portfolio demonstration only. Not for clinical diagnosis or treatment decisions.

## Verified experiment snapshot

| Model | Scope | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| TF-IDF baseline | verified baseline run | 0.8005 | 0.7343 | 0.7964 |
| ModernBERT | 10k smoke test | 0.8400 | 0.7767 | 0.8408 |
| **BiomedBERT** | 10k smoke test | **0.8665** | **0.8095** | **0.8669** |
| Historical Hybrid BiLSTM | thesis reference | 0.8430 | 0.8404 | — |

The transformer runs intentionally use a 10k smoke-test subset so the portfolio pipeline can be verified quickly. The historical result is a reference, not an identical experimental reproduction.

## Architecture

```text
Biomedical text
      |
      +----> Fine-tuned structure classifier (optional saved artifact)
      |
      +----> Biomedical NER
      |
      v
 Structured result
      |
   +--+--+
   |     |
FastAPI  Streamlit
   |     |
   +--+--+
      |
    Docker
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[app]"
streamlit run app.py
```

The first NER request downloads the pretrained model weights from Hugging Face.

## API

```bash
uvicorn clinical_intelligence.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

Endpoints: `GET /health`, `POST /entities`, `POST /classify`, `POST /analyze`.

`/classify` is enabled only when `STRUCTURE_MODEL_PATH` points to a saved fine-tuned classifier. The project never presents an unfine-tuned base checkpoint as a trained classifier.

## Docker

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## Experiments

```bash
pip install -e ".[dev,transformers]"
python scripts/train_transformer.py --config configs/transformer_tournament.yaml --model biomedbert
```

## Repository map
- `src/clinical_intelligence/data` — parsing and validation
- `src/clinical_intelligence/models` — classical and transformer models
- `src/clinical_intelligence/evaluation` — metrics and error analysis
- `src/clinical_intelligence/services` — inference layer
- `src/clinical_intelligence/api` — FastAPI application
- `configs` — reproducible experiment configuration
- `scripts` — training automation
- `tests` — automated tests
- `PROJECT_KNOWLEDGE_LOG.md` — results, bugs, decisions and lessons

## Data and model files
Datasets, MLflow runs and trained model artifacts are intentionally excluded from Git. The source dataset can be obtained from the original PubMed RCT repository. Biomedical NER model weights are downloaded at runtime.

## Future extensions
Entity normalization, relation extraction, hybrid retrieval and grounded biomedical RAG are future extensions; they are not represented here as already completed functionality.
