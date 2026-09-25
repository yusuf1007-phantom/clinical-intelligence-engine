# Milestone 2 — Transformer Tournament

## Research question
Can contextual encoders improve rhetorical sentence classification over the classical TF-IDF baseline, particularly for ambiguous labels such as OBJECTIVE and BACKGROUND, and surpass the historical 2025 F1 reference of 0.8404?

## Tournament
1. ModernBERT-base — newer general-domain encoder
2. MSR BiomedBERT — biomedical-domain encoder pretrained on PubMed/PubMedCentral

The models use the same train/dev/test data and the same evaluation family:
accuracy, macro-F1 and weighted-F1.

## Run
```bash
pip install -e ".[dev,transformers]"
python scripts/run_transformer_tournament.py
```

For a local smoke test, set `max_train_samples` and `max_eval_samples` in the YAML.
For the real experiment, keep them `null` and use a CUDA GPU.

## Automated error analysis
After both models finish, the tournament script:
- ranks completed transformer runs;
- identifies the strongest transformer;
- loads the strongest classical baseline;
- compares sentence-level predictions;
- exports cases where the transformer fixed the baseline;
- exports cases where the baseline was correct and the transformer failed.

This makes model improvement inspectable rather than reducing it to one headline metric.
