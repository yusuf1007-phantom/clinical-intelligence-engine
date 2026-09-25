# Baseline Experiment Report — Milestone 1

## Dataset verification
The uploaded official `pubmed-rct` repository was inspected and the full 200k numbers-replaced dataset was parsed and validated.

Full 200k split statistics:
- Train: 2,211,861 sentences / 190,654 abstracts
- Dev: 28,932 sentences / 2,500 abstracts
- Test: 29,493 sentences / 2,500 abstracts

The validation pipeline checked expected labels, empty text, duplicate sentence keys, and abstract-level split leakage.

## Fast reproducible smoke benchmark
To verify the complete train -> predict -> evaluate path within the available execution window, a TF-IDF + SGD linear SVM baseline was run on the official PubMed 20k RCT variant.

Results:
- Accuracy: 0.8005
- Macro F1: 0.7343
- Weighted F1: 0.7964

Per-class F1:
- BACKGROUND: 0.63
- CONCLUSIONS: 0.71
- METHODS: 0.88
- OBJECTIVE: 0.59
- RESULTS: 0.87

This is a smoke baseline, not a replacement for the paper's historical 200k result. The lower OBJECTIVE/BACKGROUND performance is consistent with the ambiguity discussed in the master's paper.

## Full 200k training status
The full 200k data pipeline completed successfully. A 30,000-feature TF-IDF matrix was constructed for all 2,211,861 training sentences, but full LinearSVC training exceeded the current execution window. The repository therefore keeps full-scale training as a local/cloud run rather than reporting an incomplete metric.
