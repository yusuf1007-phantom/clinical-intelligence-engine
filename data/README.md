# Data

Raw and processed datasets are intentionally excluded from Git.

Expected raw layout:

data/raw/PubMed_200k_RCT/
- train.txt
- dev.txt
- test.txt

The preparation command validates labels, sentence structure, duplicate sentence keys,
and abstract-level leakage before writing normalized CSV splits.
