# Data leakage

## Findings from the local audit

The original `data/raw/train` and `data/raw/test` folders contain 64 exact SHA-256 duplicate groups crossing the historical split boundary. Across the full raw tree, 166 exact duplicate groups contain 403 files. These are exact byte-identical groups according to SHA-256; patient identity cannot be inferred from them.

The clean manifest at `data/interim/leakage_free_split.csv` assigns each complete exact-hash group to only one of `train`, `validation`, or `test`. The current audit found zero duplicate hash groups crossing clean splits.

Run the reproducible audit with:

```powershell
python -m scripts.audit_data_leakage --data-root data/raw --manifest data/interim/leakage_free_split.csv --output-dir artifacts
```

It writes the ignored `artifacts/data_leakage_report.json` and `artifacts/data_leakage_report.csv`, including relative file paths for historical cross-split groups. Near-duplicate/perceptual-hash detection is not implemented.

## Protocol status

The clean trainer uses training data for optimization, validation data for checkpoint selection, and evaluates the final test split after selection. The historical trainer still evaluates its legacy test split during training and is retained only as research history.

No patient, subject, study, or acquisition metadata was supplied. Therefore, patient-level separation cannot be guaranteed even when exact file duplicates are separated.
