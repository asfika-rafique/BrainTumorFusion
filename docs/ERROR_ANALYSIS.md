# Error analysis

The repository preserves historical prediction CSVs, classification reports, and confusion matrices under ignored local outputs and archived research material. They do not have complete, reproducible checkpoint/config lineage and are marked historical/unverified.

The preserved 394-row historical artifacts are internally inconsistent: the JSON report records accuracy `0.2766497462`, while the prediction CSV recomputes to `115/394 = 0.2918781726`. The filename `best_ep18_acc0.830.pt` is not evidence of an 0.830 result. No clean-pipeline error analysis has been executed because runtime evaluation remains pending.

Observed pattern and interpretation must remain separate:

- **Observed:** the preserved historical confusion matrix contains predictions concentrated in the first two output columns.
- **Not established:** the cause, clinical meaning, generalization behavior, and whether the historical checkpoint produced the matrix.
