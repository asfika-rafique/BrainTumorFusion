# Research documentation

This directory contains implementation-grounded documentation. The supplied paper is used as a structural reference, while the repository code and local audit artifacts determine what is reported as implemented or verified.

| Document | Purpose |
|---|---|
| `PAPER_REPOSITORY_GAP_ANALYSIS.md` | Component-by-component comparison with the supplied paper |
| `PAPER_CLAIM_AUDIT.md` | Claim-level treatment of paper statements |
| `DATASET.md` | Observed local dataset structure and exact counts |
| `PREPROCESSING.md` | Transformations implemented in code |
| `DATA_LEAKAGE.md` | Exact-duplicate and split protocol findings |
| `ARCHITECTURE.md` | Active image model architecture |
| `TRAINING_METHOD.md` | Active training configuration and missing paper features |
| `TRANSFER_LEARNING.md` | What is and is not implemented for fine-tuning |
| `GRADCAM.md` | Grad-CAM implementation and limitations |
| `ERROR_ANALYSIS.md` | Evidence boundaries for historical result artifacts |
| `LIMITATIONS.md` | Verified limitations and unresolved metadata |
| `FUTURE_WORK.md` | Explicitly unimplemented future directions |
| `FINAL_REPOSITORY_AUDIT.md` | Summary of this finalization pass and verification boundaries |

Data-derived reports and figures are generated locally under the ignored `artifacts/` directory.
