# Validation reports

This directory contains evidence-based audit artifacts. The reports preserve inconvenient findings and do not replace historical outputs.

- `research_validation_audit.md` records exact cross-split duplicate groups, the absence of patient-level metadata, historical test-set selection, result conflicts, and release limitations.
- `checkpoint_manifest.csv` records checkpoint file hashes and lightweight serialization inspection without trusting filename metrics.

The clean pipeline is configured in `configs/clean_resnet18_image_only.yaml`; its generated split manifest and model outputs remain local and are excluded from Git. Clean-pipeline performance is still pending runtime execution.

Run the audit again after obtaining a compatible clean environment:

```powershell
python -m scripts.data.research_validation_audit --project-root .
```
