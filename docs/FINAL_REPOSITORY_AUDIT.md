# Final repository audit

This record summarizes the implementation-grounded finalization pass performed against the supplied paper and the local repository.

## What already existed

The repository already contained an image-only ResNet/FusionNet implementation, leakage-aware exact-hash splitting, evaluation/inference entry points, Grad-CAM, research UI prototypes, historical outputs, and release documentation.

## What was implemented in this pass

- Dataset audit with exact file properties, hashes, duplicate groups, EXIF counts, and clean-manifest counts.
- Exact-duplicate leakage audit with historical cross-split group reporting.
- Local dataset and architecture figure generation using observed data/code only.
- Architecture and experiment status artifacts.
- Training-only class-weight derivation for the clean configuration.
- Expanded metric output with macro/weighted summaries and normalized confusion matrices.
- Clean checkpoint provenance fields and an explicit final-test evaluator.
- Paper/repository gap analysis and claim audit documentation.

## What was verified

- 3,264 readable JPEG images.
- 166 exact duplicate groups, including 64 crossing the historical train/test boundary.
- Zero exact-hash groups crossing the clean train/validation/test manifest.
- 17 images containing EXIF metadata.
- 7 tests passed and 3 ML-dependent tests skipped in the current environment.
- Python compilation, data audit, leakage audit, metrics smoke test, and Markdown link validation passed.

## What was not verified

The current machine has Python 3.14, no Conda, and no PyTorch/torchvision installation. Therefore, model construction, forward passes, checkpoint loading, training, clean evaluation, baselines, ablations, ROC/PR curves, training curves, parameter totals, and runtime Grad-CAM generation were not executed. Their result files are marked `NOT_RUN`, `NOT_MEASURED`, or `PENDING` where applicable.

## Scientific limitations

Patient-level separation cannot be guaranteed because patient/subject/study/acquisition identifiers are unavailable. Dataset provenance, licensing, consent, and de-identification status remain to be confirmed. The active implementation is image-only; text fusion is not implemented. Historical metrics and checkpoint filenames are internally inconsistent and remain historical/unverified.
