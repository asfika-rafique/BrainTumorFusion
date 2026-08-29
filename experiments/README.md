# Experiments

The active release-candidate experiment is configured in `configs/clean_resnet18_image_only.yaml`. It uses the exact-duplicate-aware manifest, seed 42, validation-based checkpoint selection, and one final test evaluation.

Historical experiments and their artifacts are retained under `_archive/` or ignored local output directories. They are not interchangeable with clean-pipeline experiments.

For each executed experiment, retain the dataset version/checksum, split manifest, configuration, seed, environment, checkpoint, and metrics. No clean-pipeline performance result has been established in the repository at present.
