# BrainTumorFusion

Research software for four-class brain MRI image classification. The current implemented scope is **image-only classification** with configurable ResNet-18/ResNet-50 image encoders, a classification head, checkpoint evaluation, prediction export, and Grad-CAM visualization. The project name is retained for continuity; text fusion is not currently implemented.

This repository is research software, not a medical device or a substitute for clinical assessment.

## Project metadata

| Field | Value |
|---|---|
| Author | `Tanha Asfika Jaman` |
| Copyright | © 2026 Tanha Asfika Jaman |
| License | MIT ([LICENSE](LICENSE)) |
| GitHub repository | To be provided |

## Motivation

The project provides a reproducible home for experiments on brain MRI image classification while keeping data handling, model code, evaluation, and historical artifacts separate. Reproducibility and privacy constraints take priority over presenting unverified performance claims.

## Implemented methodology

The active model path is:

```text
class-folder MRI images
  → deterministic split manifest with exact duplicate groups kept together
  → training transforms / evaluation transforms
  → ResNet image encoder
  → projection and classification head
  → validation-based checkpoint selection
  → one final test evaluation
  → metrics, predictions, and optional Grad-CAM visualization
```

Text fusion is an archived extension point only. The available captions are class-derived templates, not clinical text or patient reports.

## Repository structure

```text
configs/                         YAML experiment configurations
data/README.md                   data access, privacy, provenance, and split instructions
data/raw/                        local medical images; excluded from Git
data/interim/                    generated manifests; excluded from Git
reports/validation/              leakage, checkpoint, and reproducibility reports
outputs/                         local generated artifacts; excluded from Git
notebooks/                       notebook navigation and retained research history
scripts/data/                    data audit, split, caption, and sanitization utilities
scripts/training/                historical and clean training entry points
scripts/evaluation/              evaluation and reporting entry points
scripts/inference/               prediction entry points
scripts/visualization/           plots and explainability entry points
src/brain_tumor_fusion/          reusable package code
tests/                           behavior-focused tests
ui/                              desktop and Streamlit research demos
_archive/                        preserved legacy code, results, and visuals
```

## Dataset

The expected local layout is:

```text
data/raw/train/<class>/*.jpg
data/raw/test/<class>/*.jpg
```

The supplied data contains 3,264 readable JPEG images. Raw images are not included in GitHub. Obtain them separately and confirm redistribution permissions before sharing any copy.

Dataset provenance, licensing, consent, and de-identification information are currently unresolved. See [data/README.md](data/README.md) for explicit completion placeholders.

## Privacy and leakage notice

The historical split contained 64 exact duplicate groups crossing train/test. The clean pipeline prevents exact SHA-256 duplicate groups from crossing its train/validation/test boundaries. Patient-level separation cannot be guaranteed from the currently available metadata because no patient, subject, study, or acquisition identifiers were supplied.

The audit also found EXIF fields in 17 images. Do not modify `data/raw` in place. Create separate release copies with:

```powershell
python -m scripts.data.sanitize_release_images --input-root data/raw --output-root data/release_safe/raw
```

The sanitization command has not been run as part of this release preparation, and any output still requires data-owner approval.

## Installation

Supported runtime: Python 3.11. The Conda specification provides PyTorch 2.5.1, torchvision 0.20.1, and CUDA 12.1:

```powershell
conda env create -f environment.yml
conda activate brain-tumor-fusion
python -m pip install -e ".[test,ui]"
```

Alternatively, with a supported Python 3.11 interpreter:

```powershell
py -3.11 -m venv .venv_release
.\.venv_release\Scripts\python.exe -m pip install --upgrade pip
.\.venv_release\Scripts\python.exe -m pip install -e ".[test,ui]"
```

The repository `.venv` is machine-specific and invalid; do not reuse it.

## Configuration

Configurations live in `configs/`. Paths are repository-relative and resolved from the configuration location. The clean protocol is configured by `configs/clean_resnet18_image_only.yaml`, with seed 42, 15% validation, and 15% final test fractions.

## Training

Historical training is retained for research history but selects checkpoints using the legacy test split and is not publication-valid:

```powershell
python -m scripts.training.train --cfg configs/resnet18_image_only.yaml
```

The clean pipeline is the release candidate:

```powershell
python -m scripts.data.create_leakage_free_split --data-root data/raw --out data/interim/leakage_free_split.csv --seed 42
python -m scripts.training.train_clean --cfg configs/clean_resnet18_image_only.yaml
```

If the ignored manifest is absent, the clean trainer generates it from the configured seed and fractions. It saves validation-selected checkpoints and evaluates the final test loader only after selection.

## Evaluation and inference

Historical checkpoint evaluation remains available for reproducing old artifacts, but filenames such as `best_ep18_acc0.830.pt` are **HISTORICAL / UNVERIFIED** and must not be interpreted as verified performance:

```powershell
python -m scripts.evaluation.evaluate --cfg configs/resnet50_image_only.yaml --ckpt outputs/checkpoints/best_ep18_acc0.830.pt
python -m scripts.inference.run_inference --cfg configs/resnet50_image_only.yaml --ckpt outputs/checkpoints/best_ep18_acc0.830.pt --out outputs/results/test_predictions.csv
```

## Reproducibility

Record the dataset version/checksum, completed provenance fields, configuration file, seed, split manifest checksum, software environment, checkpoint, and metrics artifact for every experiment. The full validation report is in `reports/validation/research_validation_audit.md`.

## Results status

The repository contains historical predictions and reports. One historical 394-image artifact recomputes to 0.2766497462 accuracy, while some checkpoint filenames claim higher values. These are conflicting historical artifacts, not a verified current result.

**Clean-pipeline performance has not yet been established; runtime training/evaluation remains pending.** No new accuracy, F1, AUC, or other clean-pipeline performance claim is published here.

## Current limitations

- Patient-level separation cannot be guaranteed from the currently available metadata.
- Dataset provenance, license, consent, and de-identification status are not confirmed.
- Seventeen raw images contain EXIF metadata requiring release review.
- Historical model selection used the test set repeatedly.
- Text fusion is not implemented.
- Captions are class-derived templates, not independent clinical text.
- Existing checkpoint-to-result lineage is incomplete.
- No clinical or external validation has been performed.

## Citation

Please cite the software using [CITATION.cff](CITATION.cff). Author and year metadata are set to `Tanha Asfika Jaman` and 2026 as provided. No DOI, journal, publication, or affiliation has been added.

GitHub repository: `GITHUB_REPOSITORY_URL_TO_BE_PROVIDED`

## License

The BrainTumorFusion project code and documentation are released under the [MIT License](LICENSE), copyright © 2026 Tanha Asfika Jaman. Dataset licensing and redistribution permissions remain separate and are still pending confirmation; see [data/README.md](data/README.md).

## Contact

`PROJECT_CONTACT_TO_BE_PROVIDED`
