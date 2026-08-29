# BrainTumorFusion

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-BrainTumorFusion-181717?logo=github)](https://github.com/asfika-rafique/BrainTumorFusion)

BrainTumorFusion is research software for four-class brain MRI image classification with a reproducible image-only deep-learning pipeline. The current implementation supports configurable ResNet-18 and ResNet-50 image encoders, a classification head, checkpoint evaluation, prediction export, and Grad-CAM visualization. Text fusion is not currently implemented; the project name is retained for continuity with the original research project.

> This repository is research software, not a medical device or a substitute for clinical assessment.

## Overview

The repository separates reusable package code, experiment configurations, command-line tools, tests, documentation, and historical research artifacts. Its release-candidate workflow keeps exact duplicate image files together across train, validation, and final-test splits.

## Key features

| Area | Current capability |
|---|---|
| Classification | Four image classes: glioma, meningioma, no tumor, and pituitary tumor |
| Encoders | Configurable ResNet-18 and ResNet-50 image encoders |
| Evaluation | Checkpoint evaluation, prediction export, classification metrics, and confusion matrices |
| Explainability | Grad-CAM overlays |
| Reproducibility | YAML configurations, deterministic seed, and exact-duplicate-aware split manifest |
| Interfaces | Command-line tools, Streamlit demo, and Tkinter viewer |

## Current implementation

The active pipeline is image-only. The text encoder and text-fusion path are retained under `_archive/` as unfinished historical work. `data/captions.csv` contains class-derived templates rather than clinical reports or independent patient text, and is not used by the active configurations.

## Methodology

```text
Authorized local MRI images
            ↓
Deterministic exact-duplicate-aware split manifest
            ↓
Training and evaluation transforms
            ↓
ResNet image encoder
            ↓
Projection and classification head
            ↓
Validation-based checkpoint selection
            ↓
One final test evaluation
            ↓
Metrics, predictions, and optional Grad-CAM visualization
```

## Quick start

The raw dataset is not distributed with this repository. After obtaining an authorized local copy, use:

```powershell
git clone https://github.com/asfika-rafique/BrainTumorFusion.git
cd BrainTumorFusion

conda env create -f environment.yml
conda activate brain-tumor-fusion
python -m pip install -e ".[test,ui]"
```

Place the data under `data/raw/train/<class>/` and `data/raw/test/<class>/`, then create the clean split and train:

```powershell
python -m scripts.data.create_leakage_free_split --data-root data/raw --out data/interim/leakage_free_split.csv --seed 42
python -m scripts.training.train_clean --cfg configs/clean_resnet18_image_only.yaml
```

The clean trainer writes validation-selected checkpoints and performs final-test evaluation after model selection. It also creates the ignored split manifest automatically if it is missing.

## Repository structure

```text
configs/                         Experiment configurations
data/                            Dataset instructions; local data is excluded
data/raw/                        Authorized local medical images; ignored
data/interim/                    Generated split manifest; ignored
notebooks/                       Notebook navigation and retained history
outputs/                         Local generated artifacts; ignored by category
reports/                         Curated reports and validation documentation
scripts/data/                    Dataset, split, audit, and sanitization tools
scripts/training/                Historical and clean training entry points
scripts/evaluation/              Evaluation and reporting tools
scripts/inference/               Inference utilities
scripts/visualization/           Plotting and Grad-CAM tools
src/brain_tumor_fusion/          Reusable Python package
tests/                           Automated tests
ui/                              Research demonstration interfaces
_archive/                        Historical code, results, and visualizations
```

## Dataset

Expected local layout:

```text
data/raw/
├── train/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── test/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

The supplied local dataset contains 3,264 readable JPEG images. Raw medical images are excluded from GitHub. Dataset source, licensing, consent, de-identification, and version information remain pending confirmation; see [data/README.md](data/README.md).

## Privacy and data handling

The historical split contained 64 exact duplicate groups crossing train/test. The clean pipeline prevents exact SHA-256 duplicate groups from crossing its train/validation/test boundaries. Patient-level separation cannot be guaranteed from the currently available metadata because no patient, subject, study, or acquisition identifiers were supplied.

The audit found EXIF fields in 17 images. Do not modify `data/raw` in place; use the separate release-copy sanitizer if authorized:

```powershell
python -m scripts.data.sanitize_release_images --input-root data/raw --output-root data/release_safe/raw
```

## Installation

The supported Python version is 3.11. The Conda environment specifies PyTorch 2.5.1, torchvision 0.20.1, and CUDA 12.1:

```powershell
conda env create -f environment.yml
conda activate brain-tumor-fusion
python -m pip install -e ".[test,ui]"
```

The package dependencies are also listed in [requirements.txt](requirements.txt). Do not reuse the repository's machine-specific `.venv`.

## Configuration

YAML files are stored in `configs/`, and paths are resolved relative to the repository. The release-candidate configuration is `configs/clean_resnet18_image_only.yaml` with seed `42`, 15% validation data, 15% final-test data, ResNet-18, and text fusion disabled.

## Training

The clean, release-candidate workflow is:

```powershell
python -m scripts.data.create_leakage_free_split --data-root data/raw --out data/interim/leakage_free_split.csv --seed 42
python -m scripts.training.train_clean --cfg configs/clean_resnet18_image_only.yaml
```

The historical trainer is retained for research history. It evaluates the legacy test split during training and must not be used to claim independent publication results:

```powershell
python -m scripts.training.train --cfg configs/resnet18_image_only.yaml
```

## Evaluation and inference

The clean trainer produces the final-test metrics artifact after validation-based checkpoint selection. Historical checkpoint evaluation and inference commands remain available for reproducing preserved artifacts:

```powershell
python -m scripts.evaluation.evaluate --cfg configs/resnet50_image_only.yaml --ckpt outputs/checkpoints/best_ep18_acc0.830.pt
python -m scripts.inference.run_inference --cfg configs/resnet50_image_only.yaml --ckpt outputs/checkpoints/best_ep18_acc0.830.pt --out outputs/results/test_predictions.csv
```

The checkpoint filename above is **HISTORICAL / UNVERIFIED** and is not a current performance claim. Grad-CAM can be generated with:

```powershell
python -m brain_tumor_fusion.visualization.gradcam --cfg configs/resnet50_image_only.yaml --ckpt outputs/checkpoints/best_ep18_acc0.830.pt --image data/raw/test/glioma_tumor/image.jpg --out outputs/heatmaps/cam_overlay.png
```

## Reproducibility

For each experiment, retain the dataset version or checksum, configuration, seed, split-manifest checksum, software environment, checkpoint, and metrics artifact. The clean workflow uses SHA-256 grouping to keep exact duplicate files in one split, selects checkpoints using validation data only, and evaluates the final test split once after selection.

The detailed validation record is available at [reports/validation/research_validation_audit.md](reports/validation/research_validation_audit.md). It documents the absence of patient-level identifiers and the limitations of historical artifacts.

## Results status

No clean-pipeline performance result has been established in the repository. Runtime training and evaluation remain pending in a supported ML environment.

Preserved historical artifacts include a 394-image report that recomputes to 0.2766497462 accuracy, while some checkpoint filenames contain higher accuracy-like values. These artifacts are retained as **historical and unverified** and are not presented as current model performance.

## Limitations

- Patient-level separation cannot be guaranteed from the currently available metadata.
- Dataset provenance, licensing, consent, and de-identification status are pending confirmation.
- Seventeen raw images contain EXIF metadata requiring release review.
- Text fusion is not currently implemented.
- Captions are class-derived templates, not independent clinical text.
- Existing checkpoint-to-result lineage is incomplete.
- No clinical or external validation has been performed.

## Citation

If you use this software, cite [CITATION.cff](CITATION.cff):

```text
Tanha Asfika Jaman. BrainTumorFusion. 2026.
```

GitHub repository: [https://github.com/asfika-rafique/BrainTumorFusion](https://github.com/asfika-rafique/BrainTumorFusion)

No DOI, journal, publication, or affiliation is currently specified.

## License

BrainTumorFusion code and documentation are released under the [MIT License](LICENSE), copyright © 2026 Tanha Asfika Jaman. Dataset licensing and redistribution permissions are separate and remain pending confirmation.

## Contact

Contact information is intentionally not specified in this repository.
