# Dataset

This page describes the local dataset observed during the 2026-08-29 audit. The raw images are not included in GitHub and must be obtained and used only with authorization from the data owner.

## Observed structure

```text
data/raw/
├── train/<class>/*.jpg
└── test/<class>/*.jpg
```

The observed class folders are `glioma_tumor`, `meningioma_tumor`, `no_tumor`, and `pituitary_tumor`. The active loader maps these folders alphabetically to indices 0–3:

| Index | Folder |
|---:|---|
| 0 | `glioma_tumor` |
| 1 | `meningioma_tumor` |
| 2 | `no_tumor` |
| 3 | `pituitary_tumor` |

## Counts observed locally

The audit found 3,264 readable JPEG files. The exact split counts are below; percentages are calculated from the observed 3,264 files.

| Original folder | Glioma | Meningioma | No tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| `train` | 826 | 822 | 395 | 827 | 2,870 |
| `test` | 100 | 115 | 105 | 74 | 394 |
| **Total** | **926** | **937** | **500** | **901** | **3,264** |

Overall class percentages are: glioma 28.370098%, meningioma 28.707108%, no tumor 15.318627%, and pituitary 27.604167%.

The deterministic clean manifest generated with seed 42 contains:

| Clean split | Glioma | Meningioma | No tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| `train` | 648 | 655 | 350 | 630 | 2,283 |
| `validation` | 139 | 141 | 75 | 136 | 491 |
| `test` | 139 | 141 | 75 | 135 | 490 |

These values are generated from `data/interim/leakage_free_split.csv`, not copied from the paper.

## File and image properties

- 3,264/3,264 files were readable during the audit.
- All observed files were JPEG and loaded as RGB by Pillow, with three channels.
- 440 distinct width-height combinations were observed; the most common was 512×512 for 2,341 images.
- Observed grayscale intensity extrema across files ranged from minimum 0–50 and maximum 130–255 after conversion for audit only.
- EXIF metadata was present in 17 raw images. The raw tree was not modified; use `scripts.data.sanitize_release_images` only for separately authorized release copies.
- Possible perceptual-duplicate detection and patient identity resolution were not performed.

## Provenance and permissions

Dataset source, license, consent status, de-identification procedure, and version/checksum remain to be confirmed. No patient, subject, study, or acquisition identifiers were supplied, so patient-level separation cannot be guaranteed.

## Audit artifacts

Run:

```powershell
python -m scripts.audit_dataset --data-root data/raw --output-dir artifacts
```

This writes the ignored local files `artifacts/dataset_audit.json`, `artifacts/dataset_audit.csv`, `artifacts/dataset_statistics.csv`, `artifacts/class_distribution.png`, and `artifacts/sample_mri_grid.png`.
