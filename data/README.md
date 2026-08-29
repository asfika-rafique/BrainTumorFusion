# Data

The project expects four class folders under `data/raw/train` and `data/raw/test`: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, and `pituitary_tumor`. Raw medical images are excluded from GitHub and must be obtained separately under terms approved by the data owner.

`captions.csv` is a generated path-to-caption index. Its captions are class-derived templates, not clinical reports, radiology reports, or independent patient text. The active pipeline is image-only and does not use this file.

## Dataset provenance and permissions

The supplied project does not identify the dataset source, license, consent status, or de-identification procedure. These fields must be completed by the project owner before release:

```text
DATASET_SOURCE_TO_BE_CONFIRMED
DATASET_LICENSE_TO_BE_CONFIRMED
DATASET_CONSENT_STATUS_TO_BE_CONFIRMED
DATASET_DEIDENTIFICATION_STATUS_TO_BE_CONFIRMED
DATASET_VERSION_OR_CHECKSUM_TO_BE_CONFIRMED
```

Do not upload raw images, patient metadata, or release copies to GitHub until those permissions are confirmed. Patient-level separation cannot be guaranteed from the currently available metadata: no patient, subject, study, or acquisition IDs were supplied.

## Leakage-aware split

Create the deterministic manifest with seed 42:

```powershell
python -m scripts.data.create_leakage_free_split --data-root data/raw --out data/interim/leakage_free_split.csv --seed 42
```

The manifest keeps exact SHA-256 duplicate groups together. It does not establish patient-level separation.

## Release-copy metadata sanitization

The audit found EXIF metadata in 17 raw images. The raw tree must not be modified in place. To create separate release copies with EXIF removed:

```powershell
python -m scripts.data.sanitize_release_images --input-root data/raw --output-root data/release_safe/raw
```

This command has not been run by the release preparation pass. It creates new JPEG files under an excluded directory; the output still requires data-owner permission and a privacy review.
