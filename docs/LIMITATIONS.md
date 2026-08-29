# Limitations

The following limitations are verified or directly supported by the current repository:

- Dataset source, license, consent, de-identification procedure, and version are not confirmed.
- Patient-level separation cannot be guaranteed because no patient, subject, study, or acquisition identifiers were supplied.
- The historical raw split contains 64 exact duplicate groups crossing train/test; the clean manifest addresses exact hashes only.
- Seventeen raw images contain EXIF metadata and require release review.
- The active implementation is single-modality image-only; text fusion is not implemented.
- Captions are class-derived templates rather than clinical reports or independent patient text.
- Historical checkpoint/result lineage is incomplete and internally inconsistent.
- No verified clean-pipeline performance result has been established.
- No external, clinical, scanner-domain, or prospective validation has been performed.
- Prediction probabilities have not been clinically calibrated.
- Grad-CAM provides coarse attribution and does not prove lesion localization or clinical correctness.
- Dataset and preprocessing bias, scanner/domain shift, and class imbalance may affect generalization.
