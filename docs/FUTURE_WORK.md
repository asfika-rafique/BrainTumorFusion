# Future work

The following are future directions, not current capabilities:

- Confirm dataset provenance, permissions, consent, de-identification, and version metadata.
- Add patient-level grouping when authoritative subject metadata becomes available.
- Implement and validate a documented freeze/unfreeze transfer-learning schedule.
- Consider additional validation of the training-only class-weight calculation across experiments.
- Add a scheduler only if a future experiment protocol requires one; the current trainer records that no scheduler is configured.
- Evaluate multimodal MRI sequences or text only after real, independently auditable modalities are available.
- Add external validation, probability calibration, uncertainty estimation, and domain-shift analysis.
- Consider segmentation/localization only with appropriate ground-truth annotations.
