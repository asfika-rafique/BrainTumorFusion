# Paper claim audit

This is a concise claim-level record for the supplied paper. It does not validate the paper; it prevents paper statements from being mistaken for executed repository experiments.

| Paper claim | Repository evidence | Status |
|---|---|---|
| Four MRI classes: glioma, meningioma, pituitary, no-tumor | Local class folders and loader | PARTIALLY VERIFIED; exact folder names are documented |
| ResNet-based image feature extraction | `models/image_encoder.py` | VERIFIED BY THIS REPOSITORY |
| Projection/fusion refinement | `models/fusion_model.py` | PARTIALLY VERIFIED; active implementation is image-branch feature integration |
| Multimodal text fusion | Placeholder text encoder; active configs disable text | NOT IMPLEMENTED |
| Two-stage transfer learning | No freeze/unfreeze logic in training engine | NOT IMPLEMENTED |
| AdamW, scheduler, AMP, checkpoint state as described | AdamW and conditional AMP exist; no scheduler is configured, while clean checkpoints retain optimizer/scaler metadata when available | PARTIALLY VERIFIED |
| Historical ~83% result | Checkpoint filename conflicts with preserved reports/CSV | CONTRADICTED/UNVERIFIED as a repository performance claim |
| Historical 394-image report accuracy | JSON records 0.2766497462; CSV recomputes 0.2918781726 | CONTRADICTED between artifacts; not current performance |
| Baseline and ablation improvements | No clean, traceable comparable runs | NOT VERIFIED |
| Clinical relevance/robust generalization | No external or clinical validation | NOT VERIFIED |
| Grad-CAM visualization | Active implementation and preserved historical grid | PARTIALLY VERIFIED; does not establish clinical correctness |

No paper metric is used as a current repository result. The clean-pipeline statement remains: **No clean-pipeline performance result has been established in the repository.**
