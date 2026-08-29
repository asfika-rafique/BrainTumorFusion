# Paper-to-repository gap analysis

The supplied paper, *FusionNet: A Deep Multimodal Feature Fusion Framework for MRI-Based Brain Tumor Classification*, is a structural and documentation reference. The repository is the implementation source of truth.

| Paper component | Repository status | Existing evidence | Missing or different | Action |
|---|---|---|---|---|
| Four-class dataset | Partially verified | `data/README.md`, `docs/DATASET.md` | Provenance and permissions unknown | Document exact local audit and placeholders |
| Dataset distribution | Verified locally | `scripts/audit_dataset.py` | Paper descriptions are not evidence | Use generated local artifacts |
| 224×224 preprocessing | Verified | `preprocessing/transforms.py` | Actual pipeline uses resize/crop, not paper's full narrative | Document code |
| Z-score normalization | Not implemented | `preprocessing/transforms.py` | ImageNet mean/std are used | Mark paper claim as different |
| Augmentation | Partially verified | `preprocessing/transforms.py` | Crop, horizontal flip, brightness/contrast jitter only | Document exact values |
| Label mapping | Verified differently | `data/datasets.py` | Alphabetical mapping differs from paper example | Document repository mapping |
| Leakage-aware split | Verified for exact duplicates | `data/splitting.py`, clean manifest | Patient grouping and near-duplicates unavailable | Keep limitation visible |
| ResNet encoder | Verified | `models/image_encoder.py` | Supports 18/34/50, not only paper framing | Document active support |
| Projection block | Verified | `models/fusion_model.py` | Actual image path is 512→512→256 | Document code |
| Multimodal/text fusion | Not implemented | `models/text_encoder.py`, `use_text=false` | Text encoder is a placeholder | Keep archived/disabled |
| Classifier | Verified | `models/fusion_model.py` | Actual head is 256→4 after fusion block | Document code |
| ImageNet transfer learning | Configurable | `image_encoder.py` | No measured clean run in current environment | Document flag |
| Two-stage freezing/unfreezing | Not implemented | `training/engine.py` | No freeze schedule or parameter groups | Mark NOT IMPLEMENTED |
| AdamW | Verified | `training/engine.py` | Active values differ from paper's typical values | Document config |
| Class-weighted loss | Derived for clean training | `training/engine.py`, `clean_resnet18_image_only.yaml` | Historical configs retain explicit weights | Document clean-only calculation |
| Scheduler | Not implemented | `training/engine.py` | No ReduceLROnPlateau | Mark NOT IMPLEMENTED |
| AMP | Conditional support | `training/engine.py` | Only enabled on CUDA; scaler state is recorded when active | Document status |
| Checkpointing | Partially verified | `training/engine.py`, validation manifest | Historical lineage is incomplete | Preserve unverified labels |
| Baseline CNN/FusionNet comparison | Not verified | Historical output files only | No clean comparable runs | Mark NOT RUN/UNVERIFIED |
| Accuracy/F1/AUC results | Historical only | `reports/validation/` and archived outputs | No clean verified result | Do not promote paper numbers |
| Training curves | Not present as verified artifact | Repository search | No current traceable curve | Mark NOT AVAILABLE |
| Confusion matrix | Historical artifact | `outputs/results/` | Not clean verified performance | Label historical/unverified |
| Grad-CAM | Implemented | `visualization/gradcam.py` | Attribution limits remain | Document limitations |
| UI/inference | Implemented research interfaces | `ui/`, `inference/predictor.py` | Requires compatible local runtime/checkpoint | Document research-only use |
| Confidence calibration | Not implemented | Predictor returns softmax probabilities | No calibration analysis | Mark NOT CALIBRATED |
| Reproducibility | Partially implemented | configs, seed, clean split | Runtime environment unavailable here | Provide commands and status |

The paper's numerical claims, clinical language, and conceptual ablations are not copied into repository results unless independently supported by repository artifacts.
