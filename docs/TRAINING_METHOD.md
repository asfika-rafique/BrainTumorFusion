# Training method

The clean release-candidate configuration is `configs/clean_resnet18_image_only.yaml`.

| Setting | Active value |
|---|---|
| Seed | 42 |
| Image size | 224 |
| Batch size | 16 |
| Epochs | 12 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Class weights | Derived from clean training counts when `derive_class_weights=true` |
| AMP | Enabled only when CUDA is available and the flag is true |
| Scheduler | Not implemented |
| Freeze/unfreeze schedule | Not implemented |

The clean configuration derives balanced cross-entropy weights from the clean training dataset only using `total / (number_of_classes × class_count)`. Validation and final-test samples are not used for this calculation. Historical configurations retain their previous explicit weights and behavior.

The clean trainer selects the best checkpoint using validation accuracy only, reloads that checkpoint, and evaluates the final test split once. Clean checkpoints now store model state, epoch, validation loss/accuracy, class names, config path, seed, training class counts, effective class weights, optimizer state, and explicit `None` scheduler state when no scheduler is configured. AMP scaler state is stored when CUDA AMP is active.
