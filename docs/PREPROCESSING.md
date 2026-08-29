# Preprocessing

The active transforms are implemented in `src/brain_tumor_fusion/preprocessing/transforms.py`.

## Training transform

1. Convert to RGB.
2. Resize to 224×224 with bilinear interpolation.
3. Apply `RandomResizedCrop(224, scale=(0.85, 1.0))`.
4. Apply `RandomHorizontalFlip(p=0.5)`.
5. Apply `ColorJitter(brightness=0.1, contrast=0.1)`.
6. Convert to a tensor.
7. Normalize with ImageNet channel statistics: mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

## Evaluation and inference transform

1. Convert to RGB.
2. Resize to 224×224 with bilinear interpolation.
3. Center-crop to 224×224.
4. Convert to a tensor.
5. Apply the same ImageNet normalization.

Evaluation and inference are deterministic with respect to the transform pipeline. The code does not implement per-image z-score normalization, rotation, vertical flips, explicit denoising, or scanner-specific intensity correction. Those operations are discussed in the paper but must not be described as active implementation.
