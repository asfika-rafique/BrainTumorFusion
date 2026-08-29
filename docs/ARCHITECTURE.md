# Active architecture

The active implementation is an image-only `FusionNet` wrapper around a configurable torchvision ResNet encoder. It is not the paper's multimodal architecture.

```text
RGB MRI image (3 × 224 × 224)
        ↓
ResNet-18 / ResNet-34 / ResNet-50 feature extractor
        ↓
Global average pooling when required
        ↓
Image projection: encoder_dim → 512
        ↓
BatchNorm1d → ReLU → Dropout
        ↓
Fusion block: 512 → 512 → 256
        ↓
BatchNorm1d → ReLU → Dropout after each linear stage
        ↓
Linear classification head: 256 → 4 logits
```

The default clean configuration uses ResNet-18, ImageNet weights, `img_out_dim=512`, `fusion_hidden=512`, dropout `0.2`, and four output classes. The ResNet-50 configuration uses an encoder dimension of 2048 and dropout `0.3`.

The text branch is guarded by `use_text`, but the active configurations set it to false and the text encoder is a placeholder that raises `NotImplementedError`. No multimodal feature concatenation is active.

Exact trainable parameter totals and runtime memory were not measured in this environment because the supported PyTorch/torchvision runtime is unavailable. Do not use the paper's approximate parameter table as a repository result.
