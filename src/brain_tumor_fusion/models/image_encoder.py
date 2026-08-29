"""Torchvision image backbones with a stable feature-extractor interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as torchvision_models


BACKBONES = {
    "resnet18": (torchvision_models.resnet18, 512, torchvision_models.ResNet18_Weights),
    "resnet34": (torchvision_models.resnet34, 512, torchvision_models.ResNet34_Weights),
    "resnet50": (torchvision_models.resnet50, 2048, torchvision_models.ResNet50_Weights),
}


def build_image_backbone(name: str = "resnet18", pretrained: bool = True):
    """Build a supported torchvision ResNet and return it with its feature size."""

    key = name.lower()
    if key not in BACKBONES:
        raise ValueError(f"Unknown image encoder {name!r}; choose from {sorted(BACKBONES)}")
    constructor, feature_dim, weights_enum = BACKBONES[key]
    model = constructor(weights=weights_enum.DEFAULT if pretrained else None)
    return model, feature_dim


def _strip_classifier(model: nn.Module) -> nn.Module:
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    return model


def _last_conv_for_gradcam(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        block = list(model.layer4.children())[-1]
        for candidate in ("conv3", "conv2"):
            if hasattr(block, candidate):
                return getattr(block, candidate)
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM")
    return last


class ImageEncoder(nn.Module):
    """ResNet feature extractor used by the image-only experiments."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        backbone, feature_dim = build_image_backbone(name, pretrained=pretrained)
        self.backbone_name = name.lower()
        self.backbone = _strip_classifier(backbone)
        self.out_dim = feature_dim
        self.gradcam_target = _last_conv_for_gradcam(self.backbone)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return features
