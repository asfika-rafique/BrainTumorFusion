"""Torchvision transforms used for training, evaluation, and inference."""

from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert images with palettes or alpha channels to three-channel RGB."""

    return image.convert("RGB") if image.mode != "RGB" else image


def get_train_test_transforms(
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
):
    """Return the augmentation pipeline and deterministic evaluation pipeline."""

    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    train = transforms.Compose(
        [
            transforms.Lambda(ensure_rgb),
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test = transforms.Compose(
        [
            transforms.Lambda(ensure_rgb),
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train, test


def make_infer_transform(
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
):
    """Return the deterministic transform used by single-image inference."""

    _, test = get_train_test_transforms(image_size, mean, std)
    return test


def build_eval_transform(image_size: int = 224):
    """Backward-compatible alias for older notebooks."""

    return make_infer_transform(image_size)


@torch.no_grad()
def pil_to_batch(
    image: Image.Image,
    device: torch.device,
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    """Transform one PIL image and add a batch dimension."""

    return make_infer_transform(image_size, mean, std)(image).unsqueeze(0).to(device)
