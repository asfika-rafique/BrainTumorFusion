"""Grad-CAM implementation for convolutional image encoders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class _Hook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.feature_map = None
        self.gradient = None
        self.forward_handle = module.register_forward_hook(self._forward)
        self.backward_handle = module.register_full_backward_hook(self._backward)

    def _forward(self, _module, _inputs, output):
        self.feature_map = (output[0] if isinstance(output, (tuple, list)) else output).detach()

    def _backward(self, _module, _inputs, output):
        gradient = output[0] if isinstance(output, (tuple, list)) else output
        self.gradient = gradient.detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def _pick_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    encoder = getattr(model, "image_encoder", None) or getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "gradcam_target"):
        return encoder.gradcam_target
    if encoder is not None and hasattr(encoder, "layer4"):
        block = list(encoder.layer4.children())[-1]
        for candidate in ("conv3", "conv2"):
            if hasattr(block, candidate):
                return getattr(block, candidate)
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    if last is None:
        raise RuntimeError("No convolutional layer found for Grad-CAM")
    return last


def _to_heatmap(cam: torch.Tensor) -> np.ndarray:
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    return cam.clamp(0, 1).detach().cpu().numpy()


def gradcam_single(model: torch.nn.Module, image_tensor: torch.Tensor, target_idx: int | None = None) -> np.ndarray:
    """Return a normalized Grad-CAM heatmap for a single input tensor."""

    model.eval()
    hook = _Hook(_pick_target_layer(model))
    logits = model(image_tensor)
    target_idx = int(logits.argmax(dim=1).item()) if target_idx is None else target_idx
    model.zero_grad(set_to_none=True)
    logits[:, target_idx].sum().backward()
    feature_map, gradient = hook.feature_map, hook.gradient
    hook.close()
    if feature_map is None or gradient is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    return _to_heatmap(cam[0, 0])


def _overlay_on_image(original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a blue-to-red activation map with an RGB image."""

    if original.shape[:2] != heatmap.shape:
        original = np.asarray(Image.fromarray(original).resize((heatmap.shape[1], heatmap.shape[0])))
    colors = np.stack([heatmap, np.zeros_like(heatmap), 1 - heatmap], axis=-1) * 255
    return (original.astype(np.float32) * (1 - alpha) + colors * alpha).clip(0, 255).astype(np.uint8)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Return a PIL image containing a Grad-CAM overlay."""

    return Image.fromarray(_overlay_on_image(np.asarray(image.convert("RGB")), heatmap, alpha))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM overlay for one MRI image")
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/heatmaps/cam_overlay.png"))
    args = parser.parse_args()

    import yaml

    from ..inference.predictor import get_device, load_fusion_model
    from ..preprocessing.transforms import make_infer_transform
    from ..utils import config_root, resolve_config_paths

    with args.cfg.open("r", encoding="utf-8") as handle:
        cfg = resolve_config_paths(yaml.safe_load(handle) or {}, config_root(args.cfg.resolve()))
    device = get_device(str(cfg.get("device", "cuda")))
    model = load_fusion_model(cfg, args.ckpt, device)
    image = Image.open(args.image).convert("RGB")
    tensor = make_infer_transform(int(cfg["train"].get("img_size", 224)))(image).unsqueeze(0).to(device)
    heatmap = gradcam_single(model, tensor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    overlay_heatmap(image.resize((tensor.shape[-1], tensor.shape[-2])), heatmap).save(args.out)
    print(f"[done] saved {args.out}")


if __name__ == "__main__":
    main()
