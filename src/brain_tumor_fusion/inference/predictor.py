"""Checkpoint loading and single-image inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
import numpy as np
from PIL import Image

from ..models.fusion_model import FusionNet
from ..preprocessing.transforms import make_infer_transform
from ..training.engine import build_model
from ..utils import config_root, resolve_config_paths
from ..visualization.gradcam import _overlay_on_image, gradcam_single

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def get_device(requested: str = "cuda") -> torch.device:
    """Select CUDA only when requested and available; otherwise use CPU."""

    return torch.device("cuda" if requested.lower() == "cuda" and torch.cuda.is_available() else "cpu")


def build_default_model(cfg: dict[str, Any]) -> FusionNet:
    """Build a model from a loaded configuration."""

    return build_model(cfg, num_classes=int(cfg["model"].get("num_classes", len(CLASS_NAMES))))


def _checkpoint_state(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_fusion_model(cfg: dict[str, Any], checkpoint_path: str | Path | None, device=None) -> FusionNet:
    """Build and optionally load a checkpoint, failing on incompatible weights."""

    device = device or get_device(str(cfg.get("device", "cuda")))
    model = build_default_model(cfg).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(_checkpoint_state(checkpoint), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint architecture mismatch; missing={list(missing)[:5]}, "
                f"unexpected={list(unexpected)[:5]}"
            )
    return model.eval()


def predict_one(model: torch.nn.Module, image: Image.Image, device: torch.device, image_size: int = 224) -> dict[str, Any]:
    """Return top-two labels and probabilities for one image."""

    tensor = make_infer_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    order = probabilities.argsort(descending=True)
    top, second = int(order[0]), int(order[1])
    return {
        "top_name": CLASS_NAMES[top],
        "top_conf": float(probabilities[top]),
        "second_name": CLASS_NAMES[second],
        "second_conf": float(probabilities[second]),
        "probabilities": probabilities.tolist(),
    }


def generate_heatmap(
    model: torch.nn.Module,
    image: Image.Image,
    output_dir: str | Path,
    device: torch.device,
    image_size: int = 224,
) -> tuple[Any, Path]:
    """Generate a Grad-CAM overlay from model activations and save it."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor = make_infer_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        target = int(model(tensor).argmax(dim=1).item())
    heatmap = gradcam_single(model, tensor, target_idx=target)
    overlay = _overlay_on_image(
        np.array(image.convert("RGB").resize((image_size, image_size))),
        heatmap,
        alpha=0.4,
    )
    output_path = output_dir / "inference_gradcam.png"
    Image.fromarray(overlay).save(output_path)
    return overlay, output_path


def evaluate_directory(model, root: Path, device: torch.device, output_csv: Path, image_size: int = 224) -> None:
    """Write predictions for a class-folder directory to a CSV file."""

    rows = []
    for image_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}):
        true_label = image_path.parent.name
        result = predict_one(model, Image.open(image_path), device, image_size)
        rows.append(
            {
                "filename": image_path.relative_to(root).as_posix(),
                "y_true": true_label,
                "y_pred": result["top_name"],
                "probs_json": json.dumps(result["probabilities"]),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["filename", "y_true", "y_pred", "probs_json"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out_csv", required=True, type=Path)
    parser.add_argument("--split", type=Path, default=None, help="Optional class-folder directory to evaluate")
    args = parser.parse_args()
    with args.cfg.open("r", encoding="utf-8") as handle:
        cfg = resolve_config_paths(yaml.safe_load(handle) or {}, config_root(args.cfg.resolve()))
    device = get_device(str(cfg.get("device", "cuda")))
    model = load_fusion_model(cfg, args.ckpt, device)
    split = args.split or Path(cfg["paths"]["test_dir"])
    evaluate_directory(model, split, device, args.out_csv, int(cfg["train"].get("img_size", 224)))
    print(f"[done] wrote predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
