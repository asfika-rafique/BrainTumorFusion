"""Minimal desktop viewer for local inference and Grad-CAM."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import torch
import yaml
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_fusion.inference.predictor import generate_heatmap, get_device, load_fusion_model, predict_one
from brain_tumor_fusion.utils import config_root, resolve_config_paths


class App(tk.Tk):
    def __init__(self, cfg: dict, checkpoint: Path | None) -> None:
        super().__init__()
        self.title("BrainTumorFusion — Inference")
        self.device = get_device(str(cfg.get("device", "cuda")))
        self.cfg = cfg
        self.model = load_fusion_model(cfg, checkpoint, self.device)
        self.image = None
        self.left_photo = None
        self.right_photo = None
        self.prediction = tk.StringVar()
        self.confidence = tk.StringVar()
        tk.Button(self, text="Load MRI image", command=self.load_image).grid(row=0, column=0, padx=8, pady=8)
        tk.Label(self, textvariable=self.prediction).grid(row=0, column=1, padx=8)
        tk.Label(self, textvariable=self.confidence).grid(row=0, column=2, padx=8)
        tk.Button(self, text="Generate Grad-CAM", command=self.heatmap).grid(row=0, column=3, padx=8)
        self.left = tk.Label(self, bg="black", width=480, height=360)
        self.left.grid(row=1, column=0, columnspan=2, padx=8, pady=8)
        self.right = tk.Label(self, bg="black", width=480, height=360)
        self.right.grid(row=1, column=2, columnspan=2, padx=8, pady=8)

    def _show(self, image: Image.Image, target: tk.Label, side: str) -> None:
        image = image.copy()
        image.thumbnail((480, 360))
        photo = ImageTk.PhotoImage(image)
        target.configure(image=photo)
        setattr(self, f"{side}_photo", photo)

    def load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            self.image = Image.open(path).convert("RGB")
            result = predict_one(self.model, self.image, self.device)
            self.prediction.set(f"Prediction: {result['top_name']}")
            self.confidence.set(f"Confidence: {result['top_conf']:.2%}")
            self._show(self.image, self.left, "left")
            self.right.configure(image="")
        except Exception as exc:
            messagebox.showerror("Inference error", str(exc))

    def heatmap(self) -> None:
        if self.image is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        try:
            overlay, _ = generate_heatmap(
                self.model,
                self.image,
                self.cfg["paths"]["heatmap_dir"],
                self.device,
                int(self.cfg["train"].get("img_size", 224)),
            )
            self._show(Image.fromarray(overlay), self.right, "right")
        except Exception as exc:
            messagebox.showerror("Grad-CAM error", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=Path("configs/resnet50_image_only.yaml"))
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    cfg_path = (ROOT / args.cfg).resolve() if not args.cfg.is_absolute() else args.cfg.resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = resolve_config_paths(yaml.safe_load(handle) or {}, config_root(cfg_path))
    checkpoint = (ROOT / args.ckpt).resolve() if not args.ckpt.is_absolute() else args.ckpt.resolve()
    App(cfg, checkpoint).mainloop()


if __name__ == "__main__":
    main()
