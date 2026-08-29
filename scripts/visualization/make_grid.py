"""Create a contact sheet from saved Grad-CAM heatmaps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image


def make_grid(image_folder: Path, out_path: Path, columns: int = 6, image_size: int = 224) -> None:
    images = sorted(image_folder.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG heatmaps found in {image_folder}")
    rows = math.ceil(len(images) / columns)
    grid = Image.new("RGB", (columns * image_size, rows * image_size), color="black")
    for index, path in enumerate(images):
        with Image.open(path) as image:
            grid.paste(image.convert("RGB").resize((image_size, image_size)), ((index % columns) * image_size, (index // columns) * image_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"[done] saved {len(images)} heatmaps to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/heatmaps"))
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/gradcam_grid.png"))
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()
    make_grid(args.input, args.out, args.columns)


if __name__ == "__main__":
    main()
