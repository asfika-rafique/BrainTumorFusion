"""Plot class counts by reading the current raw dataset on disk."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_classes(split: Path) -> dict[str, int]:
    return {
        class_dir.name: sum(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in class_dir.rglob("*")
        )
        for class_dir in sorted(split.iterdir())
        if class_dir.is_dir()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("reports/figures/dataset_distribution.png"))
    args = parser.parse_args()
    train = count_classes(args.data_root / "train")
    test = count_classes(args.data_root / "test")
    labels = sorted(set(train) | set(test))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, title, values in zip(axes, ["Train", "Test"], [train, test]):
        counts = [values.get(label, 0) for label in labels]
        bars = axis.bar(labels, counts)
        axis.set_title(f"{title} split")
        axis.set_ylabel("Images")
        axis.tick_params(axis="x", rotation=35)
        axis.bar_label(bars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[done] saved {args.out}")


if __name__ == "__main__":
    main()
