"""Generate deterministic class-derived captions for the local image index."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def caption_for_class(class_name: str) -> str:
    label = class_name.removesuffix("_tumor").replace("_", " ").strip()
    return "no tumor MRI" if label == "no" else f"{label} tumor MRI"


def scan(root: Path, project_root: Path) -> list[tuple[str, str]]:
    return [
        (image.relative_to(project_root).as_posix(), caption_for_class(image.parent.name))
        for image in sorted(root.rglob("*"))
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/captions.csv"))
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    rows = [("image", "caption")]
    for root in args.roots:
        rows.extend(scan((project_root / root).resolve(), project_root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print(f"[done] wrote {len(rows) - 1} caption rows to {args.out}")


if __name__ == "__main__":
    main()
