"""Create release copies of images with embedded EXIF metadata removed.

The input tree is read-only for this command. The output tree must be a new
location, such as ``data/release_safe/raw``; it is excluded from Git until the
dataset owner confirms permission to redistribute the images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from scripts._common import ROOT  # noqa: F401  # adds src/ for repository execution
from brain_tumor_fusion.data.splitting import IMAGE_EXTENSIONS


def sanitize_tree(input_root: Path, output_root: Path) -> tuple[int, int]:
    """Copy images while writing empty EXIF metadata to the new files."""

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if input_root == output_root or output_root.is_relative_to(input_root):
        raise ValueError("Output root must be separate from and outside the input root")
    copied = 0
    metadata_removed = 0
    for source in sorted(input_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        destination = output_root / source.relative_to(input_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            had_exif = bool(image.getexif())
            rgb = image.convert("RGB")
            save_kwargs = {"exif": b""}
            if source.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs.update({"format": "JPEG", "quality": 95})
            rgb.save(destination, **save_kwargs)
        copied += 1
        metadata_removed += int(had_exif)
    return copied, metadata_removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/release_safe/raw"))
    args = parser.parse_args()
    copied, metadata_removed = sanitize_tree(args.input_root, args.output_root)
    print(f"[done] copied={copied} exif-bearing-inputs={metadata_removed} output={args.output_root}")


if __name__ == "__main__":
    main()
