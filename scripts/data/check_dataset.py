"""Audit split counts, unreadable images, and exact duplicate image hashes."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

from scripts._common import ROOT  # noqa: F401  # adds src/ for repository execution
from brain_tumor_fusion.data.splitting import file_sha256

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def audit(data_root: Path) -> dict:
    hashes: dict[str, list[Path]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = {}
    unreadable = []
    for split in sorted(path for path in data_root.iterdir() if path.is_dir()):
        counts[split.name] = {}
        for class_dir in sorted(path for path in split.iterdir() if path.is_dir()):
            images = [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            counts[split.name][class_dir.name] = len(images)
            for image_path in images:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    hashes[file_sha256(image_path)].append(image_path)
                except Exception as exc:
                    unreadable.append(f"{image_path}: {exc}")
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split = [paths for paths in duplicates if len({path.relative_to(data_root).parts[0] for path in paths}) > 1]
    return {
        "counts": counts,
        "image_files": sum(sum(classes.values()) for classes in counts.values()),
        "duplicate_groups": len(duplicates),
        "duplicate_files": sum(len(paths) for paths in duplicates),
        "cross_split_groups": len(cross_split),
        "unreadable": unreadable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    result = audit(args.data_root)
    print(f"image_files={result['image_files']}")
    print(f"duplicate_groups={result['duplicate_groups']}")
    print(f"duplicate_files={result['duplicate_files']}")
    print(f"cross_split_groups={result['cross_split_groups']}")
    print(f"unreadable={len(result['unreadable'])}")
    for split, classes in result["counts"].items():
        print(f"{split}: {classes}")


if __name__ == "__main__":
    main()
