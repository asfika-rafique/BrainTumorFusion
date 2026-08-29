"""Deterministic exact-duplicate-group split construction.

This module deliberately does not claim patient-level separation. It groups
byte-identical files so that an exact duplicate cannot occur in more than one
new split. Patient, subject, study, and acquisition identifiers are required
for a stronger medical-data split and are not inferred here.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "validation", "test")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_image_records(data_root: str | Path) -> list[dict[str, str | int]]:
    """Collect readable image records from every class folder under raw data."""

    root = Path(data_root).resolve()
    records: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        if len(parts) < 3:
            raise ValueError(f"Expected split/class/image path below {root}: {relative}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Unreadable image {path}: {exc}") from exc
        records.append(
            {
                "path": relative,
                "source_split": parts[0],
                "class_name": parts[1],
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    if not records:
        raise FileNotFoundError(f"No readable images found below {root}")
    return records


def assign_duplicate_groups(
    records: list[dict[str, str | int]],
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> list[dict[str, str | int]]:
    """Assign complete hash groups to train/validation/test deterministically.

    Assignment is approximately class-stratified by image count. A hash group
    is never split, including groups that were already present in both legacy
    source splits.
    """

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")

    by_hash: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for record in records:
        by_hash[str(record["sha256"])].append(record)

    by_class: dict[str, list[list[dict[str, str | int]]]] = defaultdict(list)
    for group in by_hash.values():
        classes = {str(item["class_name"]) for item in group}
        if len(classes) != 1:
            raise ValueError(f"A duplicate hash spans multiple classes: {classes}")
        by_class[next(iter(classes))].append(group)

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for class_name, groups in sorted(by_class.items()):
        rng.shuffle(groups)
        groups.sort(key=lambda group: (-len(group), str(group[0]["sha256"])))
        total = sum(len(group) for group in groups)
        targets = {
            "train": total * (1 - validation_fraction - test_fraction),
            "validation": total * validation_fraction,
            "test": total * test_fraction,
        }
        counts = {split: 0 for split in SPLITS}
        for group in groups:
            # Relative deficit assigns each indivisible group to the split
            # furthest below its requested class-level target.
            split = max(SPLITS, key=lambda name: (targets[name] - counts[name]) / max(targets[name], 1))
            for item in group:
                assignments[str(item["path"])] = split
            counts[split] += len(group)

    return [dict(record, split=assignments[str(record["path"])]) for record in records]


def write_split_manifest(records: list[dict[str, str | int]], output_path: str | Path) -> Path:
    """Write a portable CSV manifest whose paths are relative to ``data/raw``."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "class_name", "source_split", "split", "sha256", "bytes"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in sorted(records, key=lambda item: str(item["path"])))
    return destination


def build_split_manifest(
    data_root: str | Path,
    output_path: str | Path,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Path:
    """Collect, group, assign, and write a leakage-aware split manifest."""

    records = collect_image_records(data_root)
    assigned = assign_duplicate_groups(records, seed, validation_fraction, test_fraction)
    return write_split_manifest(assigned, output_path)
