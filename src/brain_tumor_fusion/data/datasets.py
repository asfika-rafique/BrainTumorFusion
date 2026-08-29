"""Image-folder dataset discovery with optional caption alignment."""

from __future__ import annotations

import csv
import hashlib
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..preprocessing.transforms import get_train_test_transforms
from ..utils import resolve_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: str | Path) -> list[Path]:
    """Return sorted image files below a split directory."""

    return sorted(
        path
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normalise_key(value: str | Path) -> str:
    return Path(str(value).replace("\\", "/")).as_posix().lstrip("./")


def read_captions(caption_csv_path: str | Path) -> dict[str, str]:
    """Read captions keyed by project-relative path.

    Basename fallback is retained only for unambiguous basenames. This avoids
    silently assigning the caption for one ``image.jpg`` to every split.
    """

    path = Path(caption_csv_path)
    if not path.is_file():
        return {}

    mapping: dict[str, str] = {}
    by_basename: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = row.get("filename") or row.get("file") or row.get("image") or ""
            caption = (row.get("caption") or row.get("text") or "").strip()
            if not filename:
                continue
            mapping[_normalise_key(filename)] = caption
            by_basename.setdefault(Path(filename).name, set()).add(caption)

    for basename, captions in by_basename.items():
        if len(captions) == 1:
            mapping.setdefault(f"__basename__/{basename}", next(iter(captions)))
    return mapping


class BrainTumorDataset(Dataset):
    """Class-folder image dataset compatible with the existing checkpoints."""

    def __init__(
        self,
        image_paths: list[Path],
        class_to_idx: dict[str, int],
        transform,
        use_text: bool = False,
        captions_map: dict[str, str] | None = None,
        tokenizer: Any | None = None,
        max_len: int = 48,
        project_root: str | Path | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_text = use_text
        self.captions_map = captions_map or {}
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.project_root = Path(project_root).resolve() if project_root else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def _caption_for(self, path: Path) -> str:
        if self.project_root:
            try:
                key = path.resolve().relative_to(self.project_root).as_posix()
            except ValueError:
                key = path.as_posix()
        else:
            key = path.as_posix()
        return self.captions_map.get(key, self.captions_map.get(f"__basename__/{path.name}", ""))

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.image_paths[index]
        label_name = path.parent.name
        if label_name not in self.class_to_idx:
            raise KeyError(f"Class folder {label_name!r} is not in the training class map")

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        sample: dict[str, Any] = {
            "images": image,
            "labels": torch.tensor(self.class_to_idx[label_name], dtype=torch.long),
            "paths": str(path),
        }

        if self.use_text:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer missing while use_text=True")
            tokens = self.tokenizer(
                self._caption_for(path),
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            sample["input_ids"] = tokens["input_ids"].squeeze(0)
            sample["attention_mask"] = tokens["attention_mask"].squeeze(0)
        return sample


def gather_split(root: str | Path) -> tuple[list[Path], dict[str, int], list[str]]:
    """Discover class names and images from a class-folder split."""

    root_path = Path(root)
    classes = sorted(path.name for path in root_path.iterdir() if path.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class directories found in {root_path}")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    paths = [image for name in classes for image in list_images(root_path / name)]
    if not paths:
        raise FileNotFoundError(f"No supported image files found in {root_path}")
    return paths, class_to_idx, classes


def _auto_workers(configured: int | None) -> int:
    if configured is None:
        return 0 if platform.system().lower().startswith("win") else 2
    if platform.system().lower().startswith("win") and configured > 0:
        return 0
    return max(0, int(configured))


def make_loaders_from_cfg(
    cfg: dict[str, Any], project_root: str | Path | None = None
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train/test loaders from a resolved or repository-relative config."""

    base = Path(project_root or Path.cwd()).resolve()
    paths = cfg["paths"]
    train_dir = resolve_path(paths["train_dir"], base)
    test_dir = resolve_path(paths["test_dir"], base)
    use_text = bool(cfg.get("use_text", False))
    if use_text:
        raise NotImplementedError(
            "Text fusion is not active: the repository only contains a placeholder text encoder."
        )

    train_transform, test_transform = get_train_test_transforms(int(cfg["train"]["img_size"]))
    train_paths, class_to_idx, class_names = gather_split(train_dir)
    test_paths, test_classes, _ = gather_split(test_dir)
    if test_classes != class_to_idx:
        raise ValueError("Training and test class folders do not match")

    captions = read_captions(resolve_path(paths.get("captions_csv", ""), base))
    train_ds = BrainTumorDataset(
        train_paths, class_to_idx, train_transform, use_text, captions, project_root=base
    )
    test_ds = BrainTumorDataset(
        test_paths, class_to_idx, test_transform, use_text, captions, project_root=base
    )
    batch_size = int(cfg["train"]["batch_size"])
    workers = _auto_workers(cfg["train"].get("num_workers"))
    loader_kwargs = {"batch_size": batch_size, "num_workers": workers, "pin_memory": True}
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
        class_names,
    )


def make_clean_loaders_from_cfg(
    cfg: dict[str, Any], project_root: str | Path | None = None
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/validation/final-test loaders from an exact-group manifest.

    The manifest is the authority for membership. Raw class folders remain
    untouched, and the original ``train``/``test`` names are retained only as
    provenance columns in the manifest.
    """

    base = Path(project_root or Path.cwd()).resolve()
    paths = cfg["paths"]
    manifest_path = resolve_path(paths["split_manifest"], base)
    raw_root = resolve_path(paths.get("raw_data_dir", "data/raw"), base)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}. Run create_leakage_free_split first.")
    if bool(cfg.get("use_text", False)):
        raise NotImplementedError("The clean pipeline is image-only because the text encoder is not implemented.")

    rows: dict[str, list[Path]] = defaultdict(list)
    classes = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = raw_root / row["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Manifest image is missing: {path}")
            if path.parent.name != row["class_name"]:
                raise ValueError(f"Manifest class does not match image path: {row['path']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Manifest hash mismatch for {path}")
            rows[row["split"]].append(path)
            classes.add(row["class_name"])
    class_names = sorted(classes)
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    if set(rows) != set(SPLIT_NAMES):
        raise ValueError(f"Manifest must contain {SPLIT_NAMES}, found {sorted(rows)}")

    train_transform, test_transform = get_train_test_transforms(int(cfg["train"]["img_size"]))
    captions = read_captions(resolve_path(paths.get("captions_csv", ""), base))
    datasets = {
        "train": BrainTumorDataset(rows["train"], class_to_idx, train_transform, captions_map=captions, project_root=base),
        "validation": BrainTumorDataset(rows["validation"], class_to_idx, test_transform, captions_map=captions, project_root=base),
        "test": BrainTumorDataset(rows["test"], class_to_idx, test_transform, captions_map=captions, project_root=base),
    }
    batch_size = int(cfg["train"]["batch_size"])
    workers = _auto_workers(cfg["train"].get("num_workers"))
    loader_kwargs = {"batch_size": batch_size, "num_workers": workers, "pin_memory": True}
    return (
        DataLoader(datasets["train"], shuffle=True, **loader_kwargs),
        DataLoader(datasets["validation"], shuffle=False, **loader_kwargs),
        DataLoader(datasets["test"], shuffle=False, **loader_kwargs),
        class_names,
    )


SPLIT_NAMES = ("train", "validation", "test")
