"""Small utilities shared by command-line entry points."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root when called from an installed package."""

    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve a config path relative to ``base_dir`` unless it is absolute."""

    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def config_root(config_path: str | Path) -> Path:
    """Return the repository root for a config stored in ``configs/``."""

    return Path(config_path).resolve().parent.parent


def resolve_config_paths(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Copy a config and resolve all entries in its ``paths`` section."""

    resolved = dict(cfg)
    resolved["paths"] = {
        key: str(resolve_path(value, base_dir))
        for key, value in cfg.get("paths", {}).items()
    }
    return resolved


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and resolve its repository-relative paths."""

    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load experiment configurations") from exc
    with path.open("r", encoding="utf-8") as handle:
        return resolve_config_paths(yaml.safe_load(handle) or {}, config_root(path))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch where those libraries are available."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
