from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from brain_tumor_fusion.utils import load_config


def test_load_config_resolves_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "experiment.yaml"
    config_path.write_text("paths:\n  data_dir: data/raw\nseed: 42\n", encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg["seed"] == 42
    assert cfg["paths"]["data_dir"] == str((tmp_path / "data/raw").resolve())
