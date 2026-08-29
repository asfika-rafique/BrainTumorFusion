from pathlib import Path

from brain_tumor_fusion.utils import config_root, resolve_path


def test_resolve_path_keeps_absolute_paths(tmp_path: Path):
    absolute = tmp_path / "data"
    assert resolve_path(absolute, Path("C:/base")) == absolute


def test_resolve_path_resolves_relative_paths(tmp_path: Path):
    assert resolve_path("data/raw", tmp_path) == (tmp_path / "data/raw").resolve()


def test_config_root_points_to_repository(tmp_path: Path):
    config = tmp_path / "configs" / "experiment.yaml"
    assert config_root(config) == tmp_path
