from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from brain_tumor_fusion.data.datasets import read_captions


def test_caption_index_preserves_relative_paths(tmp_path: Path):
    csv_path = tmp_path / "captions.csv"
    csv_path.write_text(
        "image,caption\n"
        "data/raw/train/glioma_tumor/image.jpg,glioma tumor MRI\n"
        "data/raw/test/glioma_tumor/image.jpg,glioma tumor MRI\n",
        encoding="utf-8",
    )
    captions = read_captions(csv_path)
    assert captions["data/raw/train/glioma_tumor/image.jpg"] == "glioma tumor MRI"
    assert captions["data/raw/test/glioma_tumor/image.jpg"] == "glioma tumor MRI"
    assert "__basename__/image.jpg" not in captions
