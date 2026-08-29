from pathlib import Path

import pytest
from PIL import Image

from brain_tumor_fusion.data.splitting import assign_duplicate_groups, file_sha256
from brain_tumor_fusion.evaluation.metrics import classification_metrics
from scripts.data.sanitize_release_images import sanitize_tree


def test_metrics_match_confusion_counts() -> None:
    result = classification_metrics([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], ["a", "b", "c"])
    assert result["accuracy"] == 3 / 5
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert result["per_class"]["c"]["recall"] == 0.0
    assert result["macro_f1"] == pytest.approx((0.5 + (2 * (2 / 3) * 1 / ((2 / 3) + 1)) + 0.0) / 3)
    assert result["normalized_confusion_matrix"][0] == [0.5, 0.5, 0.0]


def test_duplicate_group_assignment_keeps_hash_group_together(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"same image bytes")
    digest = file_sha256(image)
    records = [
        {"path": "train/a/one.jpg", "source_split": "train", "class_name": "a", "sha256": digest, "bytes": 16},
        {"path": "test/a/two.jpg", "source_split": "test", "class_name": "a", "sha256": digest, "bytes": 16},
        {"path": "train/a/three.jpg", "source_split": "train", "class_name": "a", "sha256": "other", "bytes": 8},
    ]
    assigned = assign_duplicate_groups(records, seed=7, validation_fraction=0.2, test_fraction=0.2)
    duplicate_splits = {item["split"] for item in assigned if item["sha256"] == digest}
    assert len(duplicate_splits) == 1


def test_release_sanitizer_does_not_modify_input(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    source = input_root / "train" / "a" / "image.jpg"
    source.parent.mkdir(parents=True)
    exif = Image.Exif()
    exif[306] = "2026:01:01 00:00:00"
    Image.new("RGB", (8, 8), "white").save(source, exif=exif)
    original_bytes = source.read_bytes()

    copied, metadata_removed = sanitize_tree(input_root, tmp_path / "output")

    assert (copied, metadata_removed) == (1, 1)
    assert source.read_bytes() == original_bytes
    with Image.open(tmp_path / "output" / "train" / "a" / "image.jpg") as sanitized:
        assert not sanitized.getexif()
