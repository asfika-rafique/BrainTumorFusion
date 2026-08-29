import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.audit_data_leakage import audit_leakage
from scripts.audit_dataset import audit_dataset


def _write_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=color).save(path, format="JPEG")


def test_dataset_audit_reports_exact_duplicates_and_clean_manifest(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    first = root / "train" / "glioma_tumor" / "one.jpg"
    second = root / "test" / "glioma_tumor" / "two.jpg"
    _write_image(first, "white")
    second.parent.mkdir(parents=True)
    second.write_bytes(first.read_bytes())
    _write_image(root / "train" / "no_tumor" / "three.jpg", "black")

    manifest = tmp_path / "interim" / "leakage_free_split.csv"
    manifest.parent.mkdir()
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class_name", "source_split", "split", "sha256", "bytes"])
        writer.writeheader()
        writer.writerow({"path": "train/glioma_tumor/one.jpg", "class_name": "glioma_tumor", "source_split": "train", "split": "train", "sha256": digest, "bytes": first.stat().st_size})
        writer.writerow({"path": "test/glioma_tumor/two.jpg", "class_name": "glioma_tumor", "source_split": "test", "split": "train", "sha256": digest, "bytes": second.stat().st_size})
        third = root / "train" / "no_tumor" / "three.jpg"
        third_digest = hashlib.sha256(third.read_bytes()).hexdigest()
        writer.writerow({"path": "train/no_tumor/three.jpg", "class_name": "no_tumor", "source_split": "train", "split": "validation", "sha256": third_digest, "bytes": third.stat().st_size})

    result = audit_dataset(root, tmp_path / "artifacts", make_figures=False)
    assert result["image_files"] == 3
    assert result["exact_duplicate_groups"] == 1
    assert result["cross_split_exact_duplicate_groups"] == 1
    assert result["clean_split_counts"]["train"]["glioma_tumor"] == 2

    leakage = audit_leakage(root, manifest)
    assert len(leakage["historical_cross_split_groups"]) == 1
    assert leakage["clean_manifest_cross_split_duplicate_groups"] == {}
    saved = json.loads((tmp_path / "artifacts" / "dataset_audit.json").read_text(encoding="utf-8"))
    assert saved["unreadable_files"] == []
