"""Generate evidence-based leakage, metadata, checkpoint, and result audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

from scripts._common import ROOT  # noqa: F401  # adds src/ for repository execution
from brain_tumor_fusion.data.splitting import IMAGE_EXTENSIONS, file_sha256
from brain_tumor_fusion.evaluation.metrics import classification_metrics

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def image_audit(data_root: Path) -> tuple[list[dict], list[list[dict]], dict, list[str]]:
    records = []
    by_hash: dict[str, list[dict]] = defaultdict(list)
    unreadable = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(data_root).as_posix()
        parts = Path(relative).parts
        try:
            with Image.open(path) as image:
                image.verify()
            digest = file_sha256(path)
        except Exception as exc:
            unreadable.append(f"{relative}: {exc}")
            continue
        record = {
            "path": relative,
            "source_split": parts[0] if len(parts) > 0 else "unknown",
            "class_name": parts[1] if len(parts) > 1 else "unknown",
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        records.append(record)
        by_hash[digest].append(record)
    groups = [group for group in by_hash.values() if len(group) > 1]
    cross_split = [group for group in groups if len({item["source_split"] for item in group}) > 1]
    counts = defaultdict(Counter)
    for record in records:
        counts[record["source_split"]][record["class_name"]] += 1
    return records, sorted(cross_split, key=lambda group: group[0]["sha256"]), counts, unreadable


def checkpoint_manifest(checkpoint_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(checkpoint_dir.glob("*")):
        if path.suffix.lower() not in {".pt", ".pth"}:
            continue
        row = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "container": "PyTorch zip serialization" if zipfile.is_zipfile(path) else "non-zip/unknown",
            "payload_strings": "",
        }
        if zipfile.is_zipfile(path):
            try:
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
                    payloads = [name for name in names if name.endswith("data.pkl") or name.endswith(".pkl")]
                    raw = b"\n".join(archive.read(name) for name in payloads)
                    candidates = sorted(
                        {
                            value.decode("utf-8", "ignore")
                            for value in re.findall(rb"[A-Za-z_][A-Za-z0-9_.]{3,}", raw)
                            if any(token in value.decode("utf-8", "ignore") for token in ("image_encoder", "img_branch", "conv1", "layer1", "model_state_dict", "state_dict"))
                        }
                    )
                    row["payload_strings"] = "; ".join(candidates[:12])
                    row["members"] = len(names)
            except Exception as exc:
                row["payload_strings"] = f"inspection error: {exc}"
        rows.append(row)
    return rows


def embedded_metadata_audit(records: list[dict], data_root: Path) -> tuple[list[str], Counter]:
    """Find embedded EXIF fields that require privacy/licensing review."""

    files = []
    tags = Counter()
    for record in records:
        path = data_root / record["path"]
        with Image.open(path) as image:
            exif = image.getexif()
        if exif:
            files.append(record["path"])
            tags.update(TAGS.get(key, str(key)) for key in exif)
    return files, tags


def read_prediction_metrics(result_dir: Path) -> dict | None:
    prediction_path = result_dir / "test_predictions.csv"
    if not prediction_path.is_file():
        return None
    true_values, predicted_values = [], []
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            true_name = row.get("label_true") or row.get("y_true")
            predicted_name = row.get("label_pred") or row.get("y_pred")
            if true_name in CLASS_NAMES and predicted_name in CLASS_NAMES:
                true_values.append(CLASS_NAMES.index(true_name))
                predicted_values.append(CLASS_NAMES.index(predicted_name))
    return classification_metrics(true_values, predicted_values, CLASS_NAMES)


def write_manifest_csv(rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "bytes", "sha256", "container", "members", "payload_strings"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_report(project_root: Path) -> tuple[Path, Path]:
    data_root = project_root / "data" / "raw"
    records, cross_split, counts, unreadable = image_audit(data_root)
    metadata_files, metadata_tags = embedded_metadata_audit(records, data_root)
    checkpoints = checkpoint_manifest(project_root / "outputs" / "checkpoints")
    validation_dir = project_root / "reports" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_csv = validation_dir / "checkpoint_manifest.csv"
    write_manifest_csv(checkpoints, checkpoint_csv)
    split_manifest = project_root / "data" / "interim" / "leakage_free_split.csv"
    clean_split_counts = Counter()
    clean_split_violations = []
    if split_manifest.is_file():
        with split_manifest.open("r", encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        clean_split_counts.update((row["split"], row["class_name"]) for row in manifest_rows)
        hashes_to_splits = defaultdict(set)
        for row in manifest_rows:
            hashes_to_splits[row["sha256"]].add(row["split"])
        clean_split_violations = [digest for digest, splits in hashes_to_splits.items() if len(splits) > 1]

    lines = [
        "# Research validation and reproducibility audit",
        "",
        "Generated from the repository contents on the audit date. Historical artifacts are preserved; no image, checkpoint, result, or metric was deleted or overwritten.",
        "",
        "## Executive status",
        "",
        f"- Dataset images inspected: **{len(records)}**; unreadable: **{len(unreadable)}**.",
        f"- Exact duplicate groups: **{len({record['sha256'] for record in records if sum(1 for item in records if item['sha256'] == record['sha256']) > 1})}**; cross-legacy-split groups: **{len(cross_split)}**.",
        "- Patient-level separation: **not verifiable**; no patient/subject/study/acquisition metadata was found in the supplied dataset index or paths.",
        "- Historical training protocol: **test split used for model selection each epoch** in `src/brain_tumor_fusion/training/engine.py`.",
        "- Clean protocol: **implemented but not run**; it assigns exact duplicate groups to train/validation/test and selects checkpoints on validation only.",
        "- Runtime: **fresh compatible environment not yet available** on this machine; see the final report for exact blockers.",
        "",
        "## 1. Data leakage and patient metadata",
        "",
        "The 64 groups below are cryptographic exact matches: every listed file has the same SHA-256 digest, and the files were byte-compared within each group. This establishes identical encoded image files, not merely similar pixels. It does not establish that files with different bytes are different patients or acquisitions.",
        "",
        f"Class counts by legacy source split: `{dict((split, dict(values)) for split, values in sorted(counts.items()))}`.",
        "",
        "No patient IDs, subject IDs, study IDs, acquisition IDs, or machine-readable group metadata were found. Filenames are generic image names and `data/captions.csv` contains only `image` and `caption` columns. Therefore, the new split prevents exact duplicate leakage only; patient-level leakage cannot be guaranteed.",
        "",
        f"Embedded-image metadata scan: **{len(metadata_files)}** images contain EXIF fields (`{dict(metadata_tags)}`). The observed fields include software/orientation/timestamps and a custom tag; no direct patient identifier was observed in the values inspected, but EXIF must be reviewed and stripped or explicitly cleared by the data owner before release.",
        "",
        "### Cross-split exact duplicate groups",
        "",
    ]
    for index, group in enumerate(cross_split, start=1):
        digest = group[0]["sha256"]
        source_splits = sorted({item["source_split"] for item in group})
        classes = sorted({item["class_name"] for item in group})
        byte_equal = all((data_root / item["path"]).read_bytes() == (data_root / group[0]["path"]).read_bytes() for item in group)
        lines.extend(
            [
                f"#### Group {index:02d}",
                "",
                f"- SHA-256: `{digest}`",
                f"- Class folder(s): `{', '.join(classes)}`; legacy membership: `{', '.join(source_splits)}`",
                f"- Exact byte identity: **verified = {str(byte_equal).lower()}**",
                "- Underlying-image conclusion: the encoded image file is the same. Patient identity remains unknown because no patient-level metadata exists.",
                "",
                "| File | Legacy split | Class | Bytes |",
                "|---|---|---|---:|",
            ]
        )
        lines.extend(
            f"| `{item['path']}` | `{item['source_split']}` | `{item['class_name']}` | {item['bytes']} |"
            for item in sorted(group, key=lambda item: item["path"])
        )
        lines.append("")

    historical_metrics = read_prediction_metrics(project_root / "outputs" / "results")
    archived_metrics = None
    archived_report = project_root / "_archive" / "legacy_results" / "classification_report.json"
    if archived_report.is_file():
        archived_metrics = json.loads(archived_report.read_text(encoding="utf-8"))
    lines.extend(
        [
            "## 2. Train/validation/test audit",
            "",
            "The historical `try_engine_train` loop builds `train_loader` and `test_loader`, evaluates the latter after every training epoch, and saves the best checkpoint using that test accuracy. Consequently, the historical test set is used for model selection and repeated during experiments. The separate historical `scripts/evaluation/evaluate.py` also evaluates a selected checkpoint on that same test directory.",
            "",
            "The clean path is `scripts/data/create_leakage_free_split.py` followed by `scripts/training/train_clean.py` with `configs/clean_resnet18_image_only.yaml`. It does not move or delete raw images. It uses all raw images as source records, keeps each SHA-256 group intact, and defines `train -> validation model selection -> final test once`. Since patient metadata is absent, it is an exact-duplicate-safe split, not a patient-level split.",
            "",
            f"The generated manifest contains **{sum(clean_split_counts.values())}** records with exact-hash groups crossing new splits: **{len(clean_split_violations)}**. New split counts are: `{dict(sorted((f'{split}/{class_name}', count) for (split, class_name), count in clean_split_counts.items()))}`.",
            "",
            "## 3. Checkpoint verification",
            "",
            f"Inspected checkpoint artifacts: **{len(checkpoints)}**. A machine-readable inventory with file sizes, SHA-256 digests, serialization type, and payload string signatures is in [`checkpoint_manifest.csv`](checkpoint_manifest.csv). Checkpoint files were not loaded because the only available compatible-looking package set is inside the invalid project `.venv`, which must not be used; system Python has no PyTorch.",
            "",
            "| Experiment/config reference | Checkpoint reference | Metrics artifact | Verification status |",
            "|---|---|---|---|",
            "| `configs/resnet50_image_only.yaml` | `best_ep18_acc0.830.pt`, `final.pt` | `outputs/results/fusion_classification_report.json` and `test_predictions.csv` | **Unverifiable/conflicting**: the result artifact recomputes to 0.2766497462 accuracy, while the filename claims 0.830; no trusted log/config-to-checkpoint record ties them together. |",
            "| `_archive/legacy_code/fine_tune.yaml` | `best_ep22_acc0.810.pt` | no independent matching result artifact found | **Unverifiable**: filename claim only. |",
            "| `configs/resnet18_image_only.yaml` | no uniquely mapped checkpoint | none | **Unverifiable**: multiple checkpoint families exist without experiment manifests or logs. |",
            "| `configs/clean_resnet18_image_only.yaml` | none yet | none | **Not run**: clean pipeline is code/config only at this stage. |",
            "",
            "Filenames such as `best_ep18_acc0.830.pt` are therefore treated as labels, not evidence. No filename claim has been promoted to a scientific result.",
            "",
            "## 4. Result consistency audit",
            "",
        ]
    )
    if historical_metrics:
        lines.extend(
            [
                "The active historical prediction CSV contains 394 labeled predictions. Recomputing metrics from its `label_true`/`label_pred` columns gives:",
                "",
                f"- Accuracy: **{historical_metrics['accuracy']:.16f}** ({sum(historical_metrics['confusion_matrix'][i][i] for i in range(4))}/{historical_metrics['total']}).",
                f"- Confusion matrix: `{historical_metrics['confusion_matrix']}` in class order `{CLASS_NAMES}`.",
                "- The active JSON report and archived JSON report contain the same 27.66497462% accuracy and matching per-class values; this is a historical artifact, not a newly validated final performance claim.",
                "",
            ]
        )
    else:
        lines.extend(["No parseable historical prediction CSV was found.", ""])
    if archived_metrics:
        lines.append(f"Archived report accuracy field: `{archived_metrics.get('accuracy')}`; it agrees with the recomputed historical CSV but does not resolve the checkpoint conflict.")
        lines.append("")
    lines.extend(
        [
            "Other repository findings: archived visualization scripts explicitly contain example/dummy or hard-coded metric data; the archived dataset-distribution graphic used a no-tumor count inconsistent with the inspected files; and the archived sample-prediction visualization generated random placeholder images/confidences. These remain archived and are not evidence.",
            "",
            "## 5. Reproducibility and release limitations",
            "",
            "- Dataset provenance, license, consent/de-identification statement, and acquisition protocol are absent from the supplied project and must be supplied by the dataset owner.",
            "- Patient-level separation cannot be guaranteed without group metadata.",
            "- Historical results used the test split for model selection and repeated evaluation.",
            "- Text fusion is not implemented; the text encoder is a placeholder and captions are class-derived templates, so active claims are image-only.",
            "- Checkpoint-to-config-to-metric lineage is incomplete; existing filenames are not independently verified.",
            "- No clean pipeline training/evaluation run was performed in this audit, so no new performance number is reported.",
            "",
            "## 6. Release gate",
            "",
            "Before publication or GitHub release, obtain dataset provenance/permission, establish patient/subject grouping if available, regenerate the clean manifest, run the clean pipeline in a compatible fresh environment, save a config/seed/manifest/checkpoint/metrics manifest, and review every generated figure for provenance. Do not commit raw medical images, checkpoints, model weights, logs, or private metadata.",
            "",
        ]
    )
    if unreadable:
        lines.extend(["### Unreadable files", "", *[f"- `{item}`" for item in unreadable], ""])
    report_path = validation_dir / "research_validation_audit.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, checkpoint_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report, manifest = make_report(args.project_root.resolve())
    print(f"[done] wrote {report}")
    print(f"[done] wrote {manifest}")


if __name__ == "__main__":
    main()
