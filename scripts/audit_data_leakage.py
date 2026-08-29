"""Audit exact-duplicate and split-boundary leakage in the local dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.audit_dataset import IMAGE_EXTENSIONS, _sha256


def audit_leakage(data_root: Path, manifest_path: Path) -> dict:
    records = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative = path.relative_to(data_root).as_posix()
            parts = Path(relative).parts
            records.append({"path": relative, "source_split": parts[0], "class_name": parts[1], "sha256": _sha256(path)})
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_hash[row["sha256"]].append(row)
    historical_cross = [rows for rows in by_hash.values() if len({row["source_split"] for row in rows}) > 1]
    manifest_rows = []
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
    manifest_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        manifest_by_hash[row["sha256"]].add(row["split"])
    clean_cross = {digest: sorted(splits) for digest, splits in manifest_by_hash.items() if len(splits) > 1}
    report = {
        "data_root": data_root.as_posix(),
        "manifest": manifest_path.as_posix(),
        "image_files": len(records),
        "historical_cross_split_exact_duplicate_groups": len(historical_cross),
        "historical_cross_split_groups": historical_cross,
        "clean_manifest_present": bool(manifest_rows),
        "clean_manifest_cross_split_duplicate_groups": clean_cross,
        "patient_level_separation": "NOT GUARANTEED: no patient, subject, study, or acquisition identifiers were supplied.",
        "near_duplicate_detection": "NOT PERFORMED; only exact SHA-256 duplicates were audited.",
        "protocol_assessment": {
            "test_used_for_clean_model_selection": False,
            "validation_used_for_clean_model_selection": True,
            "final_test_evaluated_once_after_selection": True,
            "historical_pipeline_test_selection": True,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/leakage_free_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    report = audit_leakage(args.data_root, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_leakage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = []
    for digest, splits in report["clean_manifest_cross_split_duplicate_groups"].items():
        rows.append({"sha256": digest, "splits": ";".join(splits), "status": "LEAKAGE"})
    with (args.output_dir / "data_leakage_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "splits", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: report[key] for key in ("image_files", "historical_cross_split_exact_duplicate_groups", "clean_manifest_cross_split_duplicate_groups")}, indent=2))


if __name__ == "__main__":
    main()
