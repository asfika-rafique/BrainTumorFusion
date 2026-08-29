"""Audit the local image-folder dataset and write evidence-backed artifacts.

The generated files are intended for local research review. They contain
relative image paths and aggregate image properties, but no patient IDs are
invented or inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_records(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Collect image properties and unreadable-file diagnostics."""

    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(data_root).as_posix()
        parts = Path(relative).parts
        source_split = parts[0] if len(parts) >= 1 else ""
        class_name = parts[1] if len(parts) >= 2 else ""
        record: dict[str, Any] = {
            "path": relative,
            "source_split": source_split,
            "class_name": class_name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                gray = image.convert("L")
                record.update(
                    {
                        "format": image.format or "",
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                        "channels": len(image.getbands()),
                        "min_intensity": gray.getextrema()[0],
                        "max_intensity": gray.getextrema()[1],
                        "has_exif": bool(image.getexif()),
                    }
                )
        except Exception as exc:  # pragma: no cover - exercised on bad files
            record.update(
                {
                    "format": "",
                    "width": 0,
                    "height": 0,
                    "mode": "",
                    "channels": 0,
                    "min_intensity": "",
                    "max_intensity": "",
                    "has_exif": False,
                }
            )
            unreadable.append({"path": relative, "error": str(exc)})
        records.append(record)
    return records, unreadable


def _split_counts(records: list[dict[str, Any]], split_key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    counter = Counter((str(row[split_key]), str(row["class_name"])) for row in records)
    for (split, class_name), count in sorted(counter.items()):
        counts[split][class_name] = count
    return dict(counts)


def _duplicate_groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["sha256"])].append(row)
    return [group for group in grouped.values() if len(group) > 1]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _make_class_distribution(counts: dict[str, int], path: Path) -> None:
    width, height = 1200, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = "Dataset class distribution (local audit)"
    draw.text((50, 35), title, fill="black")
    maximum = max(counts.values(), default=1)
    chart_left, chart_top, chart_bottom = 100, 120, 600
    slot = max(1, (width - 180) // max(len(counts), 1))
    for index, (label, count) in enumerate(sorted(counts.items())):
        x0 = chart_left + index * slot + 30
        x1 = x0 + min(120, slot - 50)
        y1 = chart_bottom
        y0 = y1 - int((count / maximum) * (chart_bottom - chart_top))
        draw.rectangle((x0, y0, x1, y1), fill=(55, 105, 180))
        draw.text((x0, y1 + 15), label.replace("_", " "), fill="black")
        draw.text((x0, max(chart_top - 25, y0 - 25)), str(count), fill="black")
    draw.line((chart_left, chart_bottom, width - 80, chart_bottom), fill="black", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _make_sample_grid(records: list[dict[str, Any]], data_root: Path, path: Path) -> None:
    """Create a deterministic local-only sample grid without modifying inputs."""

    selected: list[dict[str, Any]] = []
    for class_name in sorted({str(row["class_name"]) for row in records}):
        class_rows = [row for row in records if row["class_name"] == class_name and row["width"] and row["height"]]
        selected.extend(class_rows[:4])
    cell_w, cell_h, columns = 220, 250, 4
    grid = Image.new("RGB", (cell_w * columns, cell_h * ((len(selected) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(grid)
    for index, row in enumerate(selected):
        with Image.open(data_root / row["path"]) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((cell_w - 20, cell_h - 55))
            x = (index % columns) * cell_w + (cell_w - thumb.width) // 2
            y = (index // columns) * cell_h + 8
            grid.paste(thumb, (x, y))
        draw.text(((index % columns) * cell_w + 8, (index // columns) * cell_h + cell_h - 38), str(row["class_name"]), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def audit_dataset(data_root: Path, output_dir: Path, make_figures: bool = True) -> dict[str, Any]:
    records, unreadable = collect_records(data_root)
    groups = _duplicate_groups(records)
    cross_split = [group for group in groups if len({row["source_split"] for row in group}) > 1]
    clean_manifest = data_root.parent / "interim" / "leakage_free_split.csv"
    clean_rows: list[dict[str, Any]] = []
    if clean_manifest.is_file():
        with clean_manifest.open("r", encoding="utf-8", newline="") as handle:
            clean_rows = list(csv.DictReader(handle))
    audit = {
        "data_root": data_root.as_posix(),
        "image_files": len(records),
        "unreadable_files": unreadable,
        "formats": dict(Counter(str(row["format"]) for row in records)),
        "modes": dict(Counter(str(row["mode"]) for row in records)),
        "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in records)),
        "exif_files": sum(bool(row["has_exif"]) for row in records),
        "source_split_counts": _split_counts(records, "source_split"),
        "exact_duplicate_groups": len(groups),
        "exact_duplicate_files": sum(len(group) for group in groups),
        "cross_split_exact_duplicate_groups": len(cross_split),
        "possible_visual_duplicate_detection": "NOT PERFORMED",
        "patient_identifier_detection": "No patient, subject, study, or acquisition identifiers are present in the supplied image-folder structure.",
        "clean_manifest_present": bool(clean_rows),
        "clean_split_counts": _split_counts(clean_rows, "split") if clean_rows else {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    fields = ["path", "source_split", "class_name", "bytes", "sha256", "format", "width", "height", "mode", "channels", "min_intensity", "max_intensity", "has_exif"]
    _write_csv(output_dir / "dataset_audit.csv", records, fields)
    stat_rows = []
    counter = Counter((row["source_split"], row["class_name"]) for row in records)
    for (split, class_name), count in sorted(counter.items()):
        stat_rows.append({"split": split, "class_name": class_name, "images": count, "percentage_of_dataset": count / max(len(records), 1)})
    if clean_rows:
        clean_counter = Counter((row["split"], row["class_name"]) for row in clean_rows)
        for (split, class_name), count in sorted(clean_counter.items()):
            stat_rows.append({"split": f"clean_{split}", "class_name": class_name, "images": count, "percentage_of_dataset": count / max(len(clean_rows), 1)})
    _write_csv(output_dir / "dataset_statistics.csv", stat_rows, ["split", "class_name", "images", "percentage_of_dataset"])
    if make_figures:
        _make_class_distribution(dict(Counter(row["class_name"] for row in records)), output_dir / "class_distribution.png")
        _make_sample_grid(records, data_root, output_dir / "sample_mri_grid.png")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.data_root}")
    result = audit_dataset(args.data_root, args.output_dir, not args.no_figures)
    print(json.dumps({key: result[key] for key in ("image_files", "exif_files", "exact_duplicate_groups", "cross_split_exact_duplicate_groups", "clean_split_counts")}, indent=2))


if __name__ == "__main__":
    main()
