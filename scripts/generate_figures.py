"""Generate local, evidence-backed dataset figures from audit artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from scripts.audit_dataset import _make_class_distribution, _make_sample_grid, collect_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()
    records, unreadable = collect_records(args.data_root)
    if unreadable:
        raise SystemExit(f"Cannot generate figures with unreadable images: {len(unreadable)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _make_class_distribution(dict(Counter(row["class_name"] for row in records)), args.output_dir / "class_distribution.png")
    _make_sample_grid(records, args.data_root, args.output_dir / "sample_mri_grid.png")
    print(f"Generated local figures in {args.output_dir}")


if __name__ == "__main__":
    main()
