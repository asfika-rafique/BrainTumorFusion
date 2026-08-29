"""Create a deterministic split manifest that keeps exact duplicate groups together."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import ROOT  # noqa: F401  # adds src/ for repository execution
from brain_tumor_fusion.data.splitting import build_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/interim/leakage_free_split.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    output = build_split_manifest(args.data_root, args.out, args.seed, args.validation_fraction, args.test_fraction)
    print(f"[done] wrote {output}")


if __name__ == "__main__":
    main()
