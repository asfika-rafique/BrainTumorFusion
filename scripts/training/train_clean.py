"""Train the leakage-aware train/validation/final-test pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import ROOT
from brain_tumor_fusion.training.engine import try_clean_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=Path("configs/clean_resnet18_image_only.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = (ROOT / args.cfg).resolve() if not args.cfg.is_absolute() else args.cfg
    try_clean_train(str(cfg), args.epochs)


if __name__ == "__main__":
    main()
