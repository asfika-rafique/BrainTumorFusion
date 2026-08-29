"""Train a configured BrainTumorFusion image model."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from scripts._common import ROOT
from brain_tumor_fusion.training.engine import try_engine_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = (ROOT / args.cfg).resolve() if not args.cfg.is_absolute() else args.cfg
    try_engine_train(str(cfg), args.epochs)


if __name__ == "__main__":
    mp.freeze_support()
    main()
