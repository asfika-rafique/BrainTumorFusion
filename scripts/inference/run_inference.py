"""Write predictions for a class-folder split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from scripts._common import ROOT
from brain_tumor_fusion.inference.predictor import evaluate_directory, load_fusion_model
from brain_tumor_fusion.utils import config_root, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/results/test_predictions.csv"))
    args = parser.parse_args()
    cfg_path = (ROOT / args.cfg).resolve() if not args.cfg.is_absolute() else args.cfg
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = resolve_config_paths(yaml.safe_load(handle) or {}, config_root(cfg_path))
    device = torch.device("cuda" if cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt = (ROOT / args.ckpt).resolve() if not args.ckpt.is_absolute() else args.ckpt
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    split = args.split or Path(cfg["paths"]["test_dir"])
    split = (ROOT / split).resolve() if not split.is_absolute() else split
    model = load_fusion_model(cfg, ckpt, device)
    evaluate_directory(model, split, device, out, int(cfg["train"].get("img_size", 224)))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
