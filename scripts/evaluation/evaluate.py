"""Evaluate one checkpoint on the configured test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from scripts._common import ROOT
from brain_tumor_fusion.data.datasets import make_loaders_from_cfg
from brain_tumor_fusion.inference.predictor import load_fusion_model
from brain_tumor_fusion.training.engine import evaluate
from brain_tumor_fusion.utils import config_root, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    cfg_path = (ROOT / args.cfg).resolve() if not args.cfg.is_absolute() else args.cfg
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = resolve_config_paths(yaml.safe_load(handle) or {}, config_root(cfg_path))
    device = torch.device("cuda" if cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    _, loader, _ = make_loaders_from_cfg(cfg, config_root(cfg_path))
    ckpt = (ROOT / args.ckpt).resolve() if not args.ckpt.is_absolute() else args.ckpt
    model = load_fusion_model(cfg, ckpt, device)
    loss, accuracy = evaluate(model, loader, device, cfg)
    print(f"test_loss={loss:.6f}\ntest_accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
