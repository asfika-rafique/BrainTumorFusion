"""Evaluate a selected clean-pipeline checkpoint on the final test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts._common import ROOT  # noqa: F401  # adds src/ for repository execution
from brain_tumor_fusion.data.datasets import make_clean_loaders_from_cfg
from brain_tumor_fusion.inference.predictor import load_fusion_model
from brain_tumor_fusion.training.engine import _clean_training_config, evaluate_with_metrics
from brain_tumor_fusion.utils import config_root, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/results/clean_test_metrics.json"))
    args = parser.parse_args()
    with args.cfg.open("r", encoding="utf-8") as handle:
        cfg = resolve_config_paths(yaml.safe_load(handle) or {}, config_root(args.cfg.resolve()))
    requested = str(cfg.get("device", "cuda")).lower()
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    train_loader, _, test_loader, class_names = make_clean_loaders_from_cfg(cfg, config_root(args.cfg.resolve()))
    cfg, train_class_counts, effective_class_weights = _clean_training_config(cfg, train_loader, class_names)
    model = load_fusion_model(cfg, args.ckpt, device)
    metrics = evaluate_with_metrics(model, test_loader, device, class_names, cfg)
    result = {
        "protocol": "final test evaluation after validation-based checkpoint selection",
        "config": args.cfg.as_posix(),
        "checkpoint": args.ckpt.as_posix(),
        "device": str(device),
        "classes": class_names,
        "train_class_counts": train_class_counts,
        "class_weights": effective_class_weights,
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
