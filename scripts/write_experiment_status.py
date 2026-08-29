"""Write explicit status registries without inventing experiment results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    reason = "NOT RUN: supported PyTorch/torchvision runtime unavailable in the current environment."
    models = ["SimpleCNN baseline", "ResNet baseline", "FusionNet"]
    _write(
        args.output_dir / "results" / "model_comparison.csv",
        ["Model", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1", "Parameters", "Trainable_Parameters", "Status", "Reason"],
        [{"Model": model, "Status": "NOT_RUN", "Reason": reason, "Accuracy": "", "Macro_Precision": "", "Macro_Recall": "", "Macro_F1": "", "Weighted_F1": "", "Parameters": "", "Trainable_Parameters": ""} for model in models],
    )
    variants = ["Full FusionNet", "Without projection/fusion", "Without dropout", "Without normalization", "Without augmentation", "Without class weighting"]
    _write(
        args.output_dir / "results" / "ablation_results.csv",
        ["Variant", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Delta_Accuracy", "Delta_Macro_F1", "Status", "Reason"],
        [{"Variant": variant, "Status": "NOT_RUN", "Reason": reason, "Accuracy": "", "Macro_Precision": "", "Macro_Recall": "", "Macro_F1": "", "Delta_Accuracy": "", "Delta_Macro_F1": ""} for variant in variants],
    )
    _write(
        args.output_dir / "experiment_registry.csv",
        ["Experiment_ID", "Model", "Dataset", "Split", "Seed", "Config", "Checkpoint", "Best_Epoch", "Validation_Metric", "Test_Accuracy", "Macro_F1", "Status"],
        [
            {"Experiment_ID": "clean_resnet18_image_only", "Model": "FusionNet image-only", "Dataset": "local data/raw", "Split": "leakage_free_split.csv", "Seed": "42", "Config": "configs/clean_resnet18_image_only.yaml", "Checkpoint": "", "Best_Epoch": "", "Validation_Metric": "", "Test_Accuracy": "", "Macro_F1": "", "Status": "NOT_RUN"},
            {"Experiment_ID": "historical_resnet50", "Model": "Historical image-only experiment", "Dataset": "local historical split", "Split": "legacy test-selected", "Seed": "42", "Config": "configs/resnet50_image_only.yaml", "Checkpoint": "outputs/checkpoints/best_ep18_acc0.830.pt", "Best_Epoch": "18", "Validation_Metric": "UNVERIFIED", "Test_Accuracy": "UNVERIFIED", "Macro_F1": "UNVERIFIED", "Status": "UNVERIFIED"},
        ],
    )
    print(f"Wrote explicit experiment status files under {args.output_dir}")


if __name__ == "__main__":
    main()
