"""Create a classification report and confusion matrix from predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from scripts._common import ROOT

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/results/test_predictions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/results"))
    args = parser.parse_args()
    predictions = (ROOT / args.predictions).resolve() if not args.predictions.is_absolute() else args.predictions
    out_dir = (ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    frame = pd.read_csv(predictions)
    required = {"y_true", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{predictions} must contain {sorted(required)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(frame["y_true"], frame["y_pred"], labels=CLASS_NAMES)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "confusion_matrix.csv")
    report = classification_report(
        frame["y_true"], frame["y_pred"], labels=CLASS_NAMES, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    with (out_dir / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[done] reports written to {out_dir}")


if __name__ == "__main__":
    main()
