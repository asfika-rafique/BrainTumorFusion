"""Create a dashboard from a prediction CSV and its classification report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_curve, auc

from scripts._common import ROOT

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/results/test_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/results/fusion_classification_report.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/model_dashboard.png"))
    args = parser.parse_args()
    predictions = (ROOT / args.predictions).resolve() if not args.predictions.is_absolute() else args.predictions
    report_path = (ROOT / args.report).resolve() if not args.report.is_absolute() else args.report
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    frame = pd.read_csv(predictions)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    cm = confusion_matrix(frame["y_true"], frame["y_pred"], labels=CLASS_NAMES)
    f1 = [report.get(name, {}).get("f1-score", np.nan) for name in CLASS_NAMES]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].imshow(cm)
    axes[0, 0].set_title("Confusion matrix")
    axes[0, 0].set_xticks(range(4), CLASS_NAMES, rotation=35, ha="right")
    axes[0, 0].set_yticks(range(4), CLASS_NAMES)
    for row in range(4):
        for column in range(4):
            axes[0, 0].text(column, row, cm[row, column], ha="center", va="center")
    axes[0, 1].bar(CLASS_NAMES, f1)
    axes[0, 1].set_title("Per-class F1")
    axes[0, 1].tick_params(axis="x", rotation=35)

    if "probs_json" in frame.columns:
        probabilities = np.array([json.loads(value) for value in frame["probs_json"]])
        truth = np.array([[label == name for name in CLASS_NAMES] for label in frame["y_true"]]).astype(int)
        fpr, tpr, _ = roc_curve(truth.ravel(), probabilities.ravel())
        axes[1, 0].plot(fpr, tpr, label=f"micro AUC={auc(fpr, tpr):.3f}")
        axes[1, 0].plot([0, 1], [0, 1], "--")
        axes[1, 0].legend()
        axes[1, 0].set_title("Micro ROC")
        precision_score = average_precision_score(truth, probabilities, average="micro")
        axes[1, 1].text(0.5, 0.5, f"micro AP={precision_score:.3f}", ha="center", va="center")
        axes[1, 1].set_title("Micro average precision")
    else:
        axes[1, 0].text(0.5, 0.5, "No probability column available", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "No probability column available", ha="center", va="center")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[done] saved dashboard to {out}")


if __name__ == "__main__":
    main()
