# scripts/ablation_compare.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Local plot util
from scripts.evaluation.plot_confusion import plot_confusion

# Fixed class order (must match training/inference)
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Inputs (update if your files live elsewhere)
IN: Dict[str, Path] = {
    "img": Path("outputs/results/ablations/preds_img.csv"),
    "fusion": Path("outputs/results/test_predictions.csv"),
}

# Output root
OUT = Path("outputs/results")
OUT.mkdir(parents=True, exist_ok=True)


def _read_preds(csv_path: Path) -> Tuple[pd.Series, pd.Series]:
    """
    Expect a CSV with at least columns: y_true, y_pred
    They may be int-coded (0..3) or string labels.
    """
    df = pd.read_csv(csv_path)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: y_true, y_pred")

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    return y_true, y_pred


def _confusion_df(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    Build a 4x4 confusion matrix DataFrame aligned to CLASS_NAMES.
    Handles both integer-coded and string-coded labels.
    """
    # Decide label space based on dtype/content
    if pd.api.types.is_integer_dtype(y_true) or pd.api.types.is_float_dtype(y_true):
        # integer-coded labels 0..3
        labels_idx = list(range(len(CLASS_NAMES)))
        cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=labels_idx)
        df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    else:
        # string labels
        # map unknowns to a sentinel not in CLASS_NAMES to avoid crash, then clamp
        y_true_str = y_true.astype(str)
        y_pred_str = y_pred.astype(str)
        # Only keep rows that have at least one known class to satisfy sklearn
        known_mask = y_true_str.isin(CLASS_NAMES)
        cm = confusion_matrix(y_true_str[known_mask], y_pred_str[known_mask], labels=CLASS_NAMES)
        df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    return df.astype(int)


def make_reports(tag: str, csv_path: Path) -> None:
    """
    For one tag (img/txt/fusion):
      - confusion_matrix.csv
      - classification_report.json
      - confusion_matrix.png
    """
    y_true, y_pred = _read_preds(csv_path)

    # 1) confusion matrix dataframe
    cm_df = _confusion_df(y_true, y_pred)
    cm_csv = OUT / f"{tag}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv)

    # 2) classification report
    # choose label list for report keys (string names)
    if pd.api.types.is_integer_dtype(y_true):
        target_names = CLASS_NAMES
        labels_for_report = list(range(len(CLASS_NAMES)))
        rep = classification_report(
            y_true.astype(int),
            y_pred.astype(int),
            labels=labels_for_report,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    else:
        rep = classification_report(
            y_true.astype(str),
            y_pred.astype(str),
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

    rep_json = OUT / f"{tag}_classification_report.json"
    with open(rep_json, "w") as f:
        json.dump(rep, f, indent=2)

    # 3) confusion heatmap
    out_png = OUT / f"{tag}_confusion_matrix.png"
    plot_confusion(cm_csv, out_png, title=f"{tag.upper()} Confusion Matrix")

    print(f"[{tag}] saved -> {cm_csv.name}, {rep_json.name}, {out_png.name}")


def main():
    for tag, path in IN.items():
        if path.exists():
            try:
                make_reports(tag, path)
            except Exception as e:
                print(f"[{tag}] FAILED: {e}")
        else:
            print(f"[{tag}] SKIP: {path} not found")


if __name__ == "__main__":
    main()
