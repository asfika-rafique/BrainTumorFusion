# scripts/plot_confusion.py
# Draw a confusion-matrix heatmap from a pd.DataFrame OR a CSV path.

from __future__ import annotations
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fixed class order (must match training/inference)
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def _load_confusion_df(cm_input: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    """
    Accept either:
      - a pandas DataFrame with confusion counts
      - a CSV path (str/Path) pointing to a saved confusion matrix
    Ensures shape is exactly 4x4 in CLASS_NAMES order and collapses duplicates.
    """
    if isinstance(cm_input, (str, Path)):
        cm_input = Path(cm_input)
        df = pd.read_csv(cm_input, index_col=0)
    elif isinstance(cm_input, pd.DataFrame):
        df = cm_input.copy()
    else:
        raise TypeError("cm_input must be a DataFrame or a path to CSV")

    # Collapse any accidental duplicate labels (rows/cols)
    df = df.groupby(df.index, sort=False).sum()
    df = df.T.groupby(df.T.index, sort=False).sum().T

    # Align to exact order and fill missing with 0
    df = df.reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)

    # Ensure int counts
    return df.astype(int)


def plot_confusion(cm_input: Union[pd.DataFrame, str, Path], out_png: Union[str, Path],
                   title: str = "Confusion Matrix") -> None:
    """
    Render a confusion matrix heatmap and save to out_png.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = _load_confusion_df(cm_input)
    cm = df.values

    # Normalized (row-wise) percentages for annotation
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = (cm / row_sums) * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", rotation=270, labelpad=12)

    # Ticks/labels
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Grid-like separation lines (optional, subtle)
    ax.set_xticks(np.arange(-.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CLASS_NAMES), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Cell annotations: count (and %)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]:d}\n({pct[i, j]:.1f}%)",
                va="center", ha="center",
                fontsize=8, color="white" if cm[i, j] > cm.max() * 0.5 else "black"
            )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] saved confusion matrix → {out_png}")


def main():
    # Default CLI behavior: read a standard CSV and write a PNG next to it
    cm_csv = Path("outputs/results/confusion_matrix.csv")
    out_png = Path("outputs/results/confusion_matrix.png")
    plot_confusion(cm_csv, out_png)


if __name__ == "__main__":
    main()
