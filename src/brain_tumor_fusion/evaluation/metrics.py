"""Dependency-light classification metrics used by audit reports."""

from __future__ import annotations

from collections.abc import Sequence


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict:
    """Compute accuracy, per-class precision/recall/F1, and a confusion matrix."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    n_classes = len(class_names)
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for actual, predicted in zip(y_true, y_pred):
        if not 0 <= int(actual) < n_classes or not 0 <= int(predicted) < n_classes:
            raise ValueError("labels must be valid class indices")
        matrix[int(actual)][int(predicted)] += 1

    total = len(y_true)
    correct = sum(matrix[index][index] for index in range(n_classes))
    report = {}
    for index, name in enumerate(class_names):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(n_classes)) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": sum(matrix[index])}

    accuracy = correct / total if total else 0.0
    macro = {
        key: sum(values[key] for values in report.values()) / n_classes if n_classes else 0.0
        for key in ("precision", "recall", "f1-score")
    }
    weighted = {
        key: sum(values[key] * values["support"] for values in report.values()) / total if total else 0.0
        for key in ("precision", "recall", "f1-score")
    }
    normalized = [
        [value / sum(row) if sum(row) else 0.0 for value in row]
        for row in matrix
    ]
    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "normalized_confusion_matrix": normalized,
        "per_class": report,
        "macro avg": dict(macro, support=total),
        "weighted avg": dict(weighted, support=total),
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1-score"],
        "weighted_f1": weighted["f1-score"],
        "total": total,
    }
