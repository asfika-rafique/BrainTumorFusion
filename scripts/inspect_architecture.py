"""Describe the active model and write conservative local architecture artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def _config_values(config_path: Path) -> dict[str, float | int | str]:
    """Read the small set of architecture values without requiring PyTorch."""

    values: dict[str, float | int | str] = {
        "image_encoder": "resnet18",
        "pretrained": True,
        "img_out_dim": 512,
        "fusion_hidden": 512,
        "num_classes": 4,
        "dropout": 0.2,
    }
    if not config_path.is_file():
        return values
    section = ""
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.endswith(":") and not line.startswith("-"):
            section = line[:-1]
            continue
        if ":" not in line:
            continue
        key, raw_value = (part.strip() for part in line.split(":", 1))
        if section != "model":
            continue
        value = raw_value.strip().lower()
        if value in {"true", "false"}:
            values[key] = value == "true"
        else:
            try:
                values[key] = int(value)
            except ValueError:
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = raw_value.strip('"\'')
    return values


def _parameter_rows(values: dict[str, float | int | str]) -> list[dict[str, str | int]]:
    encoder_dim = int(values["img_out_dim"])
    hidden = int(values["fusion_hidden"])
    classes = int(values["num_classes"])
    half = hidden // 2
    return [
        {"component": f"{values['image_encoder']} backbone", "parameters": "NOT_MEASURED", "trainable": "NOT_MEASURED", "frozen": "NOT_MEASURED"},
        {"component": f"Image projection Linear {encoder_dim}->{hidden}", "parameters": (encoder_dim + 1) * hidden, "trainable": (encoder_dim + 1) * hidden, "frozen": 0},
        {"component": f"Image projection BatchNorm1d {hidden}", "parameters": 2 * hidden, "trainable": 2 * hidden, "frozen": 0},
        {"component": f"Fusion Linear {hidden}->{hidden}", "parameters": (hidden + 1) * hidden, "trainable": (hidden + 1) * hidden, "frozen": 0},
        {"component": f"Fusion BatchNorm1d {hidden}", "parameters": 2 * hidden, "trainable": 2 * hidden, "frozen": 0},
        {"component": f"Fusion Linear {hidden}->{half}", "parameters": (hidden + 1) * half, "trainable": (hidden + 1) * half, "frozen": 0},
        {"component": f"Fusion BatchNorm1d {half}", "parameters": 2 * half, "trainable": 2 * half, "frozen": 0},
        {"component": f"Classification head Linear {half}->{classes}", "parameters": (half + 1) * classes, "trainable": (half + 1) * classes, "frozen": 0},
    ]


def _write_diagram(path: Path, values: dict[str, float | int | str]) -> None:
    labels = [
        "RGB MRI\n3 x 224 x 224",
        f"{values['image_encoder']} encoder\noutput {values['img_out_dim']}",
        f"Image projection\n{values['img_out_dim']} -> {values['fusion_hidden']}",
        f"Fusion block\n{values['fusion_hidden']} -> {int(values['fusion_hidden']) // 2}",
        f"Classifier\n{int(values['fusion_hidden']) // 2} -> {values['num_classes']} logits",
    ]
    width, height, box_w, box_h, gap = 1400, 280, 220, 110, 50
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    start = (width - (len(labels) * box_w + (len(labels) - 1) * gap)) // 2
    for index, label in enumerate(labels):
        x = start + index * (box_w + gap)
        y = 85
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=14, outline=(31, 78, 121), width=3, fill=(232, 241, 250))
        lines = label.splitlines()
        draw.text((x + 18, y + 28), lines[0], fill="black")
        draw.text((x + 18, y + 58), lines[1], fill="black")
        if index < len(labels) - 1:
            draw.line((x + box_w + 8, y + box_h // 2, x + box_w + gap - 8, y + box_h // 2), fill=(31, 78, 121), width=3)
            draw.polygon([(x + box_w + gap - 8, y + box_h // 2), (x + box_w + gap - 20, y + box_h // 2 - 8), (x + box_w + gap - 20, y + box_h // 2 + 8)], fill=(31, 78, 121))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/clean_resnet18_image_only.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    values = _config_values(args.config)
    rows = _parameter_rows(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "model_parameters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "parameters", "trainable", "frozen"])
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "Active architecture (derived from repository code and configuration)",
        f"Configuration: {args.config.as_posix()}",
        f"Encoder: {values['image_encoder']}; pretrained={values['pretrained']}; output={values['img_out_dim']}",
        f"Image projection: {values['img_out_dim']} -> {values['fusion_hidden']}",
        f"Fusion block: {values['fusion_hidden']} -> {int(values['fusion_hidden']) // 2}",
        f"Classifier: {int(values['fusion_hidden']) // 2} -> {values['num_classes']} logits",
        "Text fusion: NOT ACTIVE (use_text=false; text encoder is a placeholder)",
        "Backbone/runtime parameter totals: NOT MEASURED (PyTorch/torchvision unavailable in the audit environment)",
        "Projection/head counts are exact arithmetic from the configured layer dimensions; see model_parameters.csv.",
    ]
    (args.output_dir / "model_architecture.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    _write_diagram(args.output_dir / "model_architecture.png", values)
    print(f"Wrote architecture artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
