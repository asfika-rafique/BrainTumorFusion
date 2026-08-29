"""Training and evaluation loops for the configured image classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.datasets import make_clean_loaders_from_cfg, make_loaders_from_cfg
from ..data.splitting import build_split_manifest
from ..evaluation.metrics import classification_metrics
from ..models.fusion_model import FusionNet
from ..models.image_encoder import ImageEncoder
from ..utils import config_root, load_config, resolve_path, seed_everything


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    """Construct the configured image model without changing the research head."""

    model_cfg = cfg["model"]
    encoder = ImageEncoder(
        name=model_cfg.get("image_encoder", "resnet18"),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
    return FusionNet(
        image_encoder=encoder,
        img_out_dim=int(model_cfg.get("img_out_dim", encoder.out_dim)),
        txt_out_dim=int(model_cfg.get("txt_out_dim", 768)),
        fusion_hidden=int(model_cfg.get("fusion_hidden", 512)),
        num_classes=num_classes,
        use_text=bool(cfg.get("use_text", False)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )


def _criterion(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    weights = cfg.get("train", {}).get("class_weights")
    tensor = torch.tensor(weights, dtype=torch.float32, device=device) if weights else None
    return nn.CrossEntropyLoss(weight=tensor)


def _clean_training_config(
    cfg: dict[str, Any], train_loader: DataLoader, class_names: list[str]
) -> tuple[dict[str, Any], dict[str, int], list[float] | None]:
    """Derive optional class weights from the clean training dataset only."""

    train_cfg = dict(cfg.get("train", {}))
    dataset = train_loader.dataset
    class_to_idx = getattr(dataset, "class_to_idx", {})
    counts = Counter(class_to_idx[path.parent.name] for path in getattr(dataset, "image_paths", []))
    class_counts = {name: int(counts.get(index, 0)) for index, name in enumerate(class_names)}
    if not bool(train_cfg.get("derive_class_weights", False)):
        return cfg, class_counts, train_cfg.get("class_weights")
    if any(count <= 0 for count in class_counts.values()):
        raise ValueError(f"Cannot derive class weights from empty clean training classes: {class_counts}")
    total = sum(class_counts.values())
    weights = [total / (len(class_names) * class_counts[name]) for name in class_names]
    effective = dict(cfg)
    effective["train"] = dict(train_cfg, class_weights=weights)
    return effective, class_counts, weights


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: dict[str, Any] | None = None):
    """Return mean loss and accuracy over a loader."""

    model.eval()
    criterion = _criterion(cfg or {}, device)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate_with_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate once and return loss plus dependency-light classification metrics."""

    model.eval()
    criterion = _criterion(cfg or {}, device)
    y_true, y_pred = [], []
    total_loss = 0.0
    total = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def _train_one_epoch(model, loader, optimizer, scaler, device, cfg):
    model.train()
    criterion = _criterion(cfg, device)
    running_loss = 0.0
    use_amp = scaler is not None and device.type == "cuda"
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * labels.size(0)
    return running_loss / max(len(loader.dataset), 1)


def try_engine_train(cfg_path: str, epochs: int | None = None) -> None:
    """Run training from a repository-relative YAML config."""

    path = Path(cfg_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = load_config(path)
    seed_everything(int(cfg.get("seed", 42)))

    requested = str(cfg.get("device", "cuda")).lower()
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    train_loader, test_loader, class_names = make_loaders_from_cfg(cfg, config_root(path))
    model = build_model(cfg, num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    max_epochs = int(cfg["train"].get("epochs", 1) if epochs is None else epochs)
    checkpoint_dir = Path(cfg["paths"].get("ckpt_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        test_loss, test_accuracy = evaluate(model, test_loader, device, cfg)
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}"
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {"model": model.state_dict(), "ep": epoch, "val_acc": float(test_accuracy), "classes": class_names},
                checkpoint_dir / f"best_ep{epoch}_acc{test_accuracy:.3f}.pt",
            )

    torch.save(
        {"model": model.state_dict(), "ep": max_epochs, "val_acc": float(best_accuracy), "classes": class_names},
        checkpoint_dir / "final.pt",
    )


def try_clean_train(cfg_path: str, epochs: int | None = None) -> None:
    """Train with validation-only model selection and one final test pass.

    This is intentionally separate from ``try_engine_train`` so historical
    test-selected experiments and their artifacts remain reproducible as
    history rather than being silently rewritten.
    """

    path = Path(cfg_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = load_config(path)
    seed_everything(int(cfg.get("seed", 42)))
    requested = str(cfg.get("device", "cuda")).lower()
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    manifest_path = resolve_path(cfg["paths"]["split_manifest"], config_root(path))
    if not manifest_path.is_file():
        split_cfg = cfg.get("split", {})
        build_split_manifest(
            resolve_path(cfg["paths"].get("raw_data_dir", "data/raw"), config_root(path)),
            manifest_path,
            seed=int(cfg.get("seed", 42)),
            validation_fraction=float(split_cfg.get("validation_fraction", 0.15)),
            test_fraction=float(split_cfg.get("test_fraction", 0.15)),
        )
        print(f"[split] generated deterministic manifest at {manifest_path}")
    train_loader, validation_loader, test_loader, class_names = make_clean_loaders_from_cfg(cfg, config_root(path))
    cfg, train_class_counts, effective_class_weights = _clean_training_config(cfg, train_loader, class_names)
    model = build_model(cfg, num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    max_epochs = int(cfg["train"].get("epochs", 1) if epochs is None else epochs)
    checkpoint_dir = Path(cfg["paths"].get("clean_ckpt_dir", "outputs/checkpoints/clean"))
    results_dir = Path(cfg["paths"].get("clean_results_dir", "outputs/results/clean"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    repository_root = config_root(path)

    def portable_path(value: Path) -> str:
        try:
            return value.resolve().relative_to(repository_root).as_posix()
        except ValueError:
            return value.as_posix()

    best_validation_accuracy = -1.0
    best_checkpoint = None
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device, cfg)
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_checkpoint = checkpoint_dir / f"best_validation_ep{epoch}_acc{validation_accuracy:.3f}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": float(validation_accuracy),
                    "validation_loss": float(validation_loss),
                    "classes": class_names,
                    "config": portable_path(path),
                    "seed": int(cfg.get("seed", 42)),
                    "train_class_counts": train_class_counts,
                    "class_weights": effective_class_weights,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None,
                    "amp_scaler": scaler.state_dict() if scaler is not None else None,
                },
                best_checkpoint,
            )

    if best_checkpoint is None:
        raise RuntimeError("No validation checkpoint was produced")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_with_metrics(model, test_loader, device, class_names, cfg)
    test_loss = float(test_metrics["loss"])
    test_accuracy = float(test_metrics["accuracy"])
    final_checkpoint = checkpoint_dir / "final_after_validation_selection.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "selected_checkpoint": portable_path(best_checkpoint),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "test_metrics": test_metrics,
            "classes": class_names,
            "config": portable_path(path),
            "seed": int(cfg.get("seed", 42)),
            "train_class_counts": train_class_counts,
            "class_weights": effective_class_weights,
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "amp_scaler": scaler.state_dict() if scaler is not None else None,
        },
        final_checkpoint,
    )
    import json

    (results_dir / "final_test_metrics.json").write_text(
        json.dumps(
            {
                "protocol": "train -> validation model selection -> final test once",
                "config": portable_path(path),
                "manifest": portable_path(Path(cfg["paths"]["split_manifest"])),
                "selected_checkpoint": portable_path(best_checkpoint),
                "final_checkpoint": portable_path(final_checkpoint),
                "seed": int(cfg.get("seed", 42)),
                "train_class_counts": train_class_counts,
                "class_weights": effective_class_weights,
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "metrics": test_metrics,
                "classes": class_names,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[final test] loss={test_loss:.4f} accuracy={test_accuracy:.4f}")
