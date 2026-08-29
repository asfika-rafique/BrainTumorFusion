from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from brain_tumor_fusion.inference.predictor import load_fusion_model, predict_one
from brain_tumor_fusion.training.engine import build_model


def _config() -> dict:
    return {
        "device": "cpu",
        "use_text": False,
        "model": {
            "image_encoder": "resnet18",
            "pretrained": False,
            "img_out_dim": 512,
            "fusion_hidden": 32,
            "num_classes": 4,
            "dropout": 0.1,
        },
        "train": {"img_size": 224, "batch_size": 2, "class_weights": None},
    }


def test_model_checkpoint_and_inference_contract(tmp_path: Path) -> None:
    cfg = _config()
    model = build_model(cfg, num_classes=4)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "classes": ["a", "b", "c", "d"]}, checkpoint)
    loaded = load_fusion_model(cfg, checkpoint, torch.device("cpu"))
    result = predict_one(loaded, Image.new("RGB", (224, 224), color="black"), torch.device("cpu"), 224)
    assert result["top_name"] in {"glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"}
    assert len(result["probabilities"]) == 4
    assert sum(result["probabilities"]) == pytest.approx(1.0, abs=1e-5)
