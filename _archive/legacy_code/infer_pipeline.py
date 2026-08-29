# src/infer_pipeline.py
# GPU-aware inference + Grad-CAM with a ResNet18 backbone wrapper
import os, sys, pathlib
from typing import Tuple, Optional
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(SRC)  not in sys.path: sys.path.insert(0, str(SRC))

import torch
torch.backends.cudnn.benchmark = True
torch.set_num_threads(2)

# ---------------- config ----------------
CKPT = ROOT / "outputs" / "checkpoints" / "final.pt"   # adjust if needed
CLASS_NAMES = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
IMG_SIZE = 224
IMAGE_BACKBONE = "resnet18"
IMG_OUT_DIM = 512
TXT_MODEL = "bert-base-uncased"
TXT_OUT_DIM = 768
FUSION_HID = 256
NUM_CLASSES = len(CLASS_NAMES)

from src.transforms import build_eval_transform
from src.fusion_model import FusionNet as Model
from src.gradcam import GradCAM, overlay_heatmap

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[torch.nn.Module] = None
_transform = build_eval_transform(IMG_SIZE)
_cam: Optional[GradCAM] = None

# --------- ResNet18 encoder wrapper (has .backbone.layer4) ----------
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNet18Encoder(nn.Module):
    """
    Provides:
      - self.backbone: torchvision ResNet18 with named layers (conv1..layer4)
      - forward(x): returns projected embedding of size IMG_OUT_DIM
    Grad-CAM will hook self.backbone.layer4
    """
    def __init__(self, out_dim: int = 512, pretrained: bool = False):
        super().__init__()
        # torchvision 0.15+: weights kw; older uses pretrained
        try:
            m = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        except Exception:
            m = resnet18(weights=None)
        # keep full structure so layer4 exists
        self.backbone = m
        feat_dim = m.fc.in_features
        self.backbone.fc = nn.Identity()  # we won't use it
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        # manually run to capture layer4 features before avgpool
        m = self.backbone
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); feat = m.layer4(x)  # [B,512,H/32,W/32]
        pooled = F.adaptive_avg_pool2d(feat, (1,1)).flatten(1)                # [B,512]
        emb = self.proj(pooled)                                               # [B,out_dim]
        return emb

def _build_image_encoder():
    # Your repo had build_image_encoder; here we force a safe wrapper
    return ResNet18Encoder(out_dim=IMG_OUT_DIM, pretrained=False)

def _safe_forward_image_only(m, x: torch.Tensor):
    """Try both signatures depending on FusionNet implementation."""
    try:
        return m(image=x, use_text=False)
    except TypeError:
        try:
            return m(x)
        except TypeError:
            # last resort for strict signatures
            return m(image=x, use_text=False, input_ids=None, attention_mask=None)

def _load_state(m: torch.nn.Module, ckpt_path: pathlib.Path):
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = m.load_state_dict(state, strict=False)
    if unexpected: print("[infer] unexpected keys:", unexpected)
    if missing:    print("[infer] missing keys:", missing)

def get_model() -> torch.nn.Module:
    global _model, _cam
    if _model is not None:
        return _model
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    image_encoder = _build_image_encoder()
    model = Model(
        image_encoder=image_encoder,
        img_out_dim=IMG_OUT_DIM,
        text_encoder_name=TXT_MODEL,
        txt_out_dim=TXT_OUT_DIM,
        fusion_hidden=FUSION_HID,
        num_classes=NUM_CLASSES,
    ).to(_device).eval()
    _load_state(model, CKPT)

    # Grad-CAM: hook the last conv block of resnet18 (layer4)
    try:
        _cam = GradCAM(model, target_layer="image_encoder.backbone.layer4")
    except Exception as e:
        print("[infer] GradCAM disabled:", e)
        _cam = None

    _model = model
    return model

def preprocess(pil: Image.Image) -> torch.Tensor:
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    t = _transform(pil).unsqueeze(0).to(_device)  # [1,3,H,W]
    return t

@torch.no_grad()
def predict_top2(pil: Image.Image) -> Tuple[str,float,str,float]:
    m = get_model()
    x = preprocess(pil)
    logits = _safe_forward_image_only(m, x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    idx = np.argsort(-probs)
    a, b = int(idx[0]), int(idx[1])
    return CLASS_NAMES[a], float(probs[a]), CLASS_NAMES[b], float(probs[b])

def make_heatmap(pil: Image.Image, class_name: Optional[str]=None) -> Optional[np.ndarray]:
    if _cam is None:
        return None
    try:
        if class_name is None:
            t1, p1, *_ = predict_top2(pil)
            target = CLASS_NAMES.index(t1)
        else:
            target = CLASS_NAMES.index(class_name)
        return _cam(pil, class_index=target)
    except Exception as e:
        print("[infer] cam error:", e)
        return None

def overlay_heatmap_image(pil: Image.Image, heat: Optional[np.ndarray], alpha=0.45) -> Image.Image:
    if heat is None:
        return pil.convert("RGB")
    return overlay_heatmap(pil.convert("RGB"), heat, alpha=alpha)
