"""Streamlit front end for the reusable inference API.

This is a research demonstration only and is not a medical diagnostic device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_fusion.inference.predictor import generate_heatmap, get_device, load_fusion_model, predict_one
from brain_tumor_fusion.utils import config_root, resolve_config_paths

CFG_PATH = ROOT / "configs" / "resnet50_image_only.yaml"
CKPT_PATH = ROOT / "outputs" / "checkpoints" / "best_ep18_acc0.830.pt"


@st.cache_resource
def load_model():
    with CFG_PATH.open("r", encoding="utf-8") as handle:
        cfg = resolve_config_paths(yaml.safe_load(handle) or {}, config_root(CFG_PATH))
    device = get_device(str(cfg.get("device", "cuda")))
    return cfg, device, load_fusion_model(cfg, CKPT_PATH, device)


st.title("BrainTumorFusion research demo")
st.warning("Research software only. Do not use this output for clinical decisions.")
uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)
    if st.button("Run inference"):
        try:
            cfg, device, model = load_model()
            result = predict_one(model, image, device)
            st.metric("Predicted class", result["top_name"], f"{result['top_conf']:.2%}")
            st.caption(f"Second prediction: {result['second_name']} ({result['second_conf']:.2%})")
            overlay, path = generate_heatmap(model, image, cfg["paths"]["heatmap_dir"], device)
            st.image(overlay, caption=f"Grad-CAM overlay (saved to {path.name})")
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
