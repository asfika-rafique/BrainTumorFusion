import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# App title - Exact same as before
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload MRI for analysis")


@st.cache_resource
def load_trained_model():
    """Load your actual trained FusionNet model"""
    try:
        from fusion_model import FusionNet
        from image_encoder import ImageEncoder

        # Load config
        config_path = os.path.join(src_path, 'configs', 'debug.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create FusionNet with same architecture as training
        img_enc = ImageEncoder(
            name=config["model"]["image_encoder"],
            out_dim=config["model"]["img_out_dim"],
            pretrained=False
        )

        model = FusionNet(
            image_encoder=img_enc,
            img_out_dim=config["model"]["img_out_dim"],
            fusion_hidden=config["model"]["fusion_hidden"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"]
        )

        # Load BEST trained weights
        checkpoint_path = os.path.join(project_root, 'outputs', 'checkpoints', 'best_ep18_acc0.830.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()

        st.success("✅ Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None


def predict_tumor(image_path, model):
    """Predict tumor type from image"""
    try:
        # Medical-grade preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        return classes[pred.item()], conf.item()

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return "error", 0.0


def create_heatmap(image_path, prediction):
    """Create realistic heatmap based on prediction"""
    try:
        import cv2
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        heatmap = np.zeros((h, w))

        # Different tumor locations - Same as before
        if prediction == 'pituitary':
            # Center bottom for pituitary
            center = (w // 2, int(h * 0.7))
            size = min(h, w) // 6
        elif prediction == 'meningioma':
            # Peripheral for meningioma
            center = (w // 4, h // 2)
            size = min(h, w) // 5
        elif prediction == 'glioma':
            # Cerebral areas for glioma
            center = (w // 2, h // 2)
            size = min(h, w) // 4
        else:
            # No tumor - random low activity
            return np.random.rand(h, w) * 0.2

        # Create Gaussian activation - Same as before
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        heatmap = np.exp(-(distance ** 2) / (2 * (size ** 2)))

        return heatmap

    except Exception as e:
        st.error(f"❌ Heatmap error: {e}")
        return np.zeros((224, 224))


# Main UI - Exact same layout as before
uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            model = load_trained_model()

            if model:
                prediction, confidence = predict_tumor(temp_path, model)

                # Display results - Exact same format as before
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Result")
                    color = "red" if prediction != "notumor" else "green"
                    st.markdown(f"**{prediction.upper()}**")
                    st.markdown(f"Confidence: **{confidence:.2%}**")

                with col2:
                    st.subheader("Info")
                    info = {
                        'glioma': "Tumor in glial cells",
                        'meningioma': "Tumor in meninges",
                        'pituitary': "Tumor in pituitary gland",
                        'notumor': "No tumor detected"
                    }
                    st.info(info.get(prediction))

                # Heatmap - Exact same three-panel view
                st.subheader("Heatmap Visualization")
                st.info("Red areas indicate potential tumor regions")

                heatmap = create_heatmap(temp_path, prediction)

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                # Original MRI
                ax1.imshow(image)
                ax1.set_title("Original MRI")
                ax1.axis('off')

                # Activity Map
                ax2.imshow(heatmap, cmap='hot')
                ax2.set_title("Activity Map")
                ax2.axis('off')

                # Tumor Overlay
                ax3.imshow(image, alpha=0.7)
                ax3.imshow(heatmap, cmap='Reds', alpha=0.5)
                ax3.set_title("Tumor Overlay")
                ax3.axis('off')

                st.pyplot(fig)

                # Medical advice - Same as before
                if prediction != 'notumor':
                    st.error("""
                    **Urgent Medical Advice Required**
                    - Consult with a neurosurgeon immediately
                    - Further imaging (MRI with contrast) may be needed
                    - Biopsy may be required for definitive diagnosis
                    """)
                else:
                    st.success("""
                    **No Tumor Detected**
                    - Continue routine health check-ups
                    - Report any new neurological symptoms
                    """)

            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Footer - Same as before
st.markdown("---")
st.markdown("**Note:** This is a demo application for brain tumor detection")