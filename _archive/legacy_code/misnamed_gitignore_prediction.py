"""Historical prediction helper recovered from the original misnamed .gitignore.

This file is preserved for provenance only. It targets modules and paths that
were not part of the verified active implementation.
"""

import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(__file__))

try:
    from transforms import get_transform
except ImportError:
    from torchvision import transforms

    def get_transform():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def predict(image_path, model, text_input=""):
    """Predict a class with the historical image-only model interface."""

    try:
        image = get_transform()(Image.open(image_path).convert("RGB")).unsqueeze(0)
        image = image.to(next(model.parameters()).device)
        model.eval()
        with torch.no_grad():
            outputs = model.forward_inference(image) if hasattr(model, "forward_inference") else model(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        prediction = class_names[predicted_class.item()]
        confidence_value = confidence.item()
        print(f"Prediction: {prediction}, Confidence: {confidence_value:.4f}")
        return prediction, confidence_value
    except Exception as exc:
        print(f"Prediction error: {exc}")
        return "error", 0.0


if __name__ == "__main__":
    from utils import load_model

    model = load_model()
    prediction, confidence = predict("data/Testing/glioma/image1.jpg", model)
    print(f"Test Result: {prediction} with {confidence:.2f} confidence")
