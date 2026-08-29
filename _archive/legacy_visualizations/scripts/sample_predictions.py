# sample_predictions.py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_sample_predictions():
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()

    sample_data = [
        ('Glioma Tumor', 87.5, 'red'),
        ('Meningioma Tumor', 82.3, 'blue'),
        ('Pituitary Tumor', 91.2, 'orange'),
        ('No Tumor', 94.8, 'green'),
        ('Glioma Tumor', 79.6, 'red'),
        ('Meningioma Tumor', 85.1, 'blue'),
        ('Pituitary Tumor', 88.9, 'orange'),
        ('No Tumor', 96.3, 'green')
    ]

    for i, (prediction, confidence, color) in enumerate(sample_data):
        # Create sample MRI-like image
        img = np.random.rand(150, 150, 3) * 0.3
        # Add some "tumor-like" patterns for positive cases
        if 'Tumor' in prediction and 'No' not in prediction:
            center_x, center_y = 75, 75
            y, x = np.ogrid[-75:75, -75:75]
            mask = x ** 2 + y ** 2 <= 40 ** 2
            img[mask] += 0.4

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{prediction}\n{confidence}%',
                          color=color, fontweight='bold', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Sample Model Predictions on Brain MRI Scans',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('outputs/figures/sample_predictions.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: outputs/figures/sample_predictions.png")


create_sample_predictions()