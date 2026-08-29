# model_architecture.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Model components
    components = [
        ("Input MRI\n(224×224×3)", 1, 7, 'lightblue'),
        ("ResNet-50\nBackbone", 2, 6, 'lightgreen'),
        ("Feature Extraction\n2048 features", 3, 5, 'lightyellow'),
        ("Fusion Layers\n512→256", 4, 4, 'lightcoral'),
        ("Classification\n4 Classes", 5, 3, 'lightpink'),
        ("Output Prediction\nGlioma/Meningioma/etc.", 6, 2, 'lightgray')
    ]

    for i, (label, x, height, color) in enumerate(components):
        rect = patches.Rectangle((x, 1), 0.8, height, linewidth=2,
                                 edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + 0.4, 1 + height / 2, label, ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Arrows
    for i in range(len(components) - 1):
        ax.arrow(components[i][1] + 0.8, 1 + components[i][2] / 2,
                 0.4, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')

    ax.set_xlim(0.5, 7)
    ax.set_ylim(0.5, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title('FusionNet Architecture - Brain Tumor Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/model_architecture.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: outputs/figures/model_architecture.png")


create_architecture_diagram()