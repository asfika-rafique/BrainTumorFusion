# dataset_distribution.py
import matplotlib.pyplot as plt
import numpy as np


def create_dataset_charts():
    # Dataset distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training set distribution
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    train_counts = [826, 822, 795, 827]  # Example counts

    bars = ax1.bar(classes, train_counts, color=['red', 'blue', 'green', 'orange'])
    ax1.set_title('Training Dataset Distribution')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, train_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 10,
                 f'{count}', ha='center', va='bottom')

    # Class balance pie chart
    ax2.pie(train_counts, labels=classes, autopct='%1.1f%%',
            colors=['red', 'blue', 'green', 'orange'])
    ax2.set_title('Class Distribution (%)')

    plt.tight_layout()
    plt.savefig('outputs/figures/dataset_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: outputs/figures/dataset_distribution.png")


create_dataset_charts()