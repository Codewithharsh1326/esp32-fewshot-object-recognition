# train_fewshot_embeddings.py
# Optimized for few-shot learning on new objects
# Enhanced with thesis-quality evaluation metrics and visualizations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

# Plotting & metrics imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_curve, auc,
    silhouette_score, top_k_accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
import time

# Set plot style for thesis-quality figures
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 8),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ==================== Custom L2 Normalize Layer ====================
@keras.utils.register_keras_serializable(package="Custom")
class L2NormalizeLayer(layers.Layer):
    """Custom L2 normalization layer (serializable, unlike Lambda)."""
    def call(self, x):
        return tf.nn.l2_normalize(x, axis=1)

    def get_config(self):
        return super().get_config()


# ==================== Configuration ====================
# Resolve project root (parent of Code/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    DETAILS_DIR = os.path.join(PROJECT_ROOT, "details", "traning")
    
    # Image settings (optimized for ESP32 camera)
    IMG_SIZE = 96  # Good balance for ESP32
    
    # Model settings
    BACKBONE = 'MobileNetV2'
    ALPHA = 0.35  # Smaller for ESP32
    EMBEDDING_DIM = 128  # Sweet spot for comparison
    
    # Training
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 30
    EPOCHS_PHASE2 = 20
    LEARNING_RATE = 1e-3
    
    # Supported image extensions
    IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.DETAILS_DIR, exist_ok=True)

config = Config()

# ==================== Data Loading ====================
def load_dataset(data_dir):
    """Load image paths and labels from a directory of class subdirectories."""
    image_paths = []
    labels = []
    class_names = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No class subdirectories found in {data_dir}")
    
    print(f"Found {len(class_dirs)} training classes:")
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_names.append(class_name)
        
        images = []
        for ext in config.IMAGE_EXTENSIONS:
            images.extend(list(class_dir.glob(ext)))
        
        # Validate each image file
        valid_images = []
        for img_path in images:
            try:
                raw = tf.io.read_file(str(img_path))
                tf.image.decode_image(raw, channels=3, expand_animations=False)
                valid_images.append(img_path)
            except Exception as e:
                print(f"  ⚠ Skipping corrupt image: {img_path.name} ({e})")
        
        print(f"  {class_name}: {len(valid_images)} images (skipped {len(images) - len(valid_images)})")
        
        for img_path in valid_images:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    if len(image_paths) == 0:
        raise ValueError("No valid images found in the dataset!")
    
    print(f"\nTotal: {len(image_paths)} valid images across {len(class_names)} classes")
    return np.array(image_paths), np.array(labels), class_names

# ==================== Preprocessing ====================
def preprocess_image(image_path, label, img_size=96, augment=True):
    """Load and preprocess a single image."""
    image = tf.io.read_file(image_path)
    # Use decode_image instead of decode_jpeg to handle PNG, BMP, etc.
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    # Ensure image has a known shape after decode_image
    image.set_shape([None, None, 3])
    
    if augment:
        # Heavy augmentation for generalization
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_hue(image, 0.1)
        
        # Resize first to ensure minimum size, then random crop
        padded_size = int(img_size * 1.3)
        image = tf.image.resize(image, [padded_size, padded_size])
        image = tf.image.random_crop(image, [img_size, img_size, 3])
    else:
        image = tf.image.resize(image, [img_size, img_size])
    
    # Clip pixel values to valid range
    image = tf.clip_by_value(image, 0.0, 255.0)
    
    # MobileNetV2 preprocessing (scales to [-1, 1])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    return image, label

def create_dataset(image_paths, labels, batch_size, img_size, augment=True):
    """Create a tf.data.Dataset from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if augment:
        dataset = dataset.shuffle(len(image_paths), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, img_size, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ==================== Embedding Model ====================
def create_embedding_model(input_shape=(96, 96, 3), embedding_dim=128):
    """
    Create embedding model optimized for few-shot learning.
    Uses ImageNet pre-trained MobileNetV2 for good generalization.
    """
    inputs = keras.Input(shape=input_shape)
    
    # MobileNetV2 backbone with ImageNet weights
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=config.ALPHA
    )
    
    # Freeze bottom layers, fine-tune top layers
    base_model.trainable = True
    # Freeze all but the last 30 layers for Phase 1
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Extract features
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Embedding head
    x = layers.Dense(embedding_dim, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    
    # L2 normalization (critical for cosine similarity)
    # Use custom layer instead of Lambda for proper serialization
    outputs = L2NormalizeLayer(name='l2_normalize')(x)
    
    model = keras.Model(inputs, outputs, name='embedding_model')
    
    return model, base_model

# ==================== Training Model with Classification ====================
def create_training_model(embedding_model, num_classes):
    """Add classification head on top of embedding model for training."""
    inputs = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    embeddings = embedding_model(inputs)
    
    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax', name='classification_head')(embeddings)
    
    model = keras.Model(inputs, outputs, name='training_model')
    return model


# ==================== Thesis-Quality Evaluation & Plotting ====================

def merge_histories(history1, history2):
    """Merge two Keras training histories into one combined dict."""
    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history.get(key, [])
    return combined


def plot_training_history(combined_history, phase1_epochs, save_dir):
    """Plot training & validation accuracy/loss curves with phase boundary."""
    
    # --- Accuracy ---
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(combined_history['accuracy']) + 1)
    
    ax.plot(epochs, combined_history['accuracy'], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=3)
    ax.plot(epochs, combined_history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=3)
    ax.axvline(x=phase1_epochs, color='gray', linestyle='--', linewidth=1.5, label=f'Phase 1→2 Boundary (Epoch {phase1_epochs})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training & Validation Accuracy Over Epochs')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    
    # Add text annotations for best values
    best_val_idx = np.argmax(combined_history['val_accuracy'])
    best_val_acc = combined_history['val_accuracy'][best_val_idx]
    ax.annotate(f'Best: {best_val_acc:.4f}', 
                xy=(best_val_idx + 1, best_val_acc),
                xytext=(best_val_idx + 1 + 2, best_val_acc - 0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_history_accuracy.png'))
    plt.close(fig)
    print("  ✓ training_history_accuracy.png")
    
    # --- Loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, combined_history['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    ax.plot(epochs, combined_history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    ax.axvline(x=phase1_epochs, color='gray', linestyle='--', linewidth=1.5, label=f'Phase 1→2 Boundary (Epoch {phase1_epochs})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss Over Epochs')
    ax.legend(loc='upper right')
    
    best_val_loss_idx = np.argmin(combined_history['val_loss'])
    best_val_loss = combined_history['val_loss'][best_val_loss_idx]
    ax.annotate(f'Best: {best_val_loss:.4f}',
                xy=(best_val_loss_idx + 1, best_val_loss),
                xytext=(best_val_loss_idx + 1 + 2, best_val_loss + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_history_loss.png'))
    plt.close(fig)
    print("  ✓ training_history_loss.png")


def plot_learning_rate(combined_history, phase1_epochs, save_dir):
    """Plot the learning rate schedule over epochs."""
    if 'lr' not in combined_history:
        print("  ⚠ No LR data in history, skipping LR plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(combined_history['lr']) + 1)
    ax.plot(epochs, combined_history['lr'], 'g-', linewidth=2, marker='D', markersize=4)
    ax.axvline(x=phase1_epochs, color='gray', linestyle='--', linewidth=1.5, label=f'Phase 1→2 Boundary')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule Over Epochs')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'learning_rate_schedule.png'))
    plt.close(fig)
    print("  ✓ learning_rate_schedule.png")


def plot_dataset_distribution(train_labels, val_labels, class_names, save_dir):
    """Plot the dataset class distribution for train and validation splits."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    train_counts = [np.sum(train_labels == i) for i in range(len(class_names))]
    val_counts = [np.sum(val_labels == i) for i in range(len(class_names))]
    
    bars1 = ax.bar(x - width/2, train_counts, width, label='Training', color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, val_counts, width, label='Validation', color='#FF9800', edgecolor='black', linewidth=0.5)
    
    # Add count labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset Distribution: Training vs Validation Split')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'dataset_distribution.png'))
    plt.close(fig)
    print("  ✓ dataset_distribution.png")


def plot_confusion_matrices(y_true, y_pred, class_names, save_dir):
    """Plot both normalized and raw confusion matrices."""
    cm = confusion_matrix(y_true, y_pred)
    
    # --- Normalized ---
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # handle division by zero
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1, linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Normalized Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close(fig)
    print("  ✓ confusion_matrix.png")
    
    # --- Raw Counts ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Raw Counts)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'confusion_matrix_raw.png'))
    plt.close(fig)
    print("  ✓ confusion_matrix_raw.png")
    
    return cm, cm_norm


def plot_classification_report(y_true, y_pred, class_names, save_dir):
    """Generate and save a visual classification report table."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # Build table data
    metrics_data = []
    for cls in class_names:
        if cls in report:
            metrics_data.append([
                cls,
                f"{report[cls]['precision']:.4f}",
                f"{report[cls]['recall']:.4f}",
                f"{report[cls]['f1-score']:.4f}",
                f"{int(report[cls]['support'])}"
            ])
    
    # Add summary rows
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report:
            metrics_data.append([
                avg_type,
                f"{report[avg_type]['precision']:.4f}",
                f"{report[avg_type]['recall']:.4f}",
                f"{report[avg_type]['f1-score']:.4f}",
                f"{int(report[avg_type]['support'])}"
            ])
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(metrics_data) * 0.5 + 2)))
    ax.axis('off')
    
    table = ax.table(
        cellText=metrics_data,
        colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(5):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Style summary rows
    num_classes = len(class_names)
    for i in range(num_classes + 1, len(metrics_data) + 1):
        for j in range(5):
            table[i, j].set_facecolor('#E3F2FD')
            table[i, j].set_text_props(fontweight='bold')
    
    ax.set_title('Classification Report', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'classification_report.png'))
    plt.close(fig)
    print("  ✓ classification_report.png")
    
    return report


def plot_per_class_metrics(report, class_names, save_dir):
    """Plot grouped bar chart of Precision, Recall, F1 per class."""
    precisions = [report[cls]['precision'] for cls in class_names if cls in report]
    recalls = [report[cls]['recall'] for cls in class_names if cls in report]
    f1s = [report[cls]['f1-score'] for cls in class_names if cls in report]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, precisions, width, label='Precision', color='#1976D2', edgecolor='black', linewidth=0.5)
    ax.bar(x, recalls, width, label='Recall', color='#388E3C', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, f1s, width, label='F1-Score', color='#F57C00', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision, Recall, and F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right')
    
    # Add value labels
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
        ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'per_class_metrics.png'))
    plt.close(fig)
    print("  ✓ per_class_metrics.png")


def plot_roc_curves(y_true, y_prob, class_names, save_dir):
    """Plot One-vs-Rest ROC curves with AUC for each class."""
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Handle binary case
    if num_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Micro-average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    
    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot macro/micro averages
    ax.plot(fpr['micro'], tpr['micro'], color='deeppink', linestyle=':', linewidth=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})')
    ax.plot(fpr['macro'], tpr['macro'], color='navy', linestyle=':', linewidth=3,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})')
    
    # Plot per-class
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for i, color in zip(range(num_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, linewidth=1.5,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves — One-vs-Rest')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close(fig)
    print("  ✓ roc_curves.png")
    
    return roc_auc


def plot_tsne_embeddings(embeddings, labels, class_names, save_dir):
    """Plot t-SNE 2D visualization of embeddings colored by class."""
    print("    Computing t-SNE (this may take a moment)...")
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color], label=cls_name, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of Embedding Space')
    ax.legend(loc='best', markerscale=1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'tsne_embeddings.png'))
    plt.close(fig)
    print("  ✓ tsne_embeddings.png")


def plot_pca_embeddings(embeddings, labels, class_names, save_dir):
    """Plot PCA 2D and 3D visualizations of embeddings."""
    pca_3d = PCA(n_components=3)
    embeddings_3d = pca_3d.fit_transform(embeddings)
    explained_var = pca_3d.explained_variance_ratio_
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    # --- 2D PCA ---
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1],
                   c=[color], label=cls_name, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
    ax.set_title('PCA 2D Visualization of Embedding Space')
    ax.legend(loc='best', markerscale=1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pca_embeddings_2d.png'))
    plt.close(fig)
    print("  ✓ pca_embeddings_2d.png")
    
    # --- 3D PCA ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                   c=[color], label=cls_name, s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
    
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}%)')
    ax.set_title('PCA 3D Visualization of Embedding Space')
    ax.legend(loc='best', fontsize=8, markerscale=1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pca_embeddings_3d.png'))
    plt.close(fig)
    print("  ✓ pca_embeddings_3d.png")
    
    return explained_var


def plot_cosine_similarity_heatmap(embeddings, labels, class_names, save_dir):
    """Plot cosine similarity heatmap between class centroids."""
    num_classes = len(class_names)
    centroids = np.zeros((num_classes, embeddings.shape[1]))
    
    for i in range(num_classes):
        mask = labels == i
        if np.any(mask):
            centroid = embeddings[mask].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[i] = centroid
    
    # Cosine similarity matrix
    cos_sim = np.dot(centroids, centroids.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cos_sim, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=-1, vmax=1, linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Cosine Similarity'})
    ax.set_title('Cosine Similarity Between Class Centroids')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'cosine_similarity_heatmap.png'))
    plt.close(fig)
    print("  ✓ cosine_similarity_heatmap.png")
    
    return cos_sim


def plot_intra_inter_class_distances(embeddings, labels, class_names, save_dir):
    """Plot histogram of intra-class vs inter-class cosine distances."""
    intra_distances = []
    inter_distances = []
    
    unique_labels = np.unique(labels)
    
    for cls in unique_labels:
        cls_embeddings = embeddings[labels == cls]
        other_embeddings = embeddings[labels != cls]
        
        if len(cls_embeddings) > 1:
            # Intra-class: pairwise cosine distances within class
            intra_cos = cdist(cls_embeddings, cls_embeddings, metric='cosine')
            # Get upper triangle (exclude diagonal)
            triu_idx = np.triu_indices(len(cls_embeddings), k=1)
            intra_distances.extend(intra_cos[triu_idx].tolist())
        
        if len(cls_embeddings) > 0 and len(other_embeddings) > 0:
            # Inter-class: cosine distances to samples from other classes
            # Sample to avoid memory issues
            n_sample = min(100, len(other_embeddings))
            sampled_idx = np.random.choice(len(other_embeddings), n_sample, replace=False)
            inter_cos = cdist(cls_embeddings, other_embeddings[sampled_idx], metric='cosine')
            inter_distances.extend(inter_cos.ravel().tolist())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if intra_distances:
        ax.hist(intra_distances, bins=50, alpha=0.7, label='Intra-class Distance', color='#2196F3', edgecolor='black', linewidth=0.5, density=True)
    if inter_distances:
        ax.hist(inter_distances, bins=50, alpha=0.7, label='Inter-class Distance', color='#F44336', edgecolor='black', linewidth=0.5, density=True)
    
    ax.set_xlabel('Cosine Distance')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Intra-class vs Inter-class Cosine Distances')
    ax.legend(loc='upper right')
    
    # Add statistics text
    if intra_distances and inter_distances:
        intra_mean = np.mean(intra_distances)
        inter_mean = np.mean(inter_distances)
        separation = inter_mean - intra_mean
        stats_text = (f'Intra-class mean: {intra_mean:.4f}\n'
                      f'Inter-class mean: {inter_mean:.4f}\n'
                      f'Separation gap: {separation:.4f}')
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'intra_inter_class_distances.png'))
    plt.close(fig)
    print("  ✓ intra_inter_class_distances.png")
    
    return np.mean(intra_distances) if intra_distances else 0, np.mean(inter_distances) if inter_distances else 0


def plot_embedding_spread(embeddings, labels, class_names, save_dir):
    """Plot per-class embedding spread (standard deviation from centroid)."""
    num_classes = len(class_names)
    spreads = []
    
    for i in range(num_classes):
        mask = labels == i
        if np.sum(mask) > 1:
            cls_embeddings = embeddings[mask]
            centroid = cls_embeddings.mean(axis=0)
            distances = np.linalg.norm(cls_embeddings - centroid, axis=1)
            spreads.append(np.std(distances))
        else:
            spreads.append(0.0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(num_classes), spreads, color='#7B1FA2', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Embedding Spread (Std. Dev. from Centroid)')
    ax.set_title('Per-Class Embedding Compactness')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, spreads):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'embedding_spread.png'))
    plt.close(fig)
    print("  ✓ embedding_spread.png")
    
    return spreads


def save_model_summary(embedding_model, training_model, save_dir):
    """Save model architecture summary to text file."""
    summary_path = os.path.join(save_dir, 'model_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EMBEDDING MODEL ARCHITECTURE\n")
        f.write("=" * 80 + "\n\n")
        embedding_model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("TRAINING MODEL ARCHITECTURE (with Classification Head)\n")
        f.write("=" * 80 + "\n\n")
        training_model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Parameter counts
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("PARAMETER SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        total_params = embedding_model.count_params()
        trainable_params = sum(tf.keras.backend.count_params(w) for w in embedding_model.trainable_weights)
        non_trainable = total_params - trainable_params
        
        f.write(f"Embedding Model:\n")
        f.write(f"  Total parameters:         {total_params:,}\n")
        f.write(f"  Trainable parameters:     {trainable_params:,}\n")
        f.write(f"  Non-trainable parameters: {non_trainable:,}\n")
        f.write(f"  Model size (approx):      {total_params * 4 / (1024*1024):.2f} MB (float32)\n")
    
    print(f"  ✓ model_summary.txt")


def run_thesis_evaluation(training_model, embedding_model, val_dataset, 
                          val_labels, class_names, combined_history, 
                          phase1_epochs, train_labels, save_dir):
    """Run comprehensive thesis-quality evaluation and generate all plots."""
    
    print("\n" + "=" * 70)
    print("  THESIS-QUALITY EVALUATION & VISUALIZATION")
    print("=" * 70)
    
    metrics_dict = {}
    
    # --- 1. Training History Plots ---
    print("\n[1/9] Generating training history plots...")
    plot_training_history(combined_history, phase1_epochs, save_dir)
    plot_learning_rate(combined_history, phase1_epochs, save_dir)
    
    # --- 2. Dataset Distribution ---
    print("\n[2/9] Generating dataset distribution plot...")
    plot_dataset_distribution(train_labels, val_labels, class_names, save_dir)
    
    # --- 3. Get predictions & embeddings ---
    print("\n[3/9] Computing predictions and embeddings on validation set...")
    y_prob = training_model.predict(val_dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = val_labels
    
    # Get embeddings from the embedding model
    all_embeddings = embedding_model.predict(val_dataset, verbose=0)
    
    # --- 4. Classification accuracy metrics ---
    print("\n[4/9] Computing classification metrics...")
    overall_acc = np.mean(y_pred == y_true)
    metrics_dict['accuracy'] = float(overall_acc)
    print(f"  Overall Accuracy: {overall_acc:.4f}")
    
    # Top-K accuracy
    num_classes = len(class_names)
    for k in [1, 3, 5]:
        if k <= num_classes:
            try:
                topk_acc = top_k_accuracy_score(y_true, y_prob, k=k, labels=range(num_classes))
                metrics_dict[f'top_{k}_accuracy'] = float(topk_acc)
                print(f"  Top-{k} Accuracy: {topk_acc:.4f}")
            except Exception as e:
                print(f"  ⚠ Top-{k} accuracy failed: {e}")
    
    # Precision, Recall, F1 (macro, weighted, micro)
    for avg in ['macro', 'weighted', 'micro']:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        metrics_dict[f'precision_{avg}'] = float(p)
        metrics_dict[f'recall_{avg}'] = float(r)
        metrics_dict[f'f1_{avg}'] = float(f1)
        print(f"  {avg.capitalize()} — P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
    
    # --- 5. Confusion Matrices ---
    print("\n[5/9] Generating confusion matrices...")
    cm, cm_norm = plot_confusion_matrices(y_true, y_pred, class_names, save_dir)
    
    # --- 6. Classification Report & Per-Class Metrics ---
    print("\n[6/9] Generating classification report and per-class metrics...")
    report = plot_classification_report(y_true, y_pred, class_names, save_dir)
    plot_per_class_metrics(report, class_names, save_dir)
    
    # Save per-class metrics
    per_class = {}
    for cls in class_names:
        if cls in report:
            per_class[cls] = {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1-score': report[cls]['f1-score'],
                'support': report[cls]['support']
            }
    metrics_dict['per_class'] = per_class
    
    # --- 7. ROC Curves ---
    print("\n[7/9] Generating ROC curves...")
    roc_auc = plot_roc_curves(y_true, y_prob, class_names, save_dir)
    metrics_dict['auc_micro'] = float(roc_auc['micro'])
    metrics_dict['auc_macro'] = float(roc_auc['macro'])
    for i, cls in enumerate(class_names):
        metrics_dict[f'auc_{cls}'] = float(roc_auc[i])
    print(f"  Micro-avg AUC: {roc_auc['micro']:.4f}")
    print(f"  Macro-avg AUC: {roc_auc['macro']:.4f}")
    
    # --- 8. Embedding-Specific Metrics ---
    print("\n[8/9] Computing embedding-space metrics...")
    
    # t-SNE
    plot_tsne_embeddings(all_embeddings, y_true, class_names, save_dir)
    
    # PCA
    explained_var = plot_pca_embeddings(all_embeddings, y_true, class_names, save_dir)
    metrics_dict['pca_explained_variance'] = [float(v) for v in explained_var]
    
    # Cosine Similarity Heatmap
    cos_sim = plot_cosine_similarity_heatmap(all_embeddings, y_true, class_names, save_dir)
    
    # Intra/Inter-class distances
    intra_mean, inter_mean = plot_intra_inter_class_distances(all_embeddings, y_true, class_names, save_dir)
    metrics_dict['intra_class_distance_mean'] = float(intra_mean)
    metrics_dict['inter_class_distance_mean'] = float(inter_mean)
    metrics_dict['distance_separation_gap'] = float(inter_mean - intra_mean)
    
    # Embedding spread
    spreads = plot_embedding_spread(all_embeddings, y_true, class_names, save_dir)
    metrics_dict['embedding_spread_per_class'] = {cls: float(s) for cls, s in zip(class_names, spreads)}
    metrics_dict['embedding_spread_mean'] = float(np.mean(spreads))
    
    # Silhouette Score
    if len(np.unique(y_true)) > 1 and len(all_embeddings) > len(np.unique(y_true)):
        try:
            sil_score = silhouette_score(all_embeddings, y_true, metric='cosine')
            metrics_dict['silhouette_score'] = float(sil_score)
            print(f"  Silhouette Score (cosine): {sil_score:.4f}")
        except Exception as e:
            print(f"  ⚠ Silhouette score failed: {e}")
            metrics_dict['silhouette_score'] = None
    
    # --- 9. Save Model Summary & All Metrics ---
    print("\n[9/9] Saving model summary, config, and metrics JSON...")
    save_model_summary(embedding_model, training_model, save_dir)
    
    # Training config
    training_config = {
        'img_size': config.IMG_SIZE,
        'embedding_dim': config.EMBEDDING_DIM,
        'backbone': config.BACKBONE,
        'alpha': config.ALPHA,
        'batch_size': config.BATCH_SIZE,
        'epochs_phase1': config.EPOCHS_PHASE1,
        'epochs_phase2': config.EPOCHS_PHASE2,
        'learning_rate': config.LEARNING_RATE,
        'num_classes': len(class_names),
        'class_names': class_names,
        'train_samples': int(len(train_labels)),
        'val_samples': int(len(val_labels)),
    }
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=2)
    print("  ✓ training_config.json")
    
    # Training history (best values)
    metrics_dict['best_train_accuracy'] = float(max(combined_history['accuracy']))
    metrics_dict['best_val_accuracy'] = float(max(combined_history['val_accuracy']))
    metrics_dict['best_train_loss'] = float(min(combined_history['loss']))
    metrics_dict['best_val_loss'] = float(min(combined_history['val_loss']))
    metrics_dict['total_epochs_trained'] = len(combined_history['accuracy'])
    
    # Save all metrics
    with open(os.path.join(save_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print("  ✓ training_metrics.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Accuracy:              {metrics_dict['accuracy']:.4f}")
    print(f"  Macro F1-Score:        {metrics_dict['f1_macro']:.4f}")
    print(f"  Weighted F1-Score:     {metrics_dict['f1_weighted']:.4f}")
    print(f"  Macro AUC:             {metrics_dict['auc_macro']:.4f}")
    if 'silhouette_score' in metrics_dict and metrics_dict['silhouette_score'] is not None:
        print(f"  Silhouette Score:      {metrics_dict['silhouette_score']:.4f}")
    print(f"  Intra-class Dist Mean: {metrics_dict['intra_class_distance_mean']:.4f}")
    print(f"  Inter-class Dist Mean: {metrics_dict['inter_class_distance_mean']:.4f}")
    print(f"  Separation Gap:        {metrics_dict['distance_separation_gap']:.4f}")
    print(f"  Embedding Spread Mean: {metrics_dict['embedding_spread_mean']:.4f}")
    print("=" * 70)
    print(f"  All plots saved to: {save_dir}/")
    print("=" * 70)
    
    return metrics_dict


# ==================== Training ====================
def train():
    print("=" * 70)
    print("Training Few-Shot Embedding Model for ESP32")
    print("=" * 70)
    print(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"Model: MobileNetV2 (alpha={config.ALPHA})")
    print("=" * 70)
    
    training_start_time = time.time()
    
    # Load data
    print("\n1. Loading dataset...")
    image_paths, labels, class_names = load_dataset(config.DATA_DIR)
    num_classes = len(class_names)
    
    # Validate enough samples for stratified split
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    if min_count < 2:
        print(f"⚠ Warning: Class '{class_names[unique_labels[counts.argmin()]]}' has only {min_count} image(s).")
        print("  Need at least 2 per class for train/val split. Using random split instead.")
        stratify = None
    else:
        stratify = labels
    
    # Save class names
    with open(os.path.join(config.MODEL_DIR, 'class_names.json'), 'w') as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=stratify, random_state=42
    )
    
    print(f"Training: {len(train_paths)}, Validation: {len(val_paths)}")
    
    # Create datasets
    print("\n2. Creating datasets...")
    train_dataset = create_dataset(train_paths, train_labels, 
                                   config.BATCH_SIZE, config.IMG_SIZE, True)
    val_dataset = create_dataset(val_paths, val_labels, 
                                 config.BATCH_SIZE, config.IMG_SIZE, False)
    
    # Create models
    print("\n3. Creating model...")
    embedding_model, base_model = create_embedding_model(
        (config.IMG_SIZE, config.IMG_SIZE, 3),
        config.EMBEDDING_DIM
    )
    
    training_model = create_training_model(embedding_model, num_classes)
    
    # Compile
    training_model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nEmbedding model architecture:")
    embedding_model.summary()
    
    # Callbacks
    best_model_path = os.path.join(config.MODEL_DIR, 'best_training_model.h5')
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train with partially frozen base
    print(f"\n4. Training Phase 1 ({config.EPOCHS_PHASE1} epochs, top layers only)...")
    history1 = training_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS_PHASE1,
        callbacks=callbacks
    )
    
    phase1_epochs = len(history1.history['accuracy'])
    
    # Phase 2: Fine-tune entire model with lower LR
    print(f"\n5. Training Phase 2 ({config.EPOCHS_PHASE2} epochs, full fine-tune)...")
    for layer in base_model.layers:
        layer.trainable = True
    
    training_model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE * 0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = training_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS_PHASE2,
        callbacks=callbacks
    )
    
    training_time = time.time() - training_start_time
    
    # Save embedding model (standalone, NOT extracted from training model)
    print("\n6. Saving embedding model...")
    
    # The embedding_model has already been updated (shared weights with training_model)
    # because it's the same tf.keras.Model object used inside training_model.
    embedding_model_path = os.path.join(config.MODEL_DIR, 'embedding_model.keras')
    embedding_model.save(embedding_model_path)
    print(f"✓ Embedding model saved: {embedding_model_path}")
    
    # Also export as SavedModel for TFLite conversion (most reliable)
    saved_model_path = os.path.join(config.MODEL_DIR, 'embedding_model_savedmodel')
    try:
        embedding_model.export(saved_model_path)
        print(f"✓ SavedModel exported: {saved_model_path}")
    except Exception as e:
        print(f"⚠ SavedModel export failed ({e}), .keras file is available for conversion.")
    
    # Save config
    config_dict = {
        'img_size': config.IMG_SIZE,
        'embedding_dim': config.EMBEDDING_DIM,
        'backbone': config.BACKBONE,
        'alpha': config.ALPHA,
        'num_classes_trained': num_classes,
        'class_names': class_names,
        'training_time_seconds': training_time
    }
    with open(os.path.join(config.MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Print training summary
    best_val_acc_p1 = max(history1.history.get('val_accuracy', [0]))
    best_val_acc_p2 = max(history2.history.get('val_accuracy', [0]))
    
    print("\n" + "=" * 70)
    print("✓ Training completed!")
    print(f"  Phase 1 best val accuracy: {best_val_acc_p1:.4f}")
    print(f"  Phase 2 best val accuracy: {best_val_acc_p2:.4f}")
    print(f"  Overall best val accuracy: {max(best_val_acc_p1, best_val_acc_p2):.4f}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"✓ Embedding model: {embedding_model_path}")
    print(f"✓ Config: {config.MODEL_DIR}/config.json")
    print("=" * 70)
    
    # ==================== Thesis Evaluation ====================
    print("\n7. Running thesis-quality evaluation...")
    
    # Merge training histories
    combined_history = merge_histories(history1, history2)
    
    # Run full evaluation
    metrics = run_thesis_evaluation(
        training_model=training_model,
        embedding_model=embedding_model,
        val_dataset=val_dataset,
        val_labels=val_labels,
        class_names=class_names,
        combined_history=combined_history,
        phase1_epochs=phase1_epochs,
        train_labels=train_labels,
        save_dir=config.DETAILS_DIR
    )
    
    # Add training time to metrics
    metrics['training_time_seconds'] = training_time
    with open(os.path.join(config.DETAILS_DIR, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    train()