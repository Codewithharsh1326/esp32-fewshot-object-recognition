# compare_models.py
# Compare original Keras embedding model vs INT8 quantized TFLite model
# Generates thesis-quality comparison plots into details/comparison/

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, silhouette_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Ensure sibling imports work regardless of CWD
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_fewshot_embeddings import L2NormalizeLayer

# ==================== Config ====================
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.figsize': (10, 8), 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1, 'axes.grid': True, 'grid.alpha': 0.3,
})

# Resolve project root (parent of Code/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAVE_DIR = os.path.join(PROJECT_ROOT, "details", "comparison")
os.makedirs(SAVE_DIR, exist_ok=True)

KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "embedding_model.keras")
INT8_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "embedding_model_int8_esp32.tflite")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ==================== Helpers ====================

def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'models', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {'img_size': 96, 'embedding_dim': 128}


def get_all_images(data_dir):
    """Get all images with class labels."""
    image_paths, labels, class_names = [], [], []
    data_path = Path(data_dir)
    for class_idx, class_dir in enumerate(sorted(data_path.iterdir())):
        if class_dir.is_dir():
            class_names.append(class_dir.name)
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for img_path in class_dir.glob(ext):
                    image_paths.append(img_path)
                    labels.append(class_idx)
    return image_paths, np.array(labels), class_names


def preprocess_image(img_path, img_size):
    """Load and preprocess a single image."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_array = np.array(img).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def get_keras_embedding(model, img_array):
    """Get L2-normalized embedding from Keras model."""
    emb = model.predict(img_array, verbose=0).flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def get_int8_embedding(interpreter, img_array):
    """Get L2-normalized embedding from INT8 TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = img_array.copy()
    if input_details[0]['dtype'] == np.uint8:
        s, z = input_details[0]['quantization']
        inp = np.clip(inp / s + z, 0, 255).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] == np.int8:
        s, z = output_details[0]['quantization']
        output = (output.astype(np.float32) - z) * s

    emb = output.flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def nearest_centroid_predict(embeddings, labels, num_classes):
    """Nearest-centroid classification using cosine similarity."""
    centroids = np.zeros((num_classes, embeddings.shape[1]))
    for i in range(num_classes):
        mask = labels == i
        if np.any(mask):
            c = embeddings[mask].mean(axis=0)
            centroids[i] = c / (np.linalg.norm(c) + 1e-8)
    sims = np.dot(embeddings, centroids.T)
    return np.argmax(sims, axis=1), centroids


# ==================== Plots ====================

def plot_model_size(keras_size_kb, int8_size_kb):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Original Keras\n(Float32)', 'TFLite INT8\n(Quantized)']
    sizes = [keras_size_kb, int8_size_kb]
    colors = ['#1976D2', '#388E3C']
    bars = ax.bar(labels, sizes, color=colors, edgecolor='black', linewidth=0.7, width=0.45)

    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{s:.1f} KB\n({s/1024:.2f} MB)', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ratio = keras_size_kb / int8_size_kb if int8_size_kb > 0 else 0
    ax.annotate(f'{ratio:.1f}× smaller', xy=(1, int8_size_kb/2),
                fontsize=13, ha='center', color='white', fontweight='bold')

    ax.set_ylabel('Model Size (KB)')
    ax.set_title('Model Size: Original Keras vs INT8 Quantized')
    ax.set_ylim(0, max(sizes) * 1.35)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'model_size_comparison.png'))
    plt.close(fig)
    print("  ✓ model_size_comparison.png")
    return ratio


def plot_embedding_fidelity(cos_sims):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cos_sims, bins=40, color='#7B1FA2', edgecolor='black', linewidth=0.5, alpha=0.8, density=True)
    mean_s = np.mean(cos_sims)
    ax.axvline(mean_s, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_s:.4f}')
    ax.axvline(np.median(cos_sims), color='orange', linestyle='-.', linewidth=2, label=f'Median: {np.median(cos_sims):.4f}')
    stats = (f'Mean: {mean_s:.4f}\nMedian: {np.median(cos_sims):.4f}\n'
             f'Min: {np.min(cos_sims):.4f}\nStd: {np.std(cos_sims):.4f}')
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_xlabel('Cosine Similarity (Keras vs INT8 Embedding)')
    ax.set_ylabel('Density')
    ax.set_title('Embedding Fidelity: Original Keras Model vs INT8 Quantized')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'embedding_fidelity.png'))
    plt.close(fig)
    print("  ✓ embedding_fidelity.png")


def plot_accuracy_comparison(keras_m, int8_m, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Overall
    ax = axes[0]
    names = ['Accuracy', 'Macro\nPrecision', 'Macro\nRecall', 'Macro F1', 'Weighted F1']
    k_vals = [keras_m['accuracy'], keras_m['precision'], keras_m['recall'], keras_m['macro_f1'], keras_m['weighted_f1']]
    i_vals = [int8_m['accuracy'], int8_m['precision'], int8_m['recall'], int8_m['macro_f1'], int8_m['weighted_f1']]
    x = np.arange(len(names))
    w = 0.3
    b1 = ax.bar(x - w/2, k_vals, w, label='Keras (Float32)', color='#1976D2', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, i_vals, w, label='INT8 Quantized', color='#388E3C', edgecolor='black', linewidth=0.5)
    for b, v in zip(b1, k_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    for b, v in zip(b2, i_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics: Keras vs INT8')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15); ax.legend()

    # Per-class
    ax = axes[1]
    x = np.arange(len(class_names))
    w = 0.35
    ax.bar(x - w/2, keras_m['per_class_acc'], w, label='Keras (Float32)', color='#1976D2', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, int8_m['per_class_acc'], w, label='INT8 Quantized', color='#388E3C', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy: Keras vs INT8')
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.15); ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'accuracy_comparison.png'))
    plt.close(fig)
    print("  ✓ accuracy_comparison.png")


def plot_confusion_matrices_side_by_side(y_true, keras_preds, int8_preds, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, preds, title in zip(axes, [keras_preds, int8_preds],
                                 ['Keras (Float32)', 'INT8 Quantized']):
        cm = confusion_matrix(y_true, preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=1, linewidths=0.5, linecolor='gray',
                    cbar_kws={'label': 'Proportion'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix — {title}')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'confusion_matrices_comparison.png'))
    plt.close(fig)
    print("  ✓ confusion_matrices_comparison.png")


def plot_tsne_comparison(keras_emb, int8_emb, labels, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    perplexity = min(30, len(labels) - 1)

    for ax, emb, title in zip(axes, [keras_emb, int8_emb],
                               ['Keras (Float32)', 'INT8 Quantized']):
        print(f"    Computing t-SNE for {title}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        emb_2d = tsne.fit_transform(emb)
        for i, (cls, col) in enumerate(zip(class_names, colors)):
            mask = labels == i
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[col], label=cls,
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.4)
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.set_title(f't-SNE — {title}')
        ax.legend(loc='best', fontsize=8, markerscale=1.0)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'tsne_comparison.png'))
    plt.close(fig)
    print("  ✓ tsne_comparison.png")


def plot_pca_comparison(keras_emb, int8_emb, labels, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for ax, emb, title in zip(axes, [keras_emb, int8_emb],
                               ['Keras (Float32)', 'INT8 Quantized']):
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb)
        ev = pca.explained_variance_ratio_
        for i, (cls, col) in enumerate(zip(class_names, colors)):
            mask = labels == i
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[col], label=cls,
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.4)
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
        ax.set_title(f'PCA — {title}')
        ax.legend(loc='best', fontsize=8, markerscale=1.0)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'pca_comparison.png'))
    plt.close(fig)
    print("  ✓ pca_comparison.png")


def plot_cosine_heatmaps(keras_emb, int8_emb, labels, class_names):
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, emb, title in zip(axes, [keras_emb, int8_emb],
                               ['Keras (Float32)', 'INT8 Quantized']):
        centroids = np.zeros((num_classes, emb.shape[1]))
        for i in range(num_classes):
            mask = labels == i
            if np.any(mask):
                c = emb[mask].mean(axis=0)
                centroids[i] = c / (np.linalg.norm(c) + 1e-8)
        cos_sim = np.dot(centroids, centroids.T)

        sns.heatmap(cos_sim, annot=True, fmt='.3f', cmap='RdYlGn_r',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=-1, vmax=1, linewidths=0.5, linecolor='gray',
                    cbar_kws={'label': 'Cosine Similarity'})
        ax.set_title(f'Centroid Cosine Similarity — {title}')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'cosine_similarity_comparison.png'))
    plt.close(fig)
    print("  ✓ cosine_similarity_comparison.png")


def plot_intra_inter_comparison(keras_emb, int8_emb, labels, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, emb, title in zip(axes, [keras_emb, int8_emb],
                               ['Keras (Float32)', 'INT8 Quantized']):
        intra, inter = [], []
        for cls in np.unique(labels):
            cls_e = emb[labels == cls]
            other_e = emb[labels != cls]
            if len(cls_e) > 1:
                d = cdist(cls_e, cls_e, metric='cosine')
                triu = np.triu_indices(len(cls_e), k=1)
                intra.extend(d[triu].tolist())
            if len(cls_e) > 0 and len(other_e) > 0:
                n = min(100, len(other_e))
                idx = np.random.choice(len(other_e), n, replace=False)
                d = cdist(cls_e, other_e[idx], metric='cosine')
                inter.extend(d.ravel().tolist())

        if intra:
            ax.hist(intra, bins=50, alpha=0.7, label='Intra-class', color='#2196F3', edgecolor='black', linewidth=0.3, density=True)
        if inter:
            ax.hist(inter, bins=50, alpha=0.7, label='Inter-class', color='#F44336', edgecolor='black', linewidth=0.3, density=True)

        if intra and inter:
            stats = f'Intra mean: {np.mean(intra):.4f}\nInter mean: {np.mean(inter):.4f}\nGap: {np.mean(inter)-np.mean(intra):.4f}'
            ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel('Cosine Distance')
        ax.set_ylabel('Density')
        ax.set_title(f'Intra vs Inter-class Distance — {title}')
        ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'intra_inter_distance_comparison.png'))
    plt.close(fig)
    print("  ✓ intra_inter_distance_comparison.png")


def plot_embedding_spread_comparison(keras_emb, int8_emb, labels, class_names):
    fig, ax = plt.subplots(figsize=(14, 6))
    num_classes = len(class_names)

    keras_spreads, int8_spreads = [], []
    for i in range(num_classes):
        mask = labels == i
        for emb_arr, spreads_list in [(keras_emb, keras_spreads), (int8_emb, int8_spreads)]:
            if np.sum(mask) > 1:
                ce = emb_arr[mask]
                dists = np.linalg.norm(ce - ce.mean(axis=0), axis=1)
                spreads_list.append(np.std(dists))
            else:
                spreads_list.append(0.0)

    x = np.arange(num_classes)
    w = 0.35
    ax.bar(x - w/2, keras_spreads, w, label='Keras (Float32)', color='#1976D2', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, int8_spreads, w, label='INT8 Quantized', color='#388E3C', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Class')
    ax.set_ylabel('Embedding Spread (Std Dev from Centroid)')
    ax.set_title('Per-Class Embedding Compactness Comparison')
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'embedding_spread_comparison.png'))
    plt.close(fig)
    print("  ✓ embedding_spread_comparison.png")


def plot_latency_comparison(keras_times, int8_times):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot
    ax = axes[0]
    data = [np.array(keras_times)*1000, np.array(int8_times)*1000]
    bp = ax.boxplot(data, labels=['Keras\n(Float32)', 'INT8\n(Quantized)'], patch_artist=True,
                     medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('#BBDEFB')
    bp['boxes'][1].set_facecolor('#C8E6C9')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Latency Distribution')

    km = np.mean(keras_times)*1000
    im = np.mean(int8_times)*1000
    speedup = km / im if im > 0 else 0
    stats = f'Keras: {km:.2f} ms\nINT8: {im:.2f} ms\nSpeedup: {speedup:.2f}×'
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Bar
    ax = axes[1]
    bars = ax.bar(['Keras\n(Float32)', 'INT8\n(Quantized)'], [km, im],
                   color=['#1976D2', '#388E3C'], edgecolor='black', linewidth=0.7, width=0.4,
                   yerr=[np.std(keras_times)*1000, np.std(int8_times)*1000], capsize=8)
    for b, v in zip(bars, [km, im]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f'{v:.2f} ms',
                ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title(f'Mean Inference Latency (Speedup: {speedup:.2f}×)')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'latency_comparison.png'))
    plt.close(fig)
    print("  ✓ latency_comparison.png")
    return km, im, speedup


def plot_quantization_error_heatmap(keras_emb, int8_emb, labels, class_names, max_n=50):
    n = min(max_n, len(keras_emb))
    errors = np.abs(keras_emb[:n] - int8_emb[:n])
    sort_idx = np.argsort(labels[:n])
    errors_sorted = errors[sort_idx]
    labels_sorted = labels[:n][sort_idx]
    y_labels = [class_names[l] for l in labels_sorted]

    fig, ax = plt.subplots(figsize=(14, max(6, n*0.15 + 2)))
    sns.heatmap(errors_sorted, cmap='YlOrRd', ax=ax,
                yticklabels=y_labels if n <= 30 else False,
                xticklabels=False, cbar_kws={'label': 'Absolute Error'})
    ax.set_xlabel(f'Embedding Dimension (1–{errors.shape[1]})')
    ax.set_ylabel('Sample (grouped by class)')
    ax.set_title('Per-Dimension Quantization Error: Keras vs INT8')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quantization_error_heatmap.png'))
    plt.close(fig)
    print("  ✓ quantization_error_heatmap.png")


def plot_per_class_f1_comparison(keras_report, int8_report, class_names):
    fig, ax = plt.subplots(figsize=(14, 6))
    k_f1 = [keras_report.get(c, {}).get('f1-score', 0) for c in class_names]
    i_f1 = [int8_report.get(c, {}).get('f1-score', 0) for c in class_names]
    x = np.arange(len(class_names))
    w = 0.35
    b1 = ax.bar(x - w/2, k_f1, w, label='Keras (Float32)', color='#1976D2', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, i_f1, w, label='INT8 Quantized', color='#388E3C', edgecolor='black', linewidth=0.5)
    for b, v in zip(b1, k_f1):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'{v:.2f}', ha='center', fontsize=8)
    for b, v in zip(b2, i_f1):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_xlabel('Class')
    ax.set_ylabel('F1-Score')
    ax.set_title('Per-Class F1-Score: Keras (Float32) vs INT8 Quantized')
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.15); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'per_class_f1_comparison.png'))
    plt.close(fig)
    print("  ✓ per_class_f1_comparison.png")


def plot_summary_table(keras_m, int8_m, cos_sims, keras_sil, int8_sil,
                       keras_size_kb, int8_size_kb, km_ms, im_ms, speedup):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    data = [
        ['Model Size', f'{keras_size_kb:.1f} KB', f'{int8_size_kb:.1f} KB', f'{keras_size_kb/int8_size_kb:.1f}× smaller'],
        ['Accuracy', f'{keras_m["accuracy"]:.4f}', f'{int8_m["accuracy"]:.4f}', f'{keras_m["accuracy"]-int8_m["accuracy"]:+.4f}'],
        ['Macro F1', f'{keras_m["macro_f1"]:.4f}', f'{int8_m["macro_f1"]:.4f}', f'{keras_m["macro_f1"]-int8_m["macro_f1"]:+.4f}'],
        ['Weighted F1', f'{keras_m["weighted_f1"]:.4f}', f'{int8_m["weighted_f1"]:.4f}', f'{keras_m["weighted_f1"]-int8_m["weighted_f1"]:+.4f}'],
        ['Macro Precision', f'{keras_m["precision"]:.4f}', f'{int8_m["precision"]:.4f}', f'{keras_m["precision"]-int8_m["precision"]:+.4f}'],
        ['Macro Recall', f'{keras_m["recall"]:.4f}', f'{int8_m["recall"]:.4f}', f'{keras_m["recall"]-int8_m["recall"]:+.4f}'],
        ['Silhouette Score', f'{keras_sil:.4f}', f'{int8_sil:.4f}', f'{keras_sil-int8_sil:+.4f}'],
        ['Embedding Fidelity', '—', f'{np.mean(cos_sims):.4f}', 'cosine sim'],
        ['Latency (PC)', f'{km_ms:.2f} ms', f'{im_ms:.2f} ms', f'{speedup:.2f}× faster'],
    ]

    table = ax.table(
        cellText=data,
        colLabels=['Metric', 'Keras (Float32)', 'INT8 Quantized', 'Delta / Note'],
        cellLoc='center', loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    for j in range(4):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Comprehensive Model Comparison Summary', fontsize=16, fontweight='bold', pad=30)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'comparison_summary_table.png'))
    plt.close(fig)
    print("  ✓ comparison_summary_table.png")


# ==================== Main ====================

def main():
    print("=" * 70)
    print("  MODEL COMPARISON: Original Keras vs INT8 Quantized")
    print("=" * 70)

    cfg = load_config()
    img_size = cfg.get('img_size', 96)

    # Load models
    print("\n[1/12] Loading models...")
    keras_model = tf.keras.models.load_model(
        KERAS_MODEL_PATH, custom_objects={'L2NormalizeLayer': L2NormalizeLayer}
    )
    print(f"  ✓ Keras model: {KERAS_MODEL_PATH}")

    interp_int8 = tf.lite.Interpreter(model_path=INT8_MODEL_PATH)
    interp_int8.allocate_tensors()
    print(f"  ✓ INT8 TFLite: {INT8_MODEL_PATH}")

    keras_size_kb = os.path.getsize(KERAS_MODEL_PATH) / 1024
    int8_size_kb = os.path.getsize(INT8_MODEL_PATH) / 1024

    # Load images
    print("\n[2/12] Loading all images...")
    all_images, all_labels, class_names = get_all_images(DATA_DIR)
    num_classes = len(class_names)
    print(f"  {len(all_images)} images across {num_classes} classes: {class_names}")

    # Compute embeddings
    print("\n[3/12] Computing embeddings from both models...")
    keras_embeddings, int8_embeddings, valid_labels = [], [], []
    for i, (img_path, label) in enumerate(zip(all_images, all_labels)):
        try:
            img = preprocess_image(img_path, img_size)
            keras_embeddings.append(get_keras_embedding(keras_model, img))
            int8_embeddings.append(get_int8_embedding(interp_int8, img))
            valid_labels.append(label)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(all_images)}...")

    keras_emb = np.array(keras_embeddings)
    int8_emb = np.array(int8_embeddings)
    valid_labels = np.array(valid_labels)
    print(f"  ✓ {len(keras_emb)} embeddings computed")

    metrics = {}

    # --- Model size ---
    print("\n[4/12] Model size comparison...")
    ratio = plot_model_size(keras_size_kb, int8_size_kb)
    metrics['keras_size_kb'] = keras_size_kb
    metrics['int8_size_kb'] = int8_size_kb
    metrics['compression_ratio'] = ratio

    # --- Embedding fidelity ---
    print("\n[5/12] Embedding fidelity...")
    cos_sims = np.array([np.dot(k, i) / (np.linalg.norm(k)*np.linalg.norm(i)+1e-8)
                          for k, i in zip(keras_emb, int8_emb)])
    plot_embedding_fidelity(cos_sims)
    metrics['embedding_fidelity_mean'] = float(np.mean(cos_sims))
    metrics['embedding_fidelity_min'] = float(np.min(cos_sims))

    # --- Classification metrics ---
    print("\n[6/12] Classification metrics...")
    keras_preds, _ = nearest_centroid_predict(keras_emb, valid_labels, num_classes)
    int8_preds, _ = nearest_centroid_predict(int8_emb, valid_labels, num_classes)

    k_p, k_r, k_f1, _ = precision_recall_fscore_support(valid_labels, keras_preds, average='macro', zero_division=0)
    _, _, k_wf1, _ = precision_recall_fscore_support(valid_labels, keras_preds, average='weighted', zero_division=0)
    i_p, i_r, i_f1, _ = precision_recall_fscore_support(valid_labels, int8_preds, average='macro', zero_division=0)
    _, _, i_wf1, _ = precision_recall_fscore_support(valid_labels, int8_preds, average='weighted', zero_division=0)

    k_per_class = [np.mean(keras_preds[valid_labels == c] == c) if np.sum(valid_labels == c) > 0 else 0 for c in range(num_classes)]
    i_per_class = [np.mean(int8_preds[valid_labels == c] == c) if np.sum(valid_labels == c) > 0 else 0 for c in range(num_classes)]

    keras_m = {'accuracy': np.mean(keras_preds == valid_labels), 'precision': k_p, 'recall': k_r,
               'macro_f1': k_f1, 'weighted_f1': k_wf1, 'per_class_acc': k_per_class}
    int8_m = {'accuracy': np.mean(int8_preds == valid_labels), 'precision': i_p, 'recall': i_r,
              'macro_f1': i_f1, 'weighted_f1': i_wf1, 'per_class_acc': i_per_class}

    plot_accuracy_comparison(keras_m, int8_m, class_names)
    metrics['keras_accuracy'] = float(keras_m['accuracy'])
    metrics['int8_accuracy'] = float(int8_m['accuracy'])
    metrics['accuracy_drop'] = float(keras_m['accuracy'] - int8_m['accuracy'])
    metrics['keras_macro_f1'] = float(k_f1)
    metrics['int8_macro_f1'] = float(i_f1)

    # --- Per-class F1 ---
    print("\n[7/12] Per-class F1 comparison...")
    k_report = classification_report(valid_labels, keras_preds, target_names=class_names, output_dict=True, zero_division=0)
    i_report = classification_report(valid_labels, int8_preds, target_names=class_names, output_dict=True, zero_division=0)
    plot_per_class_f1_comparison(k_report, i_report, class_names)

    # --- Confusion matrices ---
    print("\n[8/12] Confusion matrices...")
    plot_confusion_matrices_side_by_side(valid_labels, keras_preds, int8_preds, class_names)

    # --- t-SNE comparison ---
    print("\n[9/12] t-SNE comparison...")
    plot_tsne_comparison(keras_emb, int8_emb, valid_labels, class_names)

    # --- PCA comparison ---
    print("\n[10/12] PCA comparison...")
    plot_pca_comparison(keras_emb, int8_emb, valid_labels, class_names)

    # --- Cosine similarity heatmaps ---
    print("\n[10/12] Cosine similarity heatmaps...")
    plot_cosine_heatmaps(keras_emb, int8_emb, valid_labels, class_names)

    # --- Intra/Inter distances ---
    print("\n[10/12] Intra vs inter-class distances...")
    plot_intra_inter_comparison(keras_emb, int8_emb, valid_labels, class_names)

    # --- Embedding spread ---
    print("\n[10/12] Embedding spread comparison...")
    plot_embedding_spread_comparison(keras_emb, int8_emb, valid_labels, class_names)

    # --- Quantization error heatmap ---
    print("\n[10/12] Quantization error heatmap...")
    plot_quantization_error_heatmap(keras_emb, int8_emb, valid_labels, class_names)

    # --- Silhouette scores ---
    print("\n[11/12] Silhouette scores...")
    keras_sil = silhouette_score(keras_emb, valid_labels, metric='cosine') if len(np.unique(valid_labels)) > 1 else 0
    int8_sil = silhouette_score(int8_emb, valid_labels, metric='cosine') if len(np.unique(valid_labels)) > 1 else 0
    metrics['keras_silhouette'] = float(keras_sil)
    metrics['int8_silhouette'] = float(int8_sil)
    print(f"  Keras Silhouette: {keras_sil:.4f}")
    print(f"  INT8  Silhouette: {int8_sil:.4f}")

    # --- Latency ---
    print("\n[11/12] Latency benchmark...")
    keras_times, int8_times = [], []
    # Warmup
    test_img = preprocess_image(all_images[0], img_size)
    for _ in range(5):
        get_keras_embedding(keras_model, test_img)
        get_int8_embedding(interp_int8, test_img)

    sample_idx = np.random.choice(len(all_images), min(50, len(all_images)), replace=True)
    for idx in sample_idx:
        img = preprocess_image(all_images[idx], img_size)
        t0 = time.perf_counter()
        get_keras_embedding(keras_model, img)
        keras_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        get_int8_embedding(interp_int8, img)
        int8_times.append(time.perf_counter() - t0)

    km_ms, im_ms, speedup = plot_latency_comparison(keras_times, int8_times)
    metrics['latency_keras_ms'] = float(km_ms)
    metrics['latency_int8_ms'] = float(im_ms)
    metrics['latency_speedup'] = float(speedup)

    # --- Summary table ---
    print("\n[12/12] Summary table...")
    plot_summary_table(keras_m, int8_m, cos_sims, keras_sil, int8_sil,
                       keras_size_kb, int8_size_kb, km_ms, im_ms, speedup)

    # Save metrics
    metrics['class_names'] = class_names
    metrics['num_images'] = int(len(keras_emb))
    with open(os.path.join(SAVE_DIR, 'comparison_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)
    print("\n  ✓ comparison_metrics.json")

    # Final summary
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  Keras Size:         {keras_size_kb:.1f} KB")
    print(f"  INT8 Size:          {int8_size_kb:.1f} KB  ({ratio:.1f}× smaller)")
    print(f"  Keras Accuracy:     {keras_m['accuracy']:.4f}")
    print(f"  INT8  Accuracy:     {int8_m['accuracy']:.4f}  (drop: {metrics['accuracy_drop']:.4f})")
    print(f"  Keras Macro F1:     {k_f1:.4f}")
    print(f"  INT8  Macro F1:     {i_f1:.4f}")
    print(f"  Embedding Fidelity: {np.mean(cos_sims):.4f}")
    print(f"  Keras Silhouette:   {keras_sil:.4f}")
    print(f"  INT8  Silhouette:   {int8_sil:.4f}")
    print(f"  Keras Latency:      {km_ms:.2f} ms")
    print(f"  INT8  Latency:      {im_ms:.2f} ms  ({speedup:.2f}× faster)")
    print("=" * 70)
    print(f"  All plots saved to: {SAVE_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
