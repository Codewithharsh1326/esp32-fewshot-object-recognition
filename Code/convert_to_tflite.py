# convert_to_tflite.py
# Convert the trained embedding model to TFLite format for ESP32
# Enhanced with thesis-quality quantization analysis and ESP32 deployment metrics

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import json
import os
import time

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure sibling imports work regardless of CWD
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the custom layer so model loading works
from train_fewshot_embeddings import L2NormalizeLayer

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

# Resolve project root (parent of Code/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONVERSION_DIR = os.path.join(PROJECT_ROOT, "details", "conversion")
os.makedirs(CONVERSION_DIR, exist_ok=True)


def load_config():
    """Load model config from JSON."""
    config_path = os.path.join(PROJECT_ROOT, 'models', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    # Fallback defaults
    return {'img_size': 96, 'embedding_dim': 128, 'alpha': 0.35}


def get_calibration_images(data_dir, img_size, max_per_class=5):
    """Collect calibration images from the training data directory."""
    image_paths = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            class_images = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                class_images.extend(list(class_dir.glob(ext)))
            image_paths.extend(class_images[:max_per_class])
    
    if len(image_paths) == 0:
        raise ValueError(f"No calibration images found in {data_dir}")
    
    print(f"Using {len(image_paths)} calibration images from {data_dir}")
    return image_paths


def get_all_validation_images(data_dir, img_size):
    """Get ALL images from the data directory with their class labels."""
    image_paths = []
    labels = []
    class_names = []
    data_path = Path(data_dir)
    
    for class_idx, class_dir in enumerate(sorted(data_path.iterdir())):
        if class_dir.is_dir():
            class_names.append(class_dir.name)
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for img_path in class_dir.glob(ext):
                    image_paths.append(img_path)
                    labels.append(class_idx)
    
    return image_paths, np.array(labels), class_names


def preprocess_image_for_calibration(img_path, img_size):
    """Load and preprocess a single image for calibration."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_array = np.array(img).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_tflite_inference(interpreter, img_array):
    """Run inference on a TFLite interpreter and return dequantized embedding."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    inp = img_array.copy()
    # Handle quantized input
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zp = input_details[0]['quantization']
        inp = inp / input_scale + input_zp
        inp = np.clip(inp, 0, 255).astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zp = output_details[0]['quantization']
        output = (output.astype(np.float32) - output_zp) * output_scale
    
    # L2 normalize
    norm = np.linalg.norm(output)
    if norm > 0:
        output = output / norm
    
    return output.flatten()


# ==================== Thesis-Quality Plots ====================

def plot_model_size_comparison(keras_size_mb, f32_size_mb, int8_size_mb, save_dir):
    """Bar chart comparing model sizes with compression ratios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Original Keras\n(Float32)', 'TFLite\n(Float32)', 'TFLite\n(INT8 Quantized)']
    sizes = [keras_size_mb, f32_size_mb, int8_size_mb]
    colors = ['#1976D2', '#FB8C00', '#388E3C']
    
    bars = ax.bar(labels, sizes, color=colors, edgecolor='black', linewidth=0.7, width=0.5)
    
    # Add size labels on bars
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{size:.3f} MB\n({size*1024:.1f} KB)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add compression ratio annotations
    if keras_size_mb > 0:
        ratio_f32 = keras_size_mb / f32_size_mb if f32_size_mb > 0 else 0
        ratio_int8 = keras_size_mb / int8_size_mb if int8_size_mb > 0 else 0
        ax.annotate(f'{ratio_f32:.1f}× smaller', xy=(1, f32_size_mb/2),
                    fontsize=10, ha='center', color='white', fontweight='bold')
        ax.annotate(f'{ratio_int8:.1f}× smaller', xy=(2, int8_size_mb/2),
                    fontsize=10, ha='center', color='white', fontweight='bold')
    
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size Comparison: Original vs Converted')
    ax.set_ylim(0, max(sizes) * 1.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'model_size_comparison.png'))
    plt.close(fig)
    print("  ✓ model_size_comparison.png")


def plot_embedding_fidelity(cos_similarities, save_dir):
    """Histogram of cosine similarity between Float32 & INT8 embeddings."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(cos_similarities, bins=40, color='#7B1FA2', edgecolor='black',
            linewidth=0.5, alpha=0.8, density=True)
    
    mean_sim = np.mean(cos_similarities)
    median_sim = np.median(cos_similarities)
    min_sim = np.min(cos_similarities)
    
    ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.4f}')
    ax.axvline(median_sim, color='orange', linestyle='-.', linewidth=2, label=f'Median: {median_sim:.4f}')
    
    stats_text = (f'Mean: {mean_sim:.4f}\n'
                  f'Median: {median_sim:.4f}\n'
                  f'Min: {min_sim:.4f}\n'
                  f'Std: {np.std(cos_similarities):.4f}')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('Cosine Similarity (Float32 vs INT8 Embedding)')
    ax.set_ylabel('Density')
    ax.set_title('Embedding Fidelity: Float32 vs INT8 Quantized Model')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'embedding_fidelity.png'))
    plt.close(fig)
    print("  ✓ embedding_fidelity.png")


def plot_quantization_accuracy_impact(f32_metrics, int8_metrics, class_names, save_dir):
    """Side-by-side bar chart of accuracy metrics: Float32 vs INT8."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Overall metrics ---
    ax = axes[0]
    metric_names = ['Accuracy', 'Macro F1', 'Weighted F1']
    f32_vals = [f32_metrics['accuracy'], f32_metrics['macro_f1'], f32_metrics['weighted_f1']]
    int8_vals = [int8_metrics['accuracy'], int8_metrics['macro_f1'], int8_metrics['weighted_f1']]
    
    x = np.arange(len(metric_names))
    width = 0.3
    bars1 = ax.bar(x - width/2, f32_vals, width, label='Float32', color='#1976D2', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, int8_vals, width, label='INT8', color='#388E3C', edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars1, f32_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, int8_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics: Float32 vs INT8')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.legend()
    
    # --- Per-class accuracy ---
    ax = axes[1]
    x = np.arange(len(class_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, f32_metrics['per_class_acc'], width, label='Float32', color='#1976D2', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, int8_metrics['per_class_acc'], width, label='INT8', color='#388E3C', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy: Float32 vs INT8')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'quantization_accuracy_impact.png'))
    plt.close(fig)
    print("  ✓ quantization_accuracy_impact.png")


def plot_operator_distribution(interpreter, save_dir, model_label="INT8"):
    """Pie chart of TFLite operator types."""
    # Get op details from interpreter
    op_details = interpreter._get_ops_details()
    
    op_counts = {}
    for op in op_details:
        op_name = op.get('op_name', 'Unknown')
        op_counts[op_name] = op_counts.get(op_name, 0) + 1
    
    # Sort by count
    sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
    op_names = [op[0] for op in sorted_ops]
    counts = [op[1] for op in sorted_ops]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pie chart
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(op_names)))
    wedges, texts, autotexts = ax.pie(counts, labels=op_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       pctdistance=0.85, textprops={'fontsize': 9})
    ax.set_title(f'TFLite Operator Distribution ({model_label} Model)')
    
    # Bar chart
    ax = axes[1]
    y_pos = np.arange(len(op_names))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(op_names, fontsize=9)
    ax.set_xlabel('Number of Operators')
    ax.set_title(f'Operator Count ({model_label} Model)')
    ax.invert_yaxis()
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'operator_distribution.png'))
    plt.close(fig)
    print("  ✓ operator_distribution.png")
    
    return dict(sorted_ops)


def plot_layer_analysis(interpreter, save_dir, model_label="INT8"):
    """Bar chart of tensor sizes grouped by operator type."""
    tensor_details = interpreter.get_tensor_details()
    
    # Group tensor sizes by name pattern
    layer_sizes = {}
    for tensor in tensor_details:
        name = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        
        # Calculate size in bytes
        elem_size = np.dtype(dtype).itemsize
        num_elems = np.prod(shape) if len(shape) > 0 else 1
        size_bytes = num_elems * elem_size
        
        # Group by operator type
        if 'Conv2D' in name or 'conv' in name.lower():
            category = 'Conv2D'
        elif 'depthwise' in name.lower() or 'Depthwise' in name:
            category = 'DepthwiseConv'
        elif 'dense' in name.lower() or 'Dense' in name or 'fully_connected' in name.lower():
            category = 'Dense/FC'
        elif 'batch_norm' in name.lower() or 'BatchNorm' in name or 'bn' in name.lower():
            category = 'BatchNorm'
        elif 'bias' in name.lower():
            category = 'Bias'
        elif 'pool' in name.lower():
            category = 'Pooling'
        else:
            category = 'Other'
        
        layer_sizes[category] = layer_sizes.get(category, 0) + size_bytes
    
    # Sort
    sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[1], reverse=True)
    categories = [l[0] for l in sorted_layers]
    sizes_kb = [l[1] / 1024 for l in sorted_layers]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(categories)))
    bars = ax.bar(categories, sizes_kb, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, size in zip(bars, sizes_kb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{size:.1f} KB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    total_kb = sum(sizes_kb)
    ax.set_xlabel('Layer Type')
    ax.set_ylabel('Memory (KB)')
    ax.set_title(f'Memory Usage by Layer Type ({model_label} Model) — Total: {total_kb:.1f} KB')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'layer_analysis.png'))
    plt.close(fig)
    print("  ✓ layer_analysis.png")
    
    return dict(sorted_layers)


def plot_esp32_memory_budget(int8_size_bytes, tensor_arena_est, save_dir):
    """Visual table showing ESP32-S3 memory budget."""
    ESP32_S3_SRAM = 512 * 1024  # 512 KB
    ESP32_S3_PSRAM = 8 * 1024 * 1024  # 8 MB (common XIAO ESP32-S3)
    ESP32_S3_FLASH = 8 * 1024 * 1024  # 8 MB flash
    
    model_kb = int8_size_bytes / 1024
    arena_kb = tensor_arena_est / 1024
    remaining_sram = (ESP32_S3_SRAM - tensor_arena_est) / 1024
    remaining_flash = (ESP32_S3_FLASH - int8_size_bytes) / 1024
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Memory budget table ---
    ax = axes[0]
    ax.axis('off')
    
    table_data = [
        ['Model File Size', f'{model_kb:.1f} KB', f'{ESP32_S3_FLASH/1024:.0f} KB', f'{remaining_flash:.0f} KB'],
        ['Tensor Arena (est.)', f'{arena_kb:.1f} KB', f'{ESP32_S3_SRAM/1024:.0f} KB', f'{remaining_sram:.0f} KB'],
        ['PSRAM Available', '—', f'{ESP32_S3_PSRAM/1024:.0f} KB', f'{ESP32_S3_PSRAM/1024:.0f} KB'],
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Component', 'Required', 'Available', 'Headroom'],
        cellLoc='center', loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.2)
    
    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Color headroom column
    for i in range(1, 4):
        table[i, 3].set_facecolor('#C8E6C9')
        table[i, 3].set_text_props(fontweight='bold')
    
    ax.set_title('ESP32-S3 Memory Budget Analysis', fontsize=14, fontweight='bold', pad=30)
    
    # --- Visual stacked bar ---
    ax = axes[1]
    
    # Flash usage
    flash_used = int8_size_bytes / ESP32_S3_FLASH * 100
    sram_used = tensor_arena_est / ESP32_S3_SRAM * 100
    
    categories = ['Flash\n(Model Storage)', 'SRAM\n(Tensor Arena)']
    used = [flash_used, sram_used]
    free = [100 - flash_used, 100 - sram_used]
    
    x = np.arange(len(categories))
    width = 0.4
    
    bars1 = ax.bar(x, used, width, label='Used', color='#E53935', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, free, width, bottom=used, label='Available', color='#43A047', edgecolor='black', linewidth=0.5)
    
    for bar, pct in zip(bars1, used):
        if pct > 5:
            ax.text(bar.get_x() + bar.get_width()/2, pct/2,
                    f'{pct:.1f}%', ha='center', va='center', fontsize=12, 
                    fontweight='bold', color='white')
    for bar, pct, bottom in zip(bars2, free, used):
        ax.text(bar.get_x() + bar.get_width()/2, bottom + pct/2,
                f'{pct:.1f}%', ha='center', va='center', fontsize=12,
                fontweight='bold', color='white')
    
    ax.set_ylabel('Usage (%)')
    ax.set_title('ESP32-S3 Memory Utilization')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'esp32_memory_budget.png'))
    plt.close(fig)
    print("  ✓ esp32_memory_budget.png")


def plot_inference_latency(f32_times, int8_times, save_dir):
    """Boxplot comparing Float32 and INT8 inference latency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    ax = axes[0]
    data = [np.array(f32_times) * 1000, np.array(int8_times) * 1000]  # Convert to ms
    bp = ax.boxplot(data, labels=['Float32', 'INT8'], patch_artist=True,
                     boxprops=dict(linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('#BBDEFB')
    bp['boxes'][1].set_facecolor('#C8E6C9')
    
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Latency Distribution (PC Benchmark)')
    
    # Stats text
    f32_mean = np.mean(f32_times) * 1000
    int8_mean = np.mean(int8_times) * 1000
    speedup = f32_mean / int8_mean if int8_mean > 0 else 0
    stats_text = (f'Float32: {f32_mean:.2f} ms (mean)\n'
                  f'INT8: {int8_mean:.2f} ms (mean)\n'
                  f'Speedup: {speedup:.2f}×')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Bar chart of means
    ax = axes[1]
    means = [f32_mean, int8_mean]
    stds = [np.std(f32_times)*1000, np.std(int8_times)*1000]
    colors = ['#1976D2', '#388E3C']
    bars = ax.bar(['Float32', 'INT8'], means, yerr=stds, capsize=8,
                   color=colors, edgecolor='black', linewidth=0.7, width=0.4)
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[0] + 0.5,
                f'{mean:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title(f'Mean Inference Latency (Speedup: {speedup:.2f}×)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'inference_latency.png'))
    plt.close(fig)
    print("  ✓ inference_latency.png")
    
    return f32_mean, int8_mean, speedup


def plot_weight_distribution(model, interpreter_int8, save_dir):
    """Overlay histograms of float vs quantized weights."""
    # Float32 weights from Keras model
    float_weights = []
    for layer in model.layers:
        for w in layer.get_weights():
            if w.ndim >= 2:  # Skip biases
                float_weights.extend(w.flatten().tolist())
    
    # INT8 weights from interpreter
    int8_weights = []
    tensor_details = interpreter_int8.get_tensor_details()
    for tensor in tensor_details:
        if tensor['dtype'] == np.int8:
            try:
                data = interpreter_int8.get_tensor(tensor['index'])
                if data.ndim >= 2:
                    int8_weights.extend(data.flatten().tolist())
            except Exception:
                pass
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Float32 distribution
    ax = axes[0]
    ax.hist(float_weights, bins=100, color='#1976D2', edgecolor='black', linewidth=0.3, alpha=0.8, density=True)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Float32 Weight Distribution (n={len(float_weights):,})')
    ax.text(0.02, 0.95, f'Mean: {np.mean(float_weights):.4f}\nStd: {np.std(float_weights):.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # INT8 distribution
    ax = axes[1]
    if int8_weights:
        ax.hist(int8_weights, bins=range(-128, 129, 1), color='#388E3C', edgecolor='black', 
                linewidth=0.1, alpha=0.8, density=True)
        ax.set_xlabel('Quantized Weight Value (INT8)')
        ax.set_ylabel('Density')
        ax.set_title(f'INT8 Quantized Weight Distribution (n={len(int8_weights):,})')
        ax.text(0.02, 0.95, f'Mean: {np.mean(int8_weights):.1f}\nStd: {np.std(int8_weights):.1f}\nRange: [-128, 127]',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'No INT8 weights found', ha='center', va='center', fontsize=14)
    
    plt.suptitle('Weight Distribution: Before vs After Quantization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'weight_distribution.png'))
    plt.close(fig)
    print("  ✓ weight_distribution.png")


def plot_quantization_error_heatmap(f32_embeddings, int8_embeddings, sample_labels,
                                     class_names, save_dir, max_samples=50):
    """Heatmap of per-dimension quantization error for a subset of samples."""
    n = min(max_samples, len(f32_embeddings))
    errors = np.abs(f32_embeddings[:n] - int8_embeddings[:n])
    
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.15 + 2)))
    
    # Sort by class for visual grouping
    sort_idx = np.argsort(sample_labels[:n])
    errors_sorted = errors[sort_idx]
    labels_sorted = sample_labels[:n][sort_idx]
    
    y_labels = [f'{class_names[l]}' for l in labels_sorted]
    
    sns.heatmap(errors_sorted, cmap='YlOrRd', ax=ax, 
                yticklabels=y_labels if n <= 30 else False,
                xticklabels=False, cbar_kws={'label': 'Absolute Error'})
    ax.set_xlabel(f'Embedding Dimension (1–{errors.shape[1]})')
    ax.set_ylabel('Sample (grouped by class)')
    ax.set_title('Per-Dimension Quantization Error: Float32 vs INT8 Embeddings')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'quantization_error_heatmap.png'))
    plt.close(fig)
    print("  ✓ quantization_error_heatmap.png")


def compute_classification_metrics(embeddings, labels, class_names):
    """Compute nearest-centroid classification metrics from embeddings."""
    from sklearn.metrics import precision_recall_fscore_support
    
    num_classes = len(class_names)
    
    # Compute centroids
    centroids = np.zeros((num_classes, embeddings.shape[1]))
    for i in range(num_classes):
        mask = labels == i
        if np.any(mask):
            c = embeddings[mask].mean(axis=0)
            centroids[i] = c / (np.linalg.norm(c) + 1e-8)
    
    # Predict via nearest centroid (cosine similarity)
    similarities = np.dot(embeddings, centroids.T)
    predictions = np.argmax(similarities, axis=1)
    
    accuracy = np.mean(predictions == labels)
    
    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        mask = labels == i
        if np.sum(mask) > 0:
            per_class_acc.append(np.mean(predictions[mask] == i))
        else:
            per_class_acc.append(0.0)
    
    return {
        'accuracy': accuracy,
        'macro_f1': f1,
        'weighted_f1': wf1,
        'macro_precision': p,
        'macro_recall': r,
        'per_class_acc': per_class_acc,
        'predictions': predictions
    }


# ==================== Main Conversion ====================

def convert_to_tflite():
    print("=" * 70)
    print("Converting Embedding Model to TFLite for ESP32")
    print("=" * 70)
    
    # Load config
    cfg = load_config()
    img_size = cfg.get('img_size', 96)
    print(f"Image size: {img_size}x{img_size}")
    print(f"Embedding dim: {cfg.get('embedding_dim', 128)}")
    
    # Try loading .keras first (reliable Keras format), then SavedModel
    keras_model_path = os.path.join(PROJECT_ROOT, 'models', 'embedding_model.keras')
    saved_model_path = os.path.join(PROJECT_ROOT, 'models', 'embedding_model_savedmodel')
    
    model = None
    use_saved_model_converter = False
    
    if os.path.exists(keras_model_path):
        print(f"\nLoading model from: {keras_model_path}")
        model = tf.keras.models.load_model(
            keras_model_path,
            custom_objects={'L2NormalizeLayer': L2NormalizeLayer}
        )
        print("✓ Model loaded from .keras file")
    elif os.path.exists(saved_model_path):
        # SavedModel from model.export() is a serving format — use from_saved_model converter
        print(f"\nUsing SavedModel directly for conversion: {saved_model_path}")
        use_saved_model_converter = True
    else:
        raise FileNotFoundError(
            f"No model found! Expected either:\n"
            f"  {keras_model_path}\n"
            f"  {saved_model_path}\n"
            f"Run train_fewshot_embeddings.py first."
        )
    
    if model is not None:
        model.summary()
    
    # Keras model size
    keras_size_bytes = os.path.getsize(keras_model_path) if os.path.exists(keras_model_path) else 0
    keras_size_mb = keras_size_bytes / (1024 * 1024)
    
    # Get calibration images
    cal_images = get_calibration_images(os.path.join(PROJECT_ROOT, 'data'), img_size)
    
    # ---- 1. Float32 TFLite (for PC inference / debugging) ----
    print("\n--- Converting to Float32 TFLite ---")
    if use_saved_model_converter:
        converter_f32 = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    else:
        converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_f32.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_f32 = converter_f32.convert()
    
    f32_path = os.path.join(PROJECT_ROOT, 'models', 'embedding_model_float32.tflite')
    with open(f32_path, 'wb') as f:
        f.write(tflite_f32)
    
    f32_size = len(tflite_f32) / (1024 * 1024)
    print(f"✓ Float32 model saved: {f32_path} ({f32_size:.2f} MB)")
    
    # ---- 2. INT8 TFLite (for ESP32 deployment) ----
    print("\n--- Converting to INT8 TFLite ---")
    
    def representative_dataset_gen():
        for img_path in cal_images:
            try:
                yield [preprocess_image_for_calibration(img_path, img_size)]
            except Exception as e:
                print(f"  ⚠ Skipping calibration image {img_path}: {e}")
                continue
    
    if use_saved_model_converter:
        converter_int8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    else:
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset_gen
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.uint8
    converter_int8.inference_output_type = tf.int8
    
    tflite_int8 = converter_int8.convert()
    
    int8_path = os.path.join(PROJECT_ROOT, 'models', 'embedding_model_int8_esp32.tflite')
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8)
    
    int8_size = len(tflite_int8) / (1024 * 1024)
    int8_size_bytes = len(tflite_int8)
    print(f"✓ INT8 model saved: {int8_path} ({int8_size:.2f} MB)")
    
    # ---- 3. Verify both models ----
    print("\n--- Verifying models ---")
    
    for label, model_path in [("Float32", f32_path), ("INT8", int8_path)]:
        print(f"\n[{label}] {model_path}")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Input:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
        print(f"  Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")
        
        # Test with a sample image
        if len(cal_images) > 0:
            try:
                img_array = preprocess_image_for_calibration(cal_images[0], img_size)
                
                # Handle quantization for INT8
                if input_details[0]['dtype'] == np.uint8:
                    input_scale, input_zp = input_details[0]['quantization']
                    img_array = img_array / input_scale + input_zp
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                
                # Dequantize if needed
                if output_details[0]['dtype'] == np.int8:
                    output_scale, output_zp = output_details[0]['quantization']
                    output = (output.astype(np.float32) - output_zp) * output_scale
                
                # Normalize
                norm = np.linalg.norm(output)
                if norm > 0:
                    output = output / norm
                
                print(f"  ✓ Test inference OK! Embedding shape: {output.shape}, norm: {np.linalg.norm(output):.4f}")
            except Exception as e:
                print(f"  ⚠ Test inference failed: {e}")
    
    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Conversion Summary:")
    print(f"  Float32: {f32_path} ({f32_size:.2f} MB) — for PC inference")
    print(f"  INT8:    {int8_path} ({int8_size:.2f} MB) — for ESP32 deployment")
    print("=" * 70)
    
    # ======================================================================
    #  THESIS-QUALITY CONVERSION ANALYSIS
    # ======================================================================
    print("\n" + "=" * 70)
    print("  THESIS-QUALITY CONVERSION ANALYSIS")
    print("=" * 70)
    
    metrics = {}
    
    # --- 1. Model Size Comparison ---
    print("\n[1/9] Model size comparison...")
    plot_model_size_comparison(keras_size_mb, f32_size, int8_size, CONVERSION_DIR)
    metrics['keras_size_mb'] = keras_size_mb
    metrics['float32_tflite_size_mb'] = f32_size
    metrics['int8_tflite_size_mb'] = int8_size
    metrics['compression_ratio_f32'] = keras_size_mb / f32_size if f32_size > 0 else 0
    metrics['compression_ratio_int8'] = keras_size_mb / int8_size if int8_size > 0 else 0
    metrics['int8_size_kb'] = int8_size_bytes / 1024
    
    # --- 2. Load interpreters ---
    print("\n[2/9] Loading interpreters for analysis...")
    interp_f32 = tf.lite.Interpreter(model_path=f32_path)
    interp_f32.allocate_tensors()
    interp_int8 = tf.lite.Interpreter(model_path=int8_path)
    interp_int8.allocate_tensors()
    
    # --- 3. Get all images for thorough evaluation ---
    print("\n[3/9] Computing embeddings on all images (Float32 & INT8)...")
    all_images, all_labels, class_names = get_all_validation_images(os.path.join(PROJECT_ROOT, 'data'), img_size)
    
    f32_embeddings = []
    int8_embeddings = []
    valid_labels = []
    
    for img_path, label in zip(all_images, all_labels):
        try:
            img_array = preprocess_image_for_calibration(img_path, img_size)
            
            # Float32 embedding
            emb_f32 = run_tflite_inference(interp_f32, img_array)
            # INT8 embedding
            emb_int8 = run_tflite_inference(interp_int8, img_array)
            
            f32_embeddings.append(emb_f32)
            int8_embeddings.append(emb_int8)
            valid_labels.append(label)
        except Exception as e:
            pass
    
    f32_embeddings = np.array(f32_embeddings)
    int8_embeddings = np.array(int8_embeddings)
    valid_labels = np.array(valid_labels)
    
    print(f"  Computed embeddings for {len(f32_embeddings)} images")
    
    # --- 4. Embedding Fidelity ---
    print("\n[4/9] Embedding fidelity analysis...")
    cos_similarities = []
    for e_f32, e_int8 in zip(f32_embeddings, int8_embeddings):
        cos_sim = np.dot(e_f32, e_int8) / (np.linalg.norm(e_f32) * np.linalg.norm(e_int8) + 1e-8)
        cos_similarities.append(cos_sim)
    cos_similarities = np.array(cos_similarities)
    
    plot_embedding_fidelity(cos_similarities, CONVERSION_DIR)
    metrics['embedding_fidelity_mean'] = float(np.mean(cos_similarities))
    metrics['embedding_fidelity_median'] = float(np.median(cos_similarities))
    metrics['embedding_fidelity_min'] = float(np.min(cos_similarities))
    metrics['embedding_fidelity_std'] = float(np.std(cos_similarities))
    print(f"  Mean cosine similarity: {np.mean(cos_similarities):.4f}")
    
    # --- 5. Quantization Accuracy Impact ---
    print("\n[5/9] Quantization accuracy impact...")
    f32_metrics = compute_classification_metrics(f32_embeddings, valid_labels, class_names)
    int8_metrics = compute_classification_metrics(int8_embeddings, valid_labels, class_names)
    
    plot_quantization_accuracy_impact(f32_metrics, int8_metrics, class_names, CONVERSION_DIR)
    
    metrics['float32_accuracy'] = float(f32_metrics['accuracy'])
    metrics['int8_accuracy'] = float(int8_metrics['accuracy'])
    metrics['accuracy_drop'] = float(f32_metrics['accuracy'] - int8_metrics['accuracy'])
    metrics['float32_macro_f1'] = float(f32_metrics['macro_f1'])
    metrics['int8_macro_f1'] = float(int8_metrics['macro_f1'])
    metrics['f1_drop'] = float(f32_metrics['macro_f1'] - int8_metrics['macro_f1'])
    print(f"  Float32 Accuracy: {f32_metrics['accuracy']:.4f},  INT8 Accuracy: {int8_metrics['accuracy']:.4f}")
    print(f"  Accuracy drop: {metrics['accuracy_drop']:.4f}")
    
    # --- 6. Operator Distribution ---
    print("\n[6/9] Operator distribution analysis...")
    try:
        op_counts = plot_operator_distribution(interp_int8, CONVERSION_DIR, "INT8")
        metrics['operator_counts'] = {k: int(v) for k, v in op_counts.items()}
    except Exception as e:
        print(f"  ⚠ Operator distribution failed: {e}")
    
    # --- 7. Layer Analysis ---
    print("\n[7/9] Layer memory analysis...")
    layer_sizes = plot_layer_analysis(interp_int8, CONVERSION_DIR, "INT8")
    metrics['layer_sizes_bytes'] = {k: int(v) for k, v in layer_sizes.items()}
    
    # Estimate tensor arena
    tensor_details = interp_int8.get_tensor_details()
    max_tensor_size = max(np.prod(t['shape']) * np.dtype(t['dtype']).itemsize for t in tensor_details)
    total_tensor_size = sum(np.prod(t['shape']) * np.dtype(t['dtype']).itemsize for t in tensor_details)
    tensor_arena_estimate = int(total_tensor_size * 1.2)  # 20% headroom
    
    # --- 8. ESP32 Memory Budget ---
    print("\n[8/9] ESP32-S3 memory budget analysis...")
    plot_esp32_memory_budget(int8_size_bytes, tensor_arena_estimate, CONVERSION_DIR)
    metrics['tensor_arena_estimate_kb'] = tensor_arena_estimate / 1024
    metrics['esp32_s3_sram_kb'] = 512
    metrics['esp32_s3_flash_kb'] = 8192
    metrics['sram_utilization_pct'] = (tensor_arena_estimate / (512 * 1024)) * 100
    metrics['flash_utilization_pct'] = (int8_size_bytes / (8 * 1024 * 1024)) * 100
    
    # --- 9. Inference Latency ---
    print("\n[9/9] Inference latency benchmarking...")
    NUM_RUNS = 50
    f32_times = []
    int8_times = []
    
    # Warm up
    test_img = preprocess_image_for_calibration(all_images[0], img_size)
    for _ in range(5):
        run_tflite_inference(interp_f32, test_img)
        run_tflite_inference(interp_int8, test_img)
    
    # Benchmark
    sample_indices = np.random.choice(len(all_images), min(NUM_RUNS, len(all_images)), replace=True)
    for idx in sample_indices:
        img_array = preprocess_image_for_calibration(all_images[idx], img_size)
        
        t0 = time.perf_counter()
        run_tflite_inference(interp_f32, img_array)
        f32_times.append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        run_tflite_inference(interp_int8, img_array)
        int8_times.append(time.perf_counter() - t0)
    
    f32_mean_ms, int8_mean_ms, speedup = plot_inference_latency(f32_times, int8_times, CONVERSION_DIR)
    metrics['latency_float32_mean_ms'] = float(f32_mean_ms)
    metrics['latency_int8_mean_ms'] = float(int8_mean_ms)
    metrics['latency_speedup'] = float(speedup)
    
    # --- Bonus: Weight Distribution ---
    print("\n[Bonus] Weight distribution analysis...")
    if model is not None:
        plot_weight_distribution(model, interp_int8, CONVERSION_DIR)
    
    # --- Bonus: Quantization Error Heatmap ---
    print("[Bonus] Quantization error heatmap...")
    plot_quantization_error_heatmap(f32_embeddings, int8_embeddings, valid_labels, class_names, CONVERSION_DIR)
    
    # --- Save Metrics JSON ---
    metrics['class_names'] = class_names
    metrics['num_images_evaluated'] = int(len(f32_embeddings))
    
    with open(os.path.join(CONVERSION_DIR, 'conversion_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)
    print("\n  ✓ conversion_metrics.json")
    
    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("  CONVERSION ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Keras Model:           {keras_size_mb:.3f} MB")
    print(f"  Float32 TFLite:        {f32_size:.3f} MB  ({metrics['compression_ratio_f32']:.1f}× smaller)")
    print(f"  INT8 TFLite:           {int8_size:.3f} MB  ({metrics['compression_ratio_int8']:.1f}× smaller)")
    print(f"  ──────────────────────────────────")
    print(f"  Float32 Accuracy:      {f32_metrics['accuracy']:.4f}")
    print(f"  INT8 Accuracy:         {int8_metrics['accuracy']:.4f}")
    print(f"  Accuracy Drop:         {metrics['accuracy_drop']:.4f}")
    print(f"  Embedding Fidelity:    {np.mean(cos_similarities):.4f} (cosine sim)")
    print(f"  ──────────────────────────────────")
    print(f"  Latency Float32:       {f32_mean_ms:.2f} ms")
    print(f"  Latency INT8:          {int8_mean_ms:.2f} ms  ({speedup:.2f}× faster)")
    print(f"  ──────────────────────────────────")
    print(f"  ESP32-S3 Flash Used:   {metrics['flash_utilization_pct']:.1f}%")
    print(f"  ESP32-S3 SRAM Used:    {metrics['sram_utilization_pct']:.1f}% (tensor arena est.)")
    print("=" * 70)
    print(f"  All plots saved to: {CONVERSION_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    convert_to_tflite()