# visualize_layers.py
# Visualize intermediate layer activations (feature maps) of the embedding model
# Shows what the CNN "sees" at each layer — thesis-quality visualization
# Saves all outputs to details/feature_maps/

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure sibling imports work regardless of CWD
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_fewshot_embeddings import L2NormalizeLayer

# ==================== Config ====================
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# Resolve project root (parent of Code/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAVE_DIR = os.path.join(PROJECT_ROOT, "details", "feature_maps")
os.makedirs(SAVE_DIR, exist_ok=True)

KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "embedding_model.keras")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'models', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {'img_size': 96, 'embedding_dim': 128}


def pick_sample_image(data_dir):
    """Pick one image from each class for visualization."""
    data_path = Path(data_dir)
    samples = []
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                imgs = list(class_dir.glob(ext))
                if imgs:
                    samples.append((imgs[0], class_dir.name))
                    break
    return samples


def preprocess_image(img_path, img_size):
    """Load and preprocess image; also return original for display."""
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)
    original_display = np.array(img_resized).astype(np.uint8)

    img_array = np.array(img_resized).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, original_display


def get_key_layers(model):
    """Extract key convolutional layers from the model for visualization."""
    key_layers = []

    # Walk through all layers, including nested models (MobileNetV2 backbone)
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            # This is the MobileNetV2 backbone — dig into it
            backbone = layer
            for sub_layer in backbone.layers:
                name = sub_layer.name
                # Select interesting layers: conv outputs, expanded convs, block outputs
                if any(keyword in name for keyword in [
                    'Conv1',            # First conv
                    'block_1_expand',   # Early block
                    'block_3_expand',   # Mid-early
                    'block_6_expand',   # Middle
                    'block_6_project',  
                    'block_10_expand',  # Mid-late
                    'block_13_expand',  # Late
                    'block_13_project',
                    'block_16_project', # Last block
                    'Conv_1',           # Final conv
                    'out_relu',         # Output activation
                ]):
                    # Only pick layers with 4D output (feature maps)
                    try:
                        out_shape = sub_layer.output.shape
                        if len(out_shape) == 4:
                            key_layers.append((backbone, sub_layer))
                    except Exception:
                        pass
    return key_layers


def visualize_single_layer(activation, layer_name, n_filters=16, save_path=None):
    """Visualize up to n_filters feature maps from a single layer's activation."""
    # activation shape: (1, H, W, C)
    act = activation[0]  # Remove batch dim -> (H, W, C)
    n_channels = act.shape[-1]
    n_show = min(n_filters, n_channels)

    # Grid layout
    cols = min(8, n_show)
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i, j]
            if idx < n_show:
                feature_map = act[:, :, idx]
                ax.imshow(feature_map, cmap='viridis', aspect='auto')
                ax.set_title(f'F{idx}', fontsize=7, pad=2)
            ax.axis('off')

    spatial = f'{act.shape[0]}×{act.shape[1]}'
    fig.suptitle(f'{layer_name}\n({spatial}, {n_channels} channels — showing {n_show})',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)


def create_layer_progression(original_img, activations_info, save_dir):
    """Create a single overview figure showing the progression through key layers."""
    n_layers = len(activations_info)
    n_show = min(4, activations_info[0][1].shape[-1])  # Filters to show per layer

    fig, axes = plt.subplots(n_layers + 1, n_show + 1, 
                              figsize=(n_show * 2.5 + 3, (n_layers + 1) * 2.2))

    # First row: original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Input Image', fontsize=9, fontweight='bold')
    axes[0, 0].axis('off')
    for j in range(1, n_show + 1):
        axes[0, j].axis('off')

    # Layer rows
    for i, (name, act) in enumerate(activations_info):
        row = i + 1
        a = act[0]  # (H, W, C)

        # Label
        axes[row, 0].text(0.5, 0.5, f'{name}\n{a.shape[0]}×{a.shape[1]}×{a.shape[2]}',
                          ha='center', va='center', fontsize=7, fontweight='bold',
                          transform=axes[row, 0].transAxes,
                          bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
        axes[row, 0].axis('off')

        # Feature maps
        for j in range(n_show):
            ax = axes[row, j + 1]
            if j < a.shape[-1]:
                ax.imshow(a[:, :, j], cmap='viridis', aspect='auto')
                ax.set_title(f'Ch {j}', fontsize=7, pad=1)
            ax.axis('off')

    plt.suptitle('Layer-by-Layer Feature Map Progression', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'layer_progression_overview.png'))
    plt.close(fig)
    print("  ✓ layer_progression_overview.png")


def create_activation_magnitude_plot(activations_info, save_dir):
    """Plot mean activation magnitude at each layer — shows information flow."""
    layer_names = [name for name, _ in activations_info]
    mean_activations = [np.mean(np.abs(act)) for _, act in activations_info]
    max_activations = [np.max(np.abs(act)) for _, act in activations_info]
    spatial_sizes = [f'{act.shape[1]}×{act.shape[2]}' for _, act in activations_info]
    num_channels = [act.shape[3] for _, act in activations_info]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Mean activation
    ax = axes[0]
    x = range(len(layer_names))
    ax.bar(x, mean_activations, color='#1976D2', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.set_ylabel('Mean |Activation|')
    ax.set_title('Mean Activation Magnitude per Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)

    # Spatial size & channels
    ax = axes[1]
    ax.bar(x, num_channels, color='#F57C00', edgecolor='black', linewidth=0.5, alpha=0.8)
    for xi, (s, c) in enumerate(zip(spatial_sizes, num_channels)):
        ax.text(xi, c + 2, f'{s}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_ylabel('Number of Channels')
    ax.set_title('Feature Map Dimensions per Layer (spatial size annotated)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'activation_magnitude.png'))
    plt.close(fig)
    print("  ✓ activation_magnitude.png")


def create_top_activations_grid(original_img, activations_info, save_dir):
    """For each layer, show the single most activated feature map (strongest response)."""
    n = len(activations_info)
    cols = min(6, n)
    rows = int(np.ceil(n / cols)) + 1  # +1 for input row

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = axes[np.newaxis, :]

    # Input image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Input', fontsize=9, fontweight='bold')
    axes[0, 0].axis('off')
    for j in range(1, cols):
        axes[0, j].axis('off')

    # Max-activated map per layer
    for idx, (name, act) in enumerate(activations_info):
        r = (idx // cols) + 1
        c = idx % cols
        a = act[0]
        # Find the channel with highest mean activation
        channel_means = np.mean(a, axis=(0, 1))
        best_ch = np.argmax(channel_means)
        axes[r, c].imshow(a[:, :, best_ch], cmap='inferno', aspect='auto')
        short_name = name.replace('expanded_conv_', 'blk').replace('_expand_relu', '_exp')
        short_name = short_name.replace('_depthwise_relu', '_dw').replace('_project_BN', '_proj')
        axes[r, c].set_title(f'{short_name}\n{a.shape[0]}×{a.shape[1]}', fontsize=7, pad=2)
        axes[r, c].axis('off')

    # Hide unused axes
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j > 0:
                axes[i, j].axis('off')
            elif i > 0:
                idx = (i - 1) * cols + j
                if idx >= n:
                    axes[i, j].axis('off')

    plt.suptitle('Strongest Feature Map at Each Layer', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'top_activations_grid.png'))
    plt.close(fig)
    print("  ✓ top_activations_grid.png")


def main():
    print("=" * 70)
    print("  INTERMEDIATE LAYER VISUALIZATION")
    print("=" * 70)

    cfg = load_config()
    img_size = cfg.get('img_size', 96)

    # Load model
    print("\n[1/6] Loading model...")
    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH, custom_objects={'L2NormalizeLayer': L2NormalizeLayer}
    )
    print(f"  ✓ Loaded: {KERAS_MODEL_PATH}")

    # Find the MobileNetV2 backbone
    backbone = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is None:
        print("  ⚠ Could not find backbone model. Exiting.")
        return

    print(f"  Backbone: {backbone.name} ({len(backbone.layers)} layers)")

    # Select key layers to visualize
    print("\n[2/6] Selecting key layers...")
    target_keywords = [
        'Conv1_relu',             # First convolution
        'block_1_expand_relu',    # Early expansion
        'block_1_depthwise_relu', # Early depthwise
        'block_3_expand_relu',    # Mid-early
        'block_6_expand_relu',    # Middle
        'block_6_depthwise_relu', # Middle depthwise
        'block_10_expand_relu',   # Mid-late
        'block_13_expand_relu',   # Late
        'block_13_depthwise_relu',# Late depthwise
        'block_16_project_BN',    # Final block projection
        'Conv_1_bn',              # Final convolution (before pooling)
        'out_relu',               # Output ReLU
    ]

    # Collect available target layers
    selected_layers = []
    for keyword in target_keywords:
        for sub_layer in backbone.layers:
            if sub_layer.name == keyword:
                try:
                    _ = sub_layer.output.shape
                    selected_layers.append(sub_layer)
                except Exception:
                    pass
                break

    # If exact matches fail, pick layers with 4D output heuristically
    if len(selected_layers) < 3:
        print("  Falling back to heuristic layer selection...")
        for sub_layer in backbone.layers:
            try:
                out_shape = sub_layer.output.shape
                if len(out_shape) == 4 and 'relu' in sub_layer.name.lower():
                    selected_layers.append(sub_layer)
            except Exception:
                pass

    # Deduplicate while preserving order
    seen = set()
    unique_layers = []
    for l in selected_layers:
        if l.name not in seen:
            seen.add(l.name)
            unique_layers.append(l)
    selected_layers = unique_layers

    print(f"  Selected {len(selected_layers)} layers for visualization:")
    for l in selected_layers:
        print(f"    - {l.name} → output shape: {l.output.shape}")

    # Build intermediate model
    print("\n[3/6] Building intermediate activation model...")
    intermediate_outputs = [l.output for l in selected_layers]
    intermediate_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=intermediate_outputs
    )

    # Pick a sample image from each class
    print("\n[4/6] Processing sample images...")
    samples = pick_sample_image(DATA_DIR)
    print(f"  Found {len(samples)} classes")

    # Use first sample for detailed per-layer visualization
    main_img_path, main_class = samples[0]
    print(f"\n  Using sample image: {main_img_path.name} (class: {main_class})")

    img_input, original_display = preprocess_image(main_img_path, img_size)

    # Get activations
    activations = intermediate_model.predict(img_input, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]

    layer_names = [l.name for l in selected_layers]
    activations_info = list(zip(layer_names, activations))

    # --- Save original input ---
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(original_display)
    ax.set_title(f'Input Image: "{main_class}"', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'input_image.png'))
    plt.close(fig)
    print("  ✓ input_image.png")

    # --- Individual layer visualizations ---
    print("\n[5/6] Generating per-layer feature map plots...")
    for i, (name, act) in enumerate(activations_info):
        safe_name = name.replace('/', '_')
        save_path = os.path.join(SAVE_DIR, f'layer_{i:02d}_{safe_name}.png')
        visualize_single_layer(act, name, n_filters=32, save_path=save_path)
        print(f"  ✓ layer_{i:02d}_{safe_name}.png  ({act.shape[1]}×{act.shape[2]}×{act.shape[3]})")

    # --- Overview plots ---
    print("\n[6/6] Generating overview plots...")
    create_layer_progression(original_display, activations_info, SAVE_DIR)
    create_activation_magnitude_plot(activations_info, SAVE_DIR)
    create_top_activations_grid(original_display, activations_info, SAVE_DIR)

    # --- Multi-class comparison: show top activation for one layer across all classes ---
    print("\n[Bonus] Multi-class feature comparison...")
    # Use a mid-level layer
    mid_idx = len(selected_layers) // 2
    mid_layer_name = layer_names[mid_idx]

    n_classes = len(samples)
    fig, axes = plt.subplots(2, n_classes, figsize=(n_classes * 2.2, 5))

    for col, (img_path, cls_name) in enumerate(samples):
        inp, orig = preprocess_image(img_path, img_size)
        acts = intermediate_model.predict(inp, verbose=0)
        if not isinstance(acts, list):
            acts = [acts]

        # Original
        axes[0, col].imshow(orig)
        axes[0, col].set_title(cls_name, fontsize=9, fontweight='bold')
        axes[0, col].axis('off')

        # Mid-layer top activation
        a = acts[mid_idx][0]
        best_ch = np.argmax(np.mean(a, axis=(0, 1)))
        axes[1, col].imshow(a[:, :, best_ch], cmap='inferno', aspect='auto')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Input', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel(f'{mid_layer_name}', fontsize=8, fontweight='bold')

    plt.suptitle(f'Feature Response Comparison Across Classes\n(Layer: {mid_layer_name})',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'multi_class_feature_comparison.png'))
    plt.close(fig)
    print("  ✓ multi_class_feature_comparison.png")

    # Summary
    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"  Layers visualized: {len(selected_layers)}")
    print(f"  Sample image:      {main_img_path.name} ({main_class})")
    n_files = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.png')])
    print(f"  Total plots:       {n_files}")
    print(f"  Saved to:          {SAVE_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
