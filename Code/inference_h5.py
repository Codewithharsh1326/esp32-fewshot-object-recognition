# inference_h5.py
# PC inference script using the Keras .h5 embedding model
# Workflow: Register object (3 photos) → Compute reference embedding → Query → Compare
#
# Usage examples:
#   Interactive mode:   python inference_h5.py
#   CLI mode:           python inference_h5.py --reference img1.jpg img2.jpg img3.jpg --query query.jpg
#   With webcam:        python inference_h5.py --webcam
#   Custom threshold:   python inference_h5.py --threshold 0.65

import tensorflow as tf
import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image

# Ensure sibling imports work regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom layer for model loading
from train_fewshot_embeddings import L2NormalizeLayer


# ==================== Configuration ====================
# Resolve project root (parent of Code/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "embedding_model.keras")
DEFAULT_SAVEDMODEL_PATH = os.path.join(PROJECT_ROOT, "models", "embedding_model_savedmodel")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "config.json")
DEFAULT_THRESHOLD = 0.70  # Cosine similarity threshold


# ==================== Model Loading ====================
def load_embedding_model(model_path=None):
    """Load the trained embedding model (.h5 or SavedModel)."""
    
    # Try SavedModel first, then .h5
    if model_path and os.path.exists(model_path):
        paths_to_try = [model_path]
    else:
        paths_to_try = [DEFAULT_SAVEDMODEL_PATH, DEFAULT_MODEL_PATH]
    
    model = None
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"Loading model from: {path}")
            try:
                model = tf.keras.models.load_model(
                    path,
                    custom_objects={'L2NormalizeLayer': L2NormalizeLayer}
                )
                print(f"✓ Model loaded successfully")
                break
            except Exception as e:
                print(f"⚠ Failed to load {path}: {e}")
                continue
    
    if model is None:
        raise FileNotFoundError(
            f"No model found! Tried: {paths_to_try}\n"
            f"Run train_fewshot_embeddings.py first."
        )
    
    return model


def load_config():
    """Load model configuration."""
    if os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {'img_size': 96, 'embedding_dim': 128}


# ==================== Image Processing ====================
def preprocess_image(image, img_size):
    """Preprocess a PIL Image for the model."""
    image = image.convert('RGB')
    image = image.resize((img_size, img_size), Image.BILINEAR)
    img_array = np.array(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_and_preprocess(image_path, img_size):
    """Load an image from path and preprocess it."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path)
    return preprocess_image(image, img_size)


def capture_from_webcam():
    """Capture a single frame from webcam."""
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is required for webcam capture.")
        print("Install it with: pip install opencv-python")
        sys.exit(1)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    
    print("Webcam opened. Press SPACE to capture, Q to quit.")
    
    captured_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Capture (SPACE=capture, Q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Convert BGR (OpenCV) to RGB (PIL)
            captured_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print("✓ Image captured!")
            break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_image


# ==================== Embedding Computation ====================
def compute_embedding(model, img_array):
    """Compute embedding for a preprocessed image."""
    embedding = model.predict(img_array, verbose=0)
    # Normalize (should already be normalized by L2NormalizeLayer, but ensure)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.flatten()


def compute_average_embedding(model, images, img_size):
    """Compute the average embedding from multiple images."""
    embeddings = []
    for i, img in enumerate(images):
        if isinstance(img, str) or isinstance(img, Path):
            img_array = load_and_preprocess(str(img), img_size)
        elif isinstance(img, Image.Image):
            img_array = preprocess_image(img, img_size)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        emb = compute_embedding(model, img_array)
        embeddings.append(emb)
        print(f"  Reference image {i+1}: embedding computed (norm={np.linalg.norm(emb):.4f})")
    
    # Average and re-normalize
    avg_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    
    print(f"  → Average reference embedding computed (norm={np.linalg.norm(avg_embedding):.4f})")
    return avg_embedding


# ==================== Similarity ====================
def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(emb1, emb2))


def compare_embeddings(ref_embedding, query_embedding, threshold):
    """Compare query against reference and return verdict."""
    similarity = cosine_similarity(ref_embedding, query_embedding)
    is_same = similarity >= threshold
    
    return {
        'similarity': similarity,
        'threshold': threshold,
        'is_same': is_same,
        'verdict': 'SAME OBJECT ✓' if is_same else 'DIFFERENT OBJECT ✗'
    }


# ==================== Interactive Mode ====================
def interactive_mode(model, img_size, threshold, use_webcam=False):
    """Interactive loop: register reference → detect query objects."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE FEW-SHOT OBJECT DETECTION")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(f"Input mode: {'webcam' if use_webcam else 'file paths'}")
    print("=" * 60)
    
    while True:
        # === PHASE 1: Register reference object ===
        print("\n" + "-" * 60)
        print("PHASE 1: REGISTER REFERENCE OBJECT")
        print("Provide 3 images of the object you want to detect.")
        print("-" * 60)
        
        reference_images = []
        
        if use_webcam:
            for i in range(3):
                print(f"\nCapture reference image {i+1}/3:")
                img = capture_from_webcam()
                if img is None:
                    print("Capture cancelled. Exiting.")
                    return
                reference_images.append(img)
        else:
            for i in range(3):
                while True:
                    path = input(f"\nReference image {i+1}/3 path (or 'q' to quit): ").strip().strip('"')
                    if path.lower() == 'q':
                        return
                    if os.path.exists(path):
                        reference_images.append(path)
                        break
                    else:
                        print(f"  File not found: {path}")
        
        # Compute reference embedding
        print("\nComputing reference embeddings...")
        ref_embedding = compute_average_embedding(model, reference_images, img_size)
        print("✓ Reference object registered!")
        
        # === PHASE 2: Detect / Compare ===
        print("\n" + "-" * 60)
        print("PHASE 2: DETECT OBJECT")
        print("Provide query images to compare against the reference.")
        print("Type 'new' to register a new reference object.")
        print("-" * 60)
        
        while True:
            if use_webcam:
                cmd = input("\nPress Enter to capture query (or 'new'/'q'): ").strip().lower()
                if cmd == 'q':
                    return
                if cmd == 'new':
                    break
                
                query_img = capture_from_webcam()
                if query_img is None:
                    continue
                query_array = preprocess_image(query_img, img_size)
            else:
                path = input("\nQuery image path (or 'new'/'q'): ").strip().strip('"')
                if path.lower() == 'q':
                    return
                if path.lower() == 'new':
                    break
                if not os.path.exists(path):
                    print(f"  File not found: {path}")
                    continue
                query_array = load_and_preprocess(path, img_size)
            
            # Compute query embedding
            query_embedding = compute_embedding(model, query_array)
            
            # Compare
            result = compare_embeddings(ref_embedding, query_embedding, threshold)
            
            print(f"\n  ╔═══════════════════════════════════╗")
            print(f"  ║  Cosine Similarity: {result['similarity']:>10.4f}    ║")
            print(f"  ║  Threshold:         {result['threshold']:>10.4f}    ║")
            print(f"  ║  Verdict:  {result['verdict']:>23s} ║")
            print(f"  ╚═══════════════════════════════════╝")


# ==================== CLI Mode ====================
def cli_mode(model, img_size, threshold, reference_paths, query_path):
    """Non-interactive CLI mode."""
    
    print("\n--- Computing Reference Embeddings ---")
    ref_embedding = compute_average_embedding(model, reference_paths, img_size)
    
    print(f"\n--- Computing Query Embedding ---")
    query_array = load_and_preprocess(query_path, img_size)
    query_embedding = compute_embedding(model, query_array)
    print(f"  Query embedding computed (norm={np.linalg.norm(query_embedding):.4f})")
    
    # Compare
    result = compare_embeddings(ref_embedding, query_embedding, threshold)
    
    print(f"\n{'=' * 50}")
    print(f"  RESULT")
    print(f"{'=' * 50}")
    print(f"  Reference images: {len(reference_paths)}")
    print(f"  Query image:      {query_path}")
    print(f"  Cosine Similarity: {result['similarity']:.4f}")
    print(f"  Threshold:         {result['threshold']:.4f}")
    print(f"  Verdict:           {result['verdict']}")
    print(f"{'=' * 50}")
    
    return result


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(
        description="Few-Shot Object Detection using Keras Embedding Model (.h5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive (file input):
    python inference_h5.py
    
  Interactive (webcam):
    python inference_h5.py --webcam
    
  CLI mode:
    python inference_h5.py --reference img1.jpg img2.jpg img3.jpg --query test.jpg
    
  Custom threshold:
    python inference_h5.py --threshold 0.65 --reference img1.jpg img2.jpg img3.jpg --query test.jpg
        """
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Path to .h5 or SavedModel directory')
    parser.add_argument('--reference', nargs='+', type=str, default=None,
                        help='Paths to 3 reference images')
    parser.add_argument('--query', type=str, default=None,
                        help='Path to query image')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Similarity threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam for image capture')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Few-Shot Object Detection — Keras H5 Model")
    print("=" * 60)
    
    # Load config and model
    cfg = load_config()
    img_size = cfg.get('img_size', 96)
    print(f"Image size: {img_size}x{img_size}")
    print(f"Embedding dim: {cfg.get('embedding_dim', 128)}")
    
    model = load_embedding_model(args.model)
    
    # Choose mode
    if args.reference and args.query:
        # CLI mode
        if len(args.reference) < 1:
            print("Error: Provide at least 1 reference image (3 recommended)")
            sys.exit(1)
        cli_mode(model, img_size, args.threshold, args.reference, args.query)
    else:
        # Interactive mode
        interactive_mode(model, img_size, args.threshold, args.webcam)


if __name__ == "__main__":
    main()
