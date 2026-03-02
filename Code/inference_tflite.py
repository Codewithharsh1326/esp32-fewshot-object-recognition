# inference_tflite.py
# PC inference script using the TFLite embedding model (.tflite)
# Supports both float32 and INT8 quantized models.
#
# Usage examples:
#   Interactive mode:   python inference_tflite.py
#   CLI mode:           python inference_tflite.py --reference img1.jpg img2.jpg img3.jpg --query query.jpg
#   Use INT8 model:     python inference_tflite.py --model models/embedding_model_int8_esp32.tflite
#   With webcam:        python inference_tflite.py --webcam
#   Custom threshold:   python inference_tflite.py --threshold 0.65

import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image

# Only needed for preprocess_input — lightweight import
try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError:
    # Fallback if TF is not installed (for edge devices)
    def preprocess_input(x):
        """MobileNetV2 preprocessing: scale pixels from [0,255] to [-1,1]."""
        return (x / 127.5) - 1.0

# Import TFLite interpreter
try:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLiteInterpreter = tflite.Interpreter
    except ImportError:
        print("Error: Need either tensorflow or tflite-runtime installed.")
        print("  pip install tensorflow  OR  pip install tflite-runtime")
        sys.exit(1)


# ==================== Configuration ====================
# Resolve project root (parent of Code/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "embedding_model_float32.tflite")
DEFAULT_INT8_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "embedding_model_int8_esp32.tflite")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "config.json")
DEFAULT_THRESHOLD = 0.70


# ==================== TFLite Model Wrapper ====================
class TFLiteEmbedder:
    """Wrapper around TFLite interpreter for embedding extraction."""
    
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        
        print(f"Loading TFLite model: {model_path}")
        self.interpreter = TFLiteInterpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Extract info
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        self.output_dtype = self.output_details[0]['dtype']
        
        # Check if model is quantized
        self.is_quantized = self.input_dtype in (np.uint8, np.int8)
        
        if self.is_quantized:
            self.input_scale, self.input_zp = self.input_details[0]['quantization']
            self.output_scale, self.output_zp = self.output_details[0]['quantization']
        
        # Derive image size from input shape
        self.img_size = self.input_shape[1]  # Shape is [1, H, W, C]
        
        print(f"✓ Model loaded")
        print(f"  Input:  shape={self.input_shape}, dtype={self.input_dtype}")
        print(f"  Output: shape={self.output_details[0]['shape']}, dtype={self.output_dtype}")
        print(f"  Quantized: {self.is_quantized}")
        model_size = os.path.getsize(model_path) / 1024
        print(f"  Model size: {model_size:.1f} KB")
    
    def preprocess(self, pil_image):
        """Preprocess a PIL Image for this model."""
        img = pil_image.convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32)
        
        # MobileNetV2 preprocessing: [0,255] → [-1,1]
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Quantize input if needed
        if self.is_quantized:
            img_array = img_array / self.input_scale + self.input_zp
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return img_array
    
    def extract_embedding(self, img_array):
        """Run inference and return the L2-normalized embedding."""
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize if needed
        if self.is_quantized and self.output_dtype == np.int8:
            output = (output.astype(np.float32) - self.output_zp) * self.output_scale
        
        output = output.flatten().astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(output)
        if norm > 0:
            output = output / norm
        
        return output
    
    def get_embedding(self, pil_image):
        """End-to-end: PIL Image → L2-normalized embedding."""
        img_array = self.preprocess(pil_image)
        return self.extract_embedding(img_array)


# ==================== Helper Functions ====================
def load_config():
    """Load model configuration."""
    if os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {'img_size': 96, 'embedding_dim': 128}


def load_image(image_path):
    """Load image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path)


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
            captured_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print("✓ Image captured!")
            break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_image


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(emb1, emb2))


def compute_average_embedding(embedder, images):
    """Compute average embedding from a list of images (paths or PIL Images)."""
    embeddings = []
    for i, img in enumerate(images):
        if isinstance(img, (str, Path)):
            pil_img = load_image(str(img))
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        emb = embedder.get_embedding(pil_img)
        embeddings.append(emb)
        print(f"  Reference image {i+1}: embedding computed (norm={np.linalg.norm(emb):.4f})")
    
    # Average and re-normalize
    avg_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    
    print(f"  → Average reference embedding computed (norm={np.linalg.norm(avg_embedding):.4f})")
    return avg_embedding


def compare_embeddings(ref_embedding, query_embedding, threshold):
    """Compare query embedding against reference."""
    similarity = cosine_similarity(ref_embedding, query_embedding)
    is_same = similarity >= threshold
    
    return {
        'similarity': similarity,
        'threshold': threshold,
        'is_same': is_same,
        'verdict': 'SAME OBJECT ✓' if is_same else 'DIFFERENT OBJECT ✗'
    }


# ==================== Interactive Mode ====================
def interactive_mode(embedder, threshold, use_webcam=False):
    """Interactive loop: register reference → detect query objects."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE FEW-SHOT OBJECT DETECTION (TFLite)")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(f"Model type: {'INT8 quantized' if embedder.is_quantized else 'Float32'}")
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
        ref_embedding = compute_average_embedding(embedder, reference_images)
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
            else:
                path = input("\nQuery image path (or 'new'/'q'): ").strip().strip('"')
                if path.lower() == 'q':
                    return
                if path.lower() == 'new':
                    break
                if not os.path.exists(path):
                    print(f"  File not found: {path}")
                    continue
                query_img = load_image(path)
            
            # Compute query embedding
            query_embedding = embedder.get_embedding(query_img)
            
            # Compare
            result = compare_embeddings(ref_embedding, query_embedding, threshold)
            
            print(f"\n  ╔═══════════════════════════════════╗")
            print(f"  ║  Cosine Similarity: {result['similarity']:>10.4f}    ║")
            print(f"  ║  Threshold:         {result['threshold']:>10.4f}    ║")
            print(f"  ║  Verdict:  {result['verdict']:>23s} ║")
            print(f"  ╚═══════════════════════════════════╝")


# ==================== CLI Mode ====================
def cli_mode(embedder, threshold, reference_paths, query_path):
    """Non-interactive CLI mode."""
    
    print("\n--- Computing Reference Embeddings ---")
    ref_embedding = compute_average_embedding(embedder, reference_paths)
    
    print(f"\n--- Computing Query Embedding ---")
    query_img = load_image(query_path)
    query_embedding = embedder.get_embedding(query_img)
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
        description="Few-Shot Object Detection using TFLite Embedding Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive (file input):
    python inference_tflite.py
    
  Interactive (webcam):
    python inference_tflite.py --webcam
    
  CLI mode (float32):
    python inference_tflite.py --reference img1.jpg img2.jpg img3.jpg --query test.jpg
    
  CLI mode (INT8 for ESP32):
    python inference_tflite.py --model models/embedding_model_int8_esp32.tflite \\
        --reference img1.jpg img2.jpg img3.jpg --query test.jpg
    
  Custom threshold:
    python inference_tflite.py --threshold 0.65 --reference img1.jpg img2.jpg img3.jpg --query test.jpg
        """
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Path to .tflite model (default: auto-detect)')
    parser.add_argument('--int8', action='store_true',
                        help='Use INT8 quantized model (ESP32 equivalent)')
    parser.add_argument('--reference', nargs='+', type=str, default=None,
                        help='Paths to reference images (3 recommended)')
    parser.add_argument('--query', type=str, default=None,
                        help='Path to query image')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Similarity threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam for image capture')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Few-Shot Object Detection — TFLite Model")
    print("=" * 60)
    
    # Select model path
    model_path = args.model
    if model_path is None:
        if args.int8:
            model_path = DEFAULT_INT8_MODEL_PATH
        elif os.path.exists(DEFAULT_MODEL_PATH):
            model_path = DEFAULT_MODEL_PATH
        elif os.path.exists(DEFAULT_INT8_MODEL_PATH):
            model_path = DEFAULT_INT8_MODEL_PATH
        else:
            print(f"Error: No TFLite model found!")
            print(f"  Expected: {DEFAULT_MODEL_PATH} or {DEFAULT_INT8_MODEL_PATH}")
            print(f"  Run convert_to_tflite.py first.")
            sys.exit(1)
    
    # Load model
    embedder = TFLiteEmbedder(model_path)
    
    # Choose mode
    if args.reference and args.query:
        # CLI mode
        if len(args.reference) < 1:
            print("Error: Provide at least 1 reference image (3 recommended)")
            sys.exit(1)
        cli_mode(embedder, args.threshold, args.reference, args.query)
    else:
        # Interactive mode
        interactive_mode(embedder, args.threshold, args.webcam)


if __name__ == "__main__":
    main()
