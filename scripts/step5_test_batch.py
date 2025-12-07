#!/usr/bin/env python3
"""VeriGuard - Batch Image Testing"""

import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

BASE_DIR = Path.home() / 'repo' / 'veriguard'
IMG_SIZE = 224

print("=" * 70)
print("VERIGUARD - BATCH IMAGE TESTING")
print("=" * 70)

# Build and load model
def build_model(input_shape=(224, 224, 3), num_classes=2):
    base_model = Xception(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape, 
        pooling='avg'
    )
    for layer in base_model.layers[:50]:
        layer.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = keras.applications.xception.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

print("\nLoading model...")
model = build_model()
weights_path = BASE_DIR / 'checkpoints' / 'best.h5'
if not weights_path.exists():
    weights_path = BASE_DIR / 'models' / 'xception_baseline.h5'
model.load_weights(str(weights_path))
print("✓ Model loaded\n")

def predict_image(image_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img, verbose=0)
        label = "REAL" if np.argmax(predictions[0]) == 1 else "FAKE"
        confidence = max(predictions[0]) * 100
        return label, confidence, predictions[0]
    except Exception as e:
        return None, None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 step5_test_batch.py <folder_path>")
        print("\nExample:")
        print("  python3 scripts/step5_test_batch.py data/test/fake/")
        print("  python3 scripts/step5_test_batch.py external_tests/ai_generated/")
        sys.exit(1)
    
    folder = Path(sys.argv[1])
    if not folder.exists():
        print(f"✗ Error: Folder not found: {folder}")
        sys.exit(1)
    
    # Get all images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        images.extend(list(folder.glob(ext)))
    
    if len(images) == 0:
        print(f"✗ No images found in: {folder}")
        sys.exit(1)
    
    print(f"Testing {len(images)} images from: {folder}\n")
    print("=" * 70)
    
    results = []
    for img_path in tqdm(images, desc="Processing", ncols=70):
        label, conf, probs = predict_image(img_path)
        if label:
            results.append({
                'file': img_path.name,
                'prediction': label,
                'confidence': conf,
                'fake_prob': probs[0] * 100,
                'real_prob': probs[1] * 100
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if len(results) == 0:
        print("✗ No valid predictions")
        return
    
    fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
    real_count = sum(1 for r in results if r['prediction'] == 'REAL')
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print(f"\nTotal processed: {len(results)}/{len(images)}")
    print(f"  Predicted as FAKE: {fake_count} ({fake_count/len(results)*100:.1f}%)")
    print(f"  Predicted as REAL: {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"  Average confidence: {avg_confidence:.2f}%")
    
    print("\n" + "=" * 70)
    print("INDIVIDUAL RESULTS (First 20)")
    print("=" * 70)
    print(f"{'Filename':<40s} {'Prediction':<6s} {'Confidence':<12s}")
    print("-" * 70)
    
    for r in results[:20]:
        print(f"{r['file']:<40s} {r['prediction']:<6s} {r['confidence']:>6.2f}%")
    
    if len(results) > 20:
        print(f"\n... and {len(results) - 20} more images")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
