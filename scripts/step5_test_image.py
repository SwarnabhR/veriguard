#!/usr/bin/env python3
"""VeriGuard - Step 5: Test Single Image"""

import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import cv2

BASE_DIR = Path.home() / 'repo' / 'veriguard'
IMG_SIZE = 224

# ============================================================
# REBUILD MODEL & LOAD WEIGHTS
# ============================================================

def build_model(input_shape=(224, 224, 3), num_classes=2):
    """Rebuild model architecture"""
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

# Load model
print("Loading VeriGuard model...")
model = build_model()
weights_path = BASE_DIR / 'checkpoints' / 'best.h5'
if not weights_path.exists():
    weights_path = BASE_DIR / 'models' / 'xception_baseline.h5'
model.load_weights(str(weights_path))
print("✓ Model loaded\n")

# ============================================================
# INFERENCE FUNCTION
# ============================================================

def preprocess_image(image_path):
    """Load and preprocess image"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(image_path):
    """Predict if image is fake or real"""
    # Preprocess
    img = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(img, verbose=0)
    
    # Get probabilities
    fake_prob = predictions[0][0]
    real_prob = predictions[0][1]
    
    # Get prediction
    predicted_class = np.argmax(predictions[0])
    label = "REAL" if predicted_class == 1 else "FAKE"
    confidence = max(fake_prob, real_prob) * 100
    
    return label, confidence, fake_prob, real_prob

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("VERIGUARD - DEEPFAKE DETECTION")
    print("=" * 70)
    
    # Check if image path provided
    if len(sys.argv) < 2:
        print("\nUsage: python3 step5_test_image.py <image_path>")
        print("\nExample:")
        print("  python3 scripts/step5_test_image.py test_face.jpg")
        print("  python3 scripts/step5_test_image.py /path/to/image.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"\n✗ Error: File not found: {image_path}")
        sys.exit(1)
    
    print(f"\nAnalyzing: {image_path}")
    print("-" * 70)
    
    try:
        # Predict
        label, confidence, fake_prob, real_prob = predict_image(image_path)
        
        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n  Prediction:  {label}")
        print(f"  Confidence:  {confidence:.2f}%")
        print(f"\n  Probabilities:")
        print(f"    Fake: {fake_prob * 100:.2f}%")
        print(f"    Real: {real_prob * 100:.2f}%")
        
        # Visual indicator
        print("\n" + "=" * 70)
        if label == "FAKE":
            print("⚠️  WARNING: This image appears to be FAKE/MANIPULATED")
        else:
            print("✓ This image appears to be REAL/AUTHENTIC")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error processing image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
