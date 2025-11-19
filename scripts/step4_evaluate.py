#!/usr/bin/env python3
"""VeriGuard - Step 4: Evaluate Model on Test Set"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

BASE_DIR = Path.home() / 'repo' / 'veriguard'
IMG_SIZE = 224
BATCH_SIZE = 8

print("=" * 70)
print("VERIGUARD - MODEL EVALUATION")
print("=" * 70)

# Rebuild the same model architecture
print("\nRebuilding model architecture...")

def build_model(input_shape=(224, 224, 3), num_classes=2):
    """Same architecture as training"""
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

model = build_model()
print("✓ Model architecture rebuilt")

# Load trained weights
print("\nLoading trained weights...")
try:
    # Try loading from checkpoint first
    weights_path = BASE_DIR / 'checkpoints' / 'best.h5'
    if weights_path.exists():
        model.load_weights(str(weights_path))
        print(f"✓ Loaded weights from checkpoint (best model)")
    else:
        # Fall back to final model
        weights_path = BASE_DIR / 'models' / 'xception_baseline.h5'
        model.load_weights(str(weights_path))
        print(f"✓ Loaded weights from final model")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Trying to extract weights...")
    # Last resort: load and extract weights differently
    import h5py
    weights_path = BASE_DIR / 'models' / 'xception_baseline.h5'
    model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
    print("✓ Loaded weights (with skipping mismatches)")

# Compile (needed for evaluation)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load test dataset
print("\nLoading test dataset...")
test_ds = keras.utils.image_dataset_from_directory(
    str(BASE_DIR / 'data' / 'test'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

test_ds = test_ds.prefetch(2)
print(f"✓ Test batches: {len(test_ds)}")

# Evaluate
print("\n" + "=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)

results = model.evaluate(test_ds, verbose=1)
test_loss = results[0]
test_accuracy = results[1]

print(f"\n✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy:.4f}")

# Get predictions
print("\nGenerating predictions...")
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print("\nClass mapping: 0=Fake, 1=Real")
print("\n" + classification_report(
    y_true, 
    y_pred, 
    target_names=['Fake', 'Real'],
    digits=4
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print("\n              Predicted")
print("              Fake    Real")
print(f"Actual Fake   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"       Real   {cm[1,0]:5d}   {cm[1,1]:5d}")

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "=" * 70)
print("DETAILED METRICS")
print("=" * 70)
print(f"True Positives (TP):  {tp:5d}")
print(f"True Negatives (TN):  {tn:5d}")
print(f"False Positives (FP): {fp:5d}")
print(f"False Negatives (FN): {fn:5d}")
print(f"\nPrecision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1-Score:     {f1_score:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"Accuracy:     {test_accuracy:.4f}")

# Save results
(BASE_DIR / 'results').mkdir(exist_ok=True)

results_dict = {
    'Test Loss': test_loss,
    'Test Accuracy': test_accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score,
    'Specificity': specificity,
    'True Positives': int(tp),
    'True Negatives': int(tn),
    'False Positives': int(fp),
    'False Negatives': int(fn)
}

results_df = pd.DataFrame([results_dict])
results_file = BASE_DIR / 'results' / 'evaluation_results.csv'
results_df.to_csv(results_file, index=False)

print(f"\n✓ Results saved: {results_file}")

# Save summary
summary_file = BASE_DIR / 'results' / 'summary.txt'
with open(summary_file, 'w') as f:
    f.write("VERIGUARD - EVALUATION SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1_score:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"              Fake    Real\n")
    f.write(f"Actual Fake   {cm[0,0]:5d}   {cm[0,1]:5d}\n")
    f.write(f"       Real   {cm[1,0]:5d}   {cm[1,1]:5d}\n")

print(f"✓ Summary saved: {summary_file}")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
