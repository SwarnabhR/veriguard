#!/usr/bin/env python3
"""VeriGuard - Step 3: Stable Training (No XLA Issues)"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / 'repo' / 'veriguard'
BATCH_SIZE = 8
EPOCHS = 50
IMG_SIZE = 224
LEARNING_RATE = 0.001

print("=" * 70)
print("VERIGUARD - TRAINING")
print("=" * 70)
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Epochs: {EPOCHS}")
print("=" * 70)

# ============================================================
# GPU SETUP
# ============================================================

print("\nConfiguring GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1536)]
        )
        print(f"✓ GPU: RTX 3050 (1.5GB limit)")
    except:
        pass

# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

def build_model(input_shape=(224, 224, 3), num_classes=2):
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze early layers
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
print(f"✓ Parameters: {model.count_params() / 1e6:.2f}M")

# ============================================================
# COMPILE - SIMPLE METRICS ONLY
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']  # Only accuracy, no AUC to avoid XLA issues
)

print("✓ Compiled (accuracy metric only)")

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_ds = keras.utils.image_dataset_from_directory(
    str(BASE_DIR / 'data' / 'train'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    str(BASE_DIR / 'data' / 'val'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

train_ds = train_ds.prefetch(2)
val_ds = val_ds.prefetch(2)

print(f"✓ Train batches: {len(train_ds)}")
print(f"✓ Val batches: {len(val_ds)}")

# ============================================================
# CALLBACKS
# ============================================================

(BASE_DIR / 'checkpoints').mkdir(exist_ok=True)
(BASE_DIR / 'logs').mkdir(exist_ok=True)
(BASE_DIR / 'models').mkdir(exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',  # Changed from val_auc
        patience=10,
        restore_best_weights=True,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        str(BASE_DIR / 'checkpoints' / 'best.h5'),
        monitor='val_accuracy',  # Changed from val_auc
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    CSVLogger(str(BASE_DIR / 'logs' / 'training.csv'))
]

print("✓ Callbacks ready")

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)
print("Starting training... (3-4 hours)")
print("=" * 70 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# SAVE
# ============================================================

model.save(str(BASE_DIR / 'models' / 'xception_baseline.h5'))

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Best accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Model: {BASE_DIR / 'models' / 'xception_baseline.h5'}")
