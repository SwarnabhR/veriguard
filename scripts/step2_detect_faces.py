#!/usr/bin/env python3
"""VeriGuard - Step 2: Face Detection & Train/Val/Test Split"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_DIR = Path.home() / 'repo' / 'veriguard'
PROCESSED_PATH = BASE_DIR / 'data' / 'processed'
TARGET_SIZE = 299  # XceptionNet input size
MIN_FACE_SIZE = 80
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("=" * 70)
print("VERIGUARD - STEP 2: FACE DETECTION & DATASET SPLIT")
print("=" * 70)
print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
print(f"Split ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
print("=" * 70)

# Initialize face detector
print("\nInitializing face detector...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("✓ OpenCV Haar Cascade loaded")

def detect_and_crop_face(img_path, target_size=299, min_size=80):
    """Detect largest face and crop to target size"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(min_size, min_size)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add 20% margin
        margin = int(w * 0.2)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        # Crop and resize
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (target_size, target_size))
        
        return face_resized
    except:
        return None

# ============================================================
# STEP 2.1: COLLECT FRAME PATHS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.1: COLLECTING EXTRACTED FRAMES")
print("=" * 70)

real_frames = list((PROCESSED_PATH / 'Real').glob('*.jpg'))
fake_frames = []

fake_datasets = ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']
for dataset in fake_datasets:
    frames = list((PROCESSED_PATH / dataset).glob('*.jpg'))
    fake_frames.extend(frames)
    print(f"  {dataset:15s}: {len(frames):6d} frames")

print(f"\n📊 Total collected:")
print(f"  Real frames: {len(real_frames):6d}")
print(f"  Fake frames: {len(fake_frames):6d}")
print(f"  Total:       {len(real_frames) + len(fake_frames):6d}")

# ============================================================
# STEP 2.2: TRAIN/VAL/TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.2: CREATING TRAIN/VAL/TEST SPLIT")
print("=" * 70)

# Split real frames
real_train, real_temp = train_test_split(
    real_frames, train_size=TRAIN_RATIO, random_state=42
)
real_val, real_test = train_test_split(
    real_temp, train_size=0.5, random_state=42
)

# Split fake frames
fake_train, fake_temp = train_test_split(
    fake_frames, train_size=TRAIN_RATIO, random_state=42
)
fake_val, fake_test = train_test_split(
    fake_temp, train_size=0.5, random_state=42
)

print(f"\n�� Split sizes:")
print(f"  Train: {len(real_train):5d} real + {len(fake_train):5d} fake = {len(real_train)+len(fake_train):5d} total")
print(f"  Val:   {len(real_val):5d} real + {len(fake_val):5d} fake = {len(real_val)+len(fake_val):5d} total")
print(f"  Test:  {len(real_test):5d} real + {len(fake_test):5d} fake = {len(real_test)+len(fake_test):5d} total")

# ============================================================
# STEP 2.3: FACE DETECTION & SAVING
# ============================================================

print("\n" + "=" * 70)
print("STEP 2.3: DETECTING FACES & SAVING")
print("=" * 70)
print("This will take 1-2 hours for 5,000 videos...\n")

def process_split(frame_list, output_dir, split_name, class_name):
    """Process frames with face detection"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0
    
    for frame_path in tqdm(frame_list, desc=f"  {split_name}/{class_name}", ncols=70, leave=False):
        face = detect_and_crop_face(frame_path, TARGET_SIZE, MIN_FACE_SIZE)
        
        if face is not None:
            output_path = output_dir / f"{frame_path.stem}_face.jpg"
            cv2.imwrite(str(output_path), face)
            success += 1
        else:
            failed += 1
    
    return success, failed

# Process all splits
splits = [
    (real_train, fake_train, 'train'),
    (real_val, fake_val, 'val'),
    (real_test, fake_test, 'test')
]

total_success = 0
total_failed = 0

for real_list, fake_list, split_name in splits:
    print(f"\n{'=' * 70}")
    print(f"Processing {split_name.upper()} split...")
    print('=' * 70)
    
    # Process real
    real_dir = BASE_DIR / 'data' / split_name / 'real'
    real_succ, real_fail = process_split(real_list, real_dir, split_name, 'real')
    
    # Process fake
    fake_dir = BASE_DIR / 'data' / split_name / 'fake'
    fake_succ, fake_fail = process_split(fake_list, fake_dir, split_name, 'fake')
    
    print(f"  Real: {real_succ:5d} success, {real_fail:5d} failed ({real_succ/(real_succ+real_fail)*100:.1f}%)")
    print(f"  Fake: {fake_succ:5d} success, {fake_fail:5d} failed ({fake_succ/(fake_succ+fake_fail)*100:.1f}%)")
    
    total_success += real_succ + fake_succ
    total_failed += real_fail + fake_fail

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETE!")
print("=" * 70)

print(f"\n📊 Face Detection Results:")
print(f"  Detected:  {total_success:6d} faces")
print(f"  Failed:    {total_failed:6d} frames")
print(f"  Success:   {total_success/(total_success+total_failed)*100:.1f}%")

print(f"\n📁 Final Dataset Structure:")
for split in ['train', 'val', 'test']:
    real_count = len(list((BASE_DIR / 'data' / split / 'real').glob('*.jpg')))
    fake_count = len(list((BASE_DIR / 'data' / split / 'fake').glob('*.jpg')))
    total = real_count + fake_count
    print(f"  {split.capitalize():5s}: {real_count:5d} real + {fake_count:5d} fake = {total:5d} total")

print(f"\n✓ Dataset ready for training!")
print(f"\nNext step: python3 scripts/step3_train_xception.py")
