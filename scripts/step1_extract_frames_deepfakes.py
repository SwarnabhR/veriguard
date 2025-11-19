#!/usr/bin/env python3
"""VeriGuard - Step 1b: Extract Frames from DEEPFAKES Videos"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path.home() / 'repo' / 'veriguard'
RAW_PATH = BASE_DIR / 'data' / 'raw_videos'
PROCESSED_PATH = BASE_DIR / 'data' / 'processed'
MAX_FRAMES = 30
FRAME_SKIP = 10

print("=" * 70)
print("VERIGUARD - STEP 1b: EXTRACT FRAMES FROM DEEPFAKES")
print("=" * 70)

def extract_frames(video_path, output_dir, max_frames=30, skip=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    frame_count = extracted = 0
    video_name = video_path.stem
    
    while extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip == 0:
            cv2.imwrite(str(output_dir / f"{video_name}_f{extracted:04d}.jpg"), frame)
            extracted += 1
        frame_count += 1
    
    cap.release()
    return extracted

dataset_name = 'Deepfakes'
video_path = RAW_PATH / 'manipulated_sequences/Deepfakes/c23/videos'

print(f"\n📹 Processing: {dataset_name}")
if not video_path.exists():
    print(f"   ✗ Path not found!")
    exit(1)

videos = list(video_path.glob('*.mp4'))
print(f"   Found: {len(videos)} videos")

output_dir = PROCESSED_PATH / dataset_name
output_dir.mkdir(parents=True, exist_ok=True)

total_frames = sum(extract_frames(v, output_dir, MAX_FRAMES, FRAME_SKIP) 
                   for v in tqdm(videos, desc="   Progress", ncols=70))

print(f"\n{'=' * 70}")
print(f"✓ DEEPFAKES COMPLETE: {total_frames} frames")
print(f"\nNext: python3 scripts/step1_extract_frames_faceswap.py")
