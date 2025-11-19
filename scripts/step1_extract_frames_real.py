#!/usr/bin/env python3
"""VeriGuard - Step 1a: Extract Frames from REAL Videos Only"""

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
print("VERIGUARD - STEP 1a: EXTRACT FRAMES FROM REAL VIDEOS")
print("=" * 70)
print(f"Max frames per video: {MAX_FRAMES}")
print(f"Frame sampling: Every {FRAME_SKIP}th frame")
print("=" * 70)

def extract_frames(video_path, output_dir, max_frames=30, skip=10):
    """Extract frames from a single video"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return 0
    
    frame_count = 0
    extracted = 0
    video_name = video_path.stem
    
    while extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip == 0:
            output_path = output_dir / f"{video_name}_f{extracted:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    return extracted

# Process REAL dataset only
dataset_name = 'Real'
video_path = RAW_PATH / 'original_sequences/youtube/c23/videos'

print(f"\n📹 Processing: {dataset_name}")
print(f"   Input: {video_path}")

# Check if path exists
if not video_path.exists():
    print(f"   ✗ Path not found!")
    print(f"   Please check if videos are downloaded.")
    exit(1)

# Get all video files
videos = list(video_path.glob('*.mp4'))
print(f"   Found: {len(videos)} videos")

if len(videos) == 0:
    print(f"   ✗ No videos found!")
    exit(1)

# Create output directory
output_dir = PROCESSED_PATH / dataset_name
output_dir.mkdir(parents=True, exist_ok=True)
print(f"   Output: {output_dir}")

# Extract frames
print(f"\n   Extracting frames...")
total_frames = 0

for video in tqdm(videos, desc="   Progress", ncols=70):
    frames = extract_frames(video, output_dir, MAX_FRAMES, FRAME_SKIP)
    total_frames += frames

print(f"\n{'=' * 70}")
print(f"✓ REAL DATASET COMPLETE")
print(f"{'=' * 70}")
print(f"Videos processed: {len(videos)}")
print(f"Frames extracted: {total_frames}")
print(f"Average frames/video: {total_frames/len(videos):.1f}")
print(f"Output directory: {output_dir}")
print(f"\n✓ Ready for next dataset!")
print(f"\nNext: python3 scripts/step1_extract_frames_deepfakes.py")
