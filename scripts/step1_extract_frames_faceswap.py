#!/usr/bin/env python3
"""VeriGuard - Step 1c: Extract Frames from FACESWAP Videos"""
import os, cv2
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path.home() / 'repo' / 'veriguard'
RAW_PATH = BASE_DIR / 'data' / 'raw_videos'
PROCESSED_PATH = BASE_DIR / 'data' / 'processed'

def extract_frames(video_path, output_dir, max_frames=30, skip=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return 0
    frame_count = extracted = 0
    video_name = video_path.stem
    while extracted < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % skip == 0:
            cv2.imwrite(str(output_dir / f"{video_name}_f{extracted:04d}.jpg"), frame)
            extracted += 1
        frame_count += 1
    cap.release()
    return extracted

print("=" * 70)
print("VERIGUARD - STEP 1c: FACESWAP")
print("=" * 70)
video_path = RAW_PATH / 'manipulated_sequences/FaceSwap/c23/videos'
videos = list(video_path.glob('*.mp4'))
print(f"Processing {len(videos)} videos...")
output_dir = PROCESSED_PATH / 'FaceSwap'
output_dir.mkdir(parents=True, exist_ok=True)
total = sum(extract_frames(v, output_dir, 30, 10) for v in tqdm(videos, ncols=70))
print(f"✓ Complete: {total} frames\nNext: python3 scripts/step1_extract_frames_face2face.py")
