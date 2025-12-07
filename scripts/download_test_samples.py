#!/usr/bin/env python3
"""Download test samples without Kaggle"""

import requests
import time
from pathlib import Path

BASE_DIR = Path.home() / 'repo' / 'veriguard'
TEST_DIR = BASE_DIR / 'external_tests' / 'web_samples'

# Create directories
(TEST_DIR / 'ai_generated').mkdir(parents=True, exist_ok=True)
(TEST_DIR / 'real_photos').mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DOWNLOADING TEST SAMPLES")
print("=" * 70)

# Download AI-generated faces from thispersondoesnotexist.com
print("\nDownloading AI-generated faces...")
for i in range(1, 11):
    try:
        response = requests.get('https://thispersondoesnotexist.com/', timeout=10)
        if response.status_code == 200:
            filepath = TEST_DIR / 'ai_generated' / f'ai_face_{i}.jpg'
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Downloaded: ai_face_{i}.jpg")
        time.sleep(2)  # Be nice to the server
    except Exception as e:
        print(f"  ✗ Error downloading AI face {i}: {e}")

# Download real photos from Unsplash (free stock photos)
print("\nDownloading real face photos...")
real_photo_ids = [
    "506794778202-cad84cf45f1d",  # Man with beard
    "500648767791-00dcc994a43e",  # Man smiling
    "507003603a17-c7a3226ad33f",  # Woman
    "544005313-658da97340ae",     # Woman portrait
]

for i, photo_id in enumerate(real_photo_ids, 1):
    try:
        url = f"https://images.unsplash.com/photo-{photo_id}?w=400&q=80"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            filepath = TEST_DIR / 'real_photos' / f'real_face_{i}.jpg'
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Downloaded: real_face_{i}.jpg")
        time.sleep(1)
    except Exception as e:
        print(f"  ✗ Error downloading real photo {i}: {e}")

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print(f"Location: {TEST_DIR}")
print(f"AI-generated: {len(list((TEST_DIR / 'ai_generated').glob('*.jpg')))} images")
print(f"Real photos: {len(list((TEST_DIR / 'real_photos').glob('*.jpg')))} images")
