#!/bin/bash

echo "======================================================================="
echo "VERIGUARD - DATASET DOWNLOAD"
echo "======================================================================="
echo "Configuration:"
echo "  - Compression: c23 (90% size reduction)"
echo "  - Videos per dataset: 1,000"
echo "  - Total videos: 5,000"
echo "  - Expected size: ~60-70 GB"
echo "  - Estimated time: 2-3 hours"
echo "======================================================================="

BASE_DIR="$(pwd)"
SCRIPT="$BASE_DIR/faceforensics_download_v4.py"
OUTPUT="$BASE_DIR/data/raw_videos"

# Download each dataset
python3 "$SCRIPT" "$OUTPUT" --dataset original --compression c23 --type videos --num_videos 1000 --server EU2
python3 "$SCRIPT" "$OUTPUT" --dataset Deepfakes --compression c23 --type videos --num_videos 1000 --server EU2
python3 "$SCRIPT" "$OUTPUT" --dataset FaceSwap --compression c23 --type videos --num_videos 1000 --server EU2
python3 "$SCRIPT" "$OUTPUT" --dataset Face2Face --compression c23 --type videos --num_videos 1000 --server EU2
python3 "$SCRIPT" "$OUTPUT" --dataset NeuralTextures --compression c23 --type videos --num_videos 1000 --server EU2

echo ""
echo "======================================================================="
echo "DOWNLOAD COMPLETE!"
echo "======================================================================="

# Verify
echo "Verifying downloads..."
for ds in original Deepfakes FaceSwap Face2Face NeuralTextures; do
    [ "$ds" = "original" ] && path="$OUTPUT/original_sequences/youtube/c23/videos" || path="$OUTPUT/manipulated_sequences/$ds/c23/videos"
    [ -d "$path" ] && echo "  $ds: $(ls $path/*.mp4 2>/dev/null | wc -l) videos ($(du -sh $path 2>/dev/null | cut -f1))"
done

echo ""
echo "✓ Ready for Step 1: Frame Extraction"
