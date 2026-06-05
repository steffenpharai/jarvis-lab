#!/usr/bin/env bash
# Grab a single JPEG frame from /dev/video0, downscale for VLM efficiency.
# 512x384 = ~256 vision tokens vs ~1226 at 1280x720 -> 4-5x faster encoder.
set -euo pipefail
OUT="${1:-/tmp/jarvis-frame.jpg}"
DEV="/dev/video0"
TARGET_W=${JARVIS_FRAME_W:-512}
TARGET_H=${JARVIS_FRAME_H:-384}

# Try MJPEG at 1280x720 (C615 supports this), then downscale.
if ffmpeg -y -hide_banner -loglevel error -f v4l2 -input_format mjpeg \
    -video_size 1280x720 -i "$DEV" -frames:v 1 \
    -vf "scale=${TARGET_W}:${TARGET_H}:flags=lanczos" -q:v 4 "$OUT" 2>/dev/null; then
  :
else
  ffmpeg -y -hide_banner -loglevel error -f v4l2 \
    -video_size 640x480 -i "$DEV" -frames:v 1 \
    -vf "scale=${TARGET_W}:${TARGET_H}:flags=lanczos" -q:v 4 "$OUT"
fi
echo "$OUT"
