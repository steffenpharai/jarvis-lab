#!/usr/bin/env bash
# Three repeatable VLM queries against three test images.
# Run after start_vlm_server.sh is up.
set -euo pipefail
DIR="/home/zip/jarvis-lab"
PY="$DIR/scripts/ask_vlm.py"
T="$DIR/test_images"

[[ -d "$T" ]] && [[ -n "$(ls -A "$T" 2>/dev/null)" ]] || {
  echo "Capture test images first:  ls $T/  (need 3 .jpg)"; exit 1; }

echo "=== Warm-up (load encoder + model into cache) ==="
"$PY" --image "$T"/scene*.jpg "Briefly: what is in this scene?" >/dev/null 2>&1 || true

for img in "$T"/*.jpg; do
  echo
  echo "=== $(basename "$img") ==="
  "$PY" --image "$img" "Describe what you see in one sentence. If there is any readable text or sign, transcribe it exactly."
done
