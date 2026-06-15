#!/usr/bin/env bash
# Bring up the NanoOWL open-vocabulary detection sidecar (container).
#   1) build the OWL-ViT image-encoder TensorRT engine once (GPU)
#   2) run owl_sidecar.py persistently, host network, :8086
# The engine build is GPU-heavy; on the 8GB Orin Nano stop jarvis-vlm first:
#   sudo systemctl stop jarvis-vlm && ./run_owl_sidecar.sh && sudo systemctl start jarvis-vlm
set -euo pipefail
IMG=dustynv/nanoowl:r36.4.0
DATA="$HOME/jarvis-lab/owl-data"
SCRIPTS="$HOME/jarvis-lab/scripts"
ENGINE="$DATA/owl_image_encoder_patch32.engine"
mkdir -p "$DATA"

if [ ! -f "$ENGINE" ]; then
  echo "[owl] building TensorRT image-encoder engine (one-time, ~minutes)..."
  # Tegra CMA preflight — REQUIRED, the engine build needs large contiguous
  # allocations and fails with NvMap error 12 (ENOMEM) on a fragmented map.
  sudo sync || true
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true
  echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null || true
  docker run --rm --runtime nvidia --network host -v "$DATA":/data "$IMG" \
    bash -lc "cd /opt/nanoowl && python3 -m nanoowl.build_image_encoder_engine /data/owl_image_encoder_patch32.engine"
fi

echo "[owl] (re)starting sidecar container..."
docker rm -f owl-sidecar >/dev/null 2>&1 || true
docker run -d --name owl-sidecar --restart no \
  --runtime nvidia --network host \
  -v "$DATA":/data -v "$SCRIPTS":/sidecar \
  -e OWL_ENGINE=/data/owl_image_encoder_patch32.engine -e OWL_PORT=8086 \
  "$IMG" python3 /sidecar/owl_sidecar.py >/dev/null

echo "[owl] sidecar starting; waiting for health..."
for i in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:8086/health >/dev/null 2>&1; then
    echo "[owl] healthy on :8086"; exit 0
  fi
  sleep 2
done
echo "[owl] WARN: sidecar not healthy yet; check: docker logs owl-sidecar"
exit 1
