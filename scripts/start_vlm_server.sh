#!/usr/bin/env bash
# Launch Qwen2.5-VL-3B as a llama-server inside the dustynv llama_cpp container.
# Exposes OpenAI-compatible API on :8080 (host network, so /v1/chat/completions
# reachable from any process on the Jetson).
set -euo pipefail

IMAGE="dustynv/llama_cpp:b5283-r36.4-cu128-24.04"
MODEL_DIR="/home/zip/jarvis-lab/models/qwen2.5-vl-3b"
MODEL="${MODEL_DIR}/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ="${MODEL_DIR}/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"

# Sanity
for f in "$MODEL" "$MMPROJ"; do
  [[ -f "$f" ]] || { echo "missing: $f"; exit 1; }
done

# Detach mode if --detach passed; otherwise foreground for benchmarks.
DETACH=""
NAME="jarvis-vlm"
[[ "${1:-}" == "--detach" ]] && DETACH="-d --restart unless-stopped"

# --runtime nvidia: tegra/L4T GPU access
# --network host: simpler than port mapping; llama-server binds 0.0.0.0:8080
# -ngl 999: offload everything to GPU
# --ctx-size 4096: matches budget; longer = KV bloat
# --n-batch 256: matches prior memory tuning notes on this box
# --jinja: use built-in chat templates for Qwen2.5-VL
docker rm -f "$NAME" 2>/dev/null || true
exec docker run $DETACH --name "$NAME" \
  --runtime nvidia \
  --network host \
  -v /home/zip/jarvis-lab/models:/models:ro \
  -v /home/zip/jarvis-lab/logs:/logs \
  "$IMAGE" \
  llama-server \
    --model /models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
    --mmproj /models/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
    -ngl 999 \
    --ctx-size 4096 \
    --n-batch 256 \
    --jinja \
    --host 0.0.0.0 --port 8080 \
    --log-prefix \
    2>&1 | tee /logs/llama-server.log
