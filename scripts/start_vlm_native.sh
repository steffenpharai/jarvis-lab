#!/usr/bin/env bash
# Native llama.cpp llama-server for Qwen2.5-VL-3B on Jetson Orin Nano Super.
# Includes Tegra CMA-compact preflight (drop caches + compact_memory) which
# is REQUIRED for the ~800MB mmproj contiguous CUDA allocation to succeed.
set -euo pipefail
LAB=/home/zip/jarvis-lab
SBIN=$LAB/build/llama.cpp/build/bin
MODEL=$LAB/models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
MMPROJ=$LAB/models/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf

pkill -x llama-server 2>/dev/null || true
sleep 1

# CMA preflight (REQUIRED on Jetson Orin Nano - see ARCHITECTURE.md)
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null

exec $SBIN/llama-server \
  --model     "$MODEL" \
  --mmproj    "$MMPROJ" \
  -ngl 999 \
  --ctx-size 4096 \
  --batch-size 512 \
  --ubatch-size 512 \
  --jinja \
  -fa on \
  --mmproj-offload \
  --host 127.0.0.1 --port 8080
