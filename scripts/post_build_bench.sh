#!/usr/bin/env bash
# After native llama.cpp build completes, start llama-server with our local
# GGUFs and run the real benchmark with GPU vision encoder.
set -uo pipefail
exec > >(tee -a /home/zip/jarvis-lab/logs/post-build-bench.log) 2>&1
LAB=/home/zip/jarvis-lab
SBIN="$LAB/build/llama.cpp/build/bin"
STATUS=$LAB/logs/post-build-bench.status
write() { echo "$(date -u +%FT%TZ) $*" > "$STATUS"; echo ">>> $*"; }

write "waiting_for_build"
while true; do
  s=$(cat $LAB/logs/llama-build.status 2>/dev/null || echo "?")
  case "$s" in *BUILD_DONE_OK*) break ;; *BUILD_ABORT*) write "ABORT: $s"; exit 2 ;; esac
  if ! pgrep -f do_build.sh >/dev/null; then
    if ! grep -q BUILD_DONE_OK $LAB/logs/llama-build.status 2>/dev/null; then
      write "ABORT: build process died without DONE marker"; exit 3
    fi
  fi
  sleep 20
done

[[ -x "$SBIN/llama-server" ]] || { write "ABORT: llama-server missing at $SBIN"; exit 4; }

write "stopping_ollama"
sudo systemctl stop ollama

write "starting_native_llama_server"
nohup "$SBIN/llama-server" \
  --model "$LAB/models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" \
  --mmproj "$LAB/models/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" \
  -ngl 999 --ctx-size 4096 --n-batch 512 --n-ubatch 512 \
  --jinja --flash-attn on \
  --host 127.0.0.1 --port 8080 \
  > $LAB/logs/llama-server-native.log 2>&1 &
disown
LLAMA_PID=$!
write "llama_server_pid=$LLAMA_PID"

write "waiting_for_health"
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
    write "server_ready_after_${i}s"; break
  fi
  if ! kill -0 $LLAMA_PID 2>/dev/null; then
    write "ABORT: server died; tail log:"
    tail -40 $LAB/logs/llama-server-native.log
    exit 5
  fi
  sleep 1
done
curl -fsS http://127.0.0.1:8080/health >/dev/null || { write "ABORT: never healthy"; exit 6; }

# Repoint ask_vlm.py to native server (port 8080) and remove model name (single-model server)
sed -i "s|http://127.0.0.1:11434/v1/chat/completions|http://127.0.0.1:8080/v1/chat/completions|" $LAB/scripts/ask_vlm.py

write "running_benchmark"
{
  echo "=== free + nvmap BEFORE ==="
  free -h | head -2
  sudo cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | head -10
  echo

  echo "=== warm-up (text only) ==="
  $LAB/scripts/ask_vlm.py --no-capture "say hi"
  echo
  for size in "" "_small"; do
    for letter in a b c; do
      img="$LAB/test_images/scene_${letter}${size}.jpg"
      [[ -f "$img" ]] || continue
      echo
      echo "=== $(basename "$img") ($(identify "$img" 2>/dev/null | awk "{print \$3}" || echo "?")) ==="
      $LAB/scripts/ask_vlm.py --image "$img" "Describe what you see in one sentence. If there is any readable text or sign, transcribe it exactly."
    done
  done

  echo
  echo "=== free + nvmap PEAK ==="
  free -h | head -2
  sudo cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | head -10
} 2>&1 | tee $LAB/logs/benchmark-native.log

write "DONE"
