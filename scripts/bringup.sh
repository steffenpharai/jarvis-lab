#!/usr/bin/env bash
# End-to-end Jarvis VLM bring-up.
# Waits for downloads, starts server, runs benchmark, writes status.
set -uo pipefail
exec > >(tee -a /home/zip/jarvis-lab/logs/bringup.log) 2>&1

LAB=/home/zip/jarvis-lab
MODEL_DIR="$LAB/models/qwen2.5-vl-3b"
EXPECT_MODEL_SIZE=1929901056    # bytes
EXPECT_MMPROJ_SIZE=844757728
STATUS=$LAB/logs/bringup.status

write_status() { echo "$(date -u +%FT%TZ) $*" > "$STATUS"; echo ">>> $*"; }

write_status "waiting_for_downloads"
# Poll until both PIDs from /tmp/jarvis-*.log are gone (wgets done)
while pgrep -f "wget.*Qwen2.5-VL" >/dev/null 2>&1; do
  ACT=$(stat -c%s "$MODEL_DIR/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" 2>/dev/null || echo 0)
  MPR=$(stat -c%s "$MODEL_DIR/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" 2>/dev/null || echo 0)
  write_status "downloading model=$((ACT/1024/1024))MB/$((EXPECT_MODEL_SIZE/1024/1024))MB mmproj=$((MPR/1024/1024))MB/$((EXPECT_MMPROJ_SIZE/1024/1024))MB"
  sleep 20
done

# Wait for docker pull
while pgrep -f "docker pull dustynv" >/dev/null 2>&1; do
  write_status "docker_pulling $(tail -1 /tmp/jarvis-docker-pull.log 2>/dev/null | head -c 100)"
  sleep 15
done

# Verify
M=$(stat -c%s "$MODEL_DIR/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" 2>/dev/null || echo 0)
P=$(stat -c%s "$MODEL_DIR/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" 2>/dev/null || echo 0)
if [[ "$M" != "$EXPECT_MODEL_SIZE" ]] || [[ "$P" != "$EXPECT_MMPROJ_SIZE" ]]; then
  write_status "ABORT: size mismatch model=$M expected=$EXPECT_MODEL_SIZE mmproj=$P expected=$EXPECT_MMPROJ_SIZE"
  exit 2
fi
if ! docker image inspect dustynv/llama_cpp:b5283-r36.4-cu128-24.04 >/dev/null 2>&1; then
  write_status "ABORT: docker image not present"; exit 3
fi

write_status "starting_vlm_server"
docker rm -f jarvis-vlm 2>/dev/null || true
docker run -d --name jarvis-vlm \
  --runtime nvidia \
  --network host \
  -v "$LAB/models:/models:ro" \
  -v "$LAB/logs:/logs" \
  dustynv/llama_cpp:b5283-r36.4-cu128-24.04 \
  llama-server \
    --model /models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
    --mmproj /models/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
    -ngl 999 --ctx-size 4096 --n-batch 256 --jinja \
    --host 0.0.0.0 --port 8080 --log-prefix

write_status "waiting_for_server_ready"
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
    write_status "server_ready_after_${i}s"; break
  fi
  if ! docker ps --filter name=jarvis-vlm --format "{{.Names}}" | grep -q jarvis-vlm; then
    write_status "ABORT: container exited; tail of log:"
    docker logs jarvis-vlm 2>&1 | tail -30
    exit 4
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:8080/health || { write_status "ABORT: never healthy"; exit 5; }

write_status "capturing_test_frames"
mkdir -p "$LAB/test_images"
sleep 1
"$LAB/scripts/capture_frame.sh" "$LAB/test_images/scene_a.jpg" || true
sleep 1
"$LAB/scripts/capture_frame.sh" "$LAB/test_images/scene_b.jpg" || true
sleep 1
"$LAB/scripts/capture_frame.sh" "$LAB/test_images/scene_c.jpg" || true
ls -la "$LAB/test_images/"

write_status "running_benchmark"
{
  echo "=== free + nvmap BEFORE ==="
  free -h | head -2
  sudo cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | head -15
  echo
  for img in "$LAB/test_images/"*.jpg; do
    echo
    echo "=== $(basename "$img") ==="
    "$LAB/scripts/ask_vlm.py" --image "$img" "Describe what you see in one sentence. If there is any readable text or sign, transcribe it exactly."
  done
  echo
  echo "=== free + nvmap AFTER ==="
  free -h | head -2
  sudo cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | head -15
} 2>&1 | tee "$LAB/logs/benchmark.log"

write_status "DONE"
