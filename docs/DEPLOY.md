# Deployment

How to bring up Jarvis Lab on a fresh Jetson Orin Nano Super, or to
recover the existing one after a wipe.

---

## Prerequisites

- NVIDIA Jetson Orin Nano Super 8 GB
- JetPack 6.2.x flashed (Ubuntu 22.04 jammy)
- Network access (Wi-Fi or Ethernet)
- `gh` CLI authenticated as a user with read access to
  `steffenpharai/jarvis-lab`
- USB UVC camera with a built-in mic (e.g. Logitech C615)
- Either: physical access to attach the camera, or a working SSH session
  (`zip-jetson` alias if you cloned that config)

---

## 1. Free the box

Skip whichever steps don't apply (a clean Jetson won't need them).

```bash
# Stop any prior workloads
sudo systemctl stop ollama          # if installed and running
pkill -x llama-server               # if a prior native server is up

# Switch to headless (REQUIRED — see ARCHITECTURE §3 / §5)
sudo systemctl set-default multi-user.target
sudo systemctl stop gdm             # immediate

# Verify
free -h                              # expect 6+ GB available
sudo cat /sys/kernel/debug/nvmap/iovmm/clients   # expect total ~0K
df -h /                              # ensure 25+ GB free
```

---

## 2. Clone the repo

```bash
cd ~
gh repo clone steffenpharai/jarvis-lab
cd jarvis-lab
mkdir -p models build piper logs
```

---

## 3. Disable C615 USB autosuspend (camera + mic stability)

```bash
sudo tee /etc/udev/rules.d/90-jarvis-c615.rules >/dev/null <<'EOF'
# Logitech C615 - keep USB power on for camera+mic continuous use
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="082c", TEST=="power/control", ATTR{power/control}="on"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger

# Also set mic gain to max
amixer -c C615 sset 'Mic' Capture 100%
```

Without this rule, `arecord`/`ffmpeg` capture cuts out at exactly 2 s
with `read error: Input/output error`.

---

## 4. Build llama.cpp with CUDA

```bash
cd ~/jarvis-lab/build
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CUDA_F16=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4 \
  --target llama-server llama-mtmd-cli llama-cli
```

Build takes ~25–30 minutes; the FlashAttention + MMQ CUDA template
instantiations are the slowest part.

Verify the multimodal flag is present:
```bash
build/bin/llama-server --help 2>&1 | grep mmproj
# expect:  -mm,   --mmproj FILE   path to a multimodal projector file.
```

---

## 5. Download the Qwen2.5-VL-3B GGUF + mmproj

```bash
mkdir -p ~/jarvis-lab/models/qwen2.5-vl-3b
cd ~/jarvis-lab/models/qwen2.5-vl-3b
wget -c \
  https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
wget -c \
  https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf
```

Expected sizes (verify exact bytes if you suspect a partial download):
- `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` — 1,929,901,056 bytes (~1.9 GB)
- `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf` — 844,757,728 bytes (~845 MB)

---

## 6. Build whisper.cpp (CPU only)

```bash
cd ~/jarvis-lab/build
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DWHISPER_BUILD_EXAMPLES=ON
cmake --build build -j 4 --target whisper-cli main
bash ./models/download-ggml-model.sh tiny.en
```

CPU-only is deliberate — keeps the GPU dedicated to the VLM. `tiny.en`
runs at ~5× realtime on Orin Nano's CPU.

---

## 7. Install Piper TTS

```bash
mkdir -p ~/jarvis-lab/piper && cd ~/jarvis-lab/piper
curl -L -o piper_aarch64.tar.gz \
  https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_aarch64.tar.gz

mkdir -p voices && cd voices
curl -L -o en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -L -o en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# Smoke test
cd ~/jarvis-lab/piper
echo "Hello. Jarvis is ready." | ./piper/piper \
  --model voices/en_US-amy-medium.onnx --output_file /tmp/jarvis_hello.wav
file /tmp/jarvis_hello.wav   # expect: RIFF (little-endian) data, WAVE audio
```

---

## 8. Bring it up

```bash
# Run the VLM server (does CMA preflight, then launches llama-server on :8080)
~/jarvis-lab/scripts/start_vlm_native.sh &

# Wait ~40s for the model + mmproj to load
until curl -fsS http://127.0.0.1:8080/health >/dev/null; do sleep 1; done
echo "VLM ready"

# Run the dashboard / orchestrator on :8085
python3 ~/jarvis-lab/scripts/jarvis_voice.py &
```

Open `http://<jetson-ip>:8085/` in any browser on your LAN.

---

## 9. Verify

```bash
curl -s http://127.0.0.1:8085/metrics | python3 -m json.tool
```

Expected:
```json
{
  "ram_total_mb": 7620,
  "ram_avail_mb": 2200,     // ~2.2 GB free after full stack
  "soc_temp_c": 53.0,
  "vlm_up": true,
  "whisper_ok": true,
  "piper_ok": true,
  "cam_fps": 9.5,
  "history_pairs": 0
}
```

Trigger a SNAP:
```bash
curl -s -X POST http://127.0.0.1:8085/turn \
  -H 'Content-Type: application/json' \
  -d '{"kind":"snap"}'
# expect: {"turn_id": "20260605-..."}
```

Then check `/history` after ~3 s — you should see the reply.

---

## 10. systemd auto-start (TODO)

Not yet shipped. Manual recipe to add yourself:

```ini
# /etc/systemd/system/jarvis-vlm.service
[Unit]
Description=Jarvis VLM (llama-server)
After=network.target

[Service]
Type=simple
User=zip
ExecStartPre=/bin/sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches && echo 1 > /proc/sys/vm/compact_memory'
ExecStart=/home/zip/jarvis-lab/scripts/start_vlm_native.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/jarvis-voice.service
[Unit]
Description=Jarvis dashboard + voice loop
After=jarvis-vlm.service
Requires=jarvis-vlm.service

[Service]
Type=simple
User=zip
ExecStart=/usr/bin/python3 /home/zip/jarvis-lab/scripts/jarvis_voice.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

The `ExecStartPre` for `jarvis-vlm.service` does the CMA preflight at
root level (the script already does it via sudo, but with systemd we can
drop the sudo).

---

## Troubleshooting

See [`docs/OPERATIONS.md`](OPERATIONS.md) §Troubleshooting.
