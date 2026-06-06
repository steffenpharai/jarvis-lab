# Operations

Day-to-day running, monitoring, and recovery of a deployed Jarvis Lab.

---

## Daily ops

### Start

```bash
ssh zip-jetson

# 1. Bring the VLM up (CMA preflight + llama-server on :8080)
~/jarvis-lab/scripts/start_vlm_native.sh &

# 2. Wait for it
until curl -fsS http://127.0.0.1:8080/health >/dev/null; do sleep 1; done

# 3. Start the dashboard / orchestrator on :8085
python3 ~/jarvis-lab/scripts/jarvis_voice.py \
  >> ~/jarvis-lab/logs/jarvis_voice.log 2>&1 &
disown
```

### Stop

```bash
# Use exact-name kill so an SSH command line containing "llama-server"
# doesn't self-kill (don't use `pkill -f`).
pkill -x llama-server                       # VLM
kill $(pgrep -f scripts/jarvis_voice.py)    # dashboard
```

### Status

```bash
# Quick visual check from any browser on the LAN:
open http://<jetson-ip>:8085/

# Raw JSON
curl -s http://<jetson-ip>:8085/metrics | python3 -m json.tool
```

Expected healthy state:
```json
{
  "vlm_up": true,
  "whisper_ok": true,
  "piper_ok": true,
  "cam_fps": 9.5,
  "soc_temp_c": <70,
  "ram_avail_mb": >1500
}
```

---

## Logs

| Component         | Log file                                          |
|-------------------|---------------------------------------------------|
| llama-server      | `~/jarvis-lab/logs/llama-server-native.log`       |
| dashboard         | `~/jarvis-lab/logs/jarvis_voice.log`              |
| per-turn artefacts| `~/jarvis-lab/logs/sessions/<turn_id>/`           |
| llama.cpp build   | `~/jarvis-lab/logs/llama-build.log` (one-time)    |
| whisper.cpp build | `~/jarvis-lab/logs/whisper-build.log` (one-time)  |

Per-turn artefacts are kept indefinitely; each is small (~200 KB):
- `user.wav` — captured audio (talk-mode turns)
- `frame.jpg` — captured frame fed to the VLM (512×384)
- `reply.wav` — synthesized response audio

You can `rsync` the `logs/sessions/` tree off-device for offline review.

---

## Monitoring

The dashboard itself shows live telemetry in the HUD (top-right of the
camera panel):

| Field      | Healthy range                  |
|------------|--------------------------------|
| cam fps    | 8.5–10.5 (target 10)           |
| SoC temp   | <70 °C (warn ≥65, bad ≥80)     |
| VLM        | `up`                           |
| RAM avail  | >1500 MB                       |

Settings drawer (gear icon) → "System telemetry" expands these into a
verbose readout including whisper/piper presence.

Cheap headless check:
```bash
watch -n 2 'curl -sS http://127.0.0.1:8085/metrics | python3 -m json.tool'
```

---

## Tuning at runtime

Open the settings drawer (gear icon, top-right). Changes apply on
"Save" without restarting anything.

| Knob              | Default | Range   | Effect                              |
|-------------------|---------|---------|-------------------------------------|
| System prompt     | 5-rule  | text    | Persona + ground rules. Keep the GROUND RULES section. |
| Response length   | 240     | 60–500  | Caps tokens generated. ~22 tok/s.  |
| Temperature       | 0.2     | 0.0–1.0 | Lower = more grounded, less creative. |
| Voice recording   | 6 s     | 3–12 s  | Default talk-mode record window.   |

"Reset" returns all four to defaults (including the canonical anti-
hallucination system prompt — keep this).

---

## Troubleshooting

### "VLM never becomes healthy" / `cudaMalloc failed`

You skipped or are racing the Tegra CMA preflight.

```
NvMapMemAllocInternalTagged: 1075072515 error 12
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 805.66 MiB ...
cudaMalloc failed: out of memory
```

Recovery:
```bash
pkill -x llama-server
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
echo 1 | sudo tee /proc/sys/vm/compact_memory
cat /proc/meminfo | grep CmaFree   # expect >150000 kB
~/jarvis-lab/scripts/start_vlm_native.sh &
```

If `CmaFree` is still under 100 MB even after compaction, something is
still holding contiguous CUDA pages. Most likely you forgot to switch to
`multi-user.target`:
```bash
sudo systemctl set-default multi-user.target
sudo systemctl stop gdm
```

### "Camera shows 0.0 fps" / no live feed

```bash
# Is the ffmpeg streamer alive?
pgrep -af 'ffmpeg.*-i /dev/video0'

# Is anyone else holding the camera?
sudo lsof /dev/video0

# Re-check the autosuspend rule
cat /etc/udev/rules.d/90-jarvis-c615.rules
for f in /sys/bus/usb/devices/*/product; do
  grep -qi "C615" "$f" 2>/dev/null && \
    cat "$(dirname "$f")/power/control"   # expect "on"
done
```

Restart the dashboard (it will spawn a fresh ffmpeg streamer):
```bash
kill $(pgrep -f scripts/jarvis_voice.py)
python3 ~/jarvis-lab/scripts/jarvis_voice.py >> ~/jarvis-lab/logs/jarvis_voice.log 2>&1 &
disown
```

### "arecord: read error: Input/output error" at exactly 2 s

USB autosuspend is enabled. Re-apply the udev rule from
[`DEPLOY.md`](DEPLOY.md) §3 and reboot the camera USB:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Dashboard JS isn't running (HUD stuck at `0.0 fps · —`)

Usually a JS parse error in the inline `<script>`. Open the page in a
real browser with devtools, look at the Console tab, and fix the error.
Recent gotcha: SVG attributes inside JS single-quoted strings need
`"` (not `\\'`, not `\'`) because Python `r"""..."""` passes backslashes
through unchanged and `\\` then `'` terminates the string mid-attribute.

### "Connection refused" on `/turn`

`llama-server` died or the dashboard is pointing at the wrong port.
Check `~/jarvis-lab/logs/llama-server-native.log` — typically you'll see
the OOM trace above. Recovery: re-run the CMA preflight + start script.

### Jarvis is hallucinating (inventing text/counts/objects)

Open the settings drawer and verify:
- System prompt still contains the GROUND RULES section (rules 1–5).
- Temperature is 0.2 or lower.

If the prompt was edited and the rules removed, click "Reset" or paste
the prompt back from [`PROMPT_DESIGN.md`](PROMPT_DESIGN.md). Then test
on a known-empty scene with `/read` — the correct answer is "I don't
see any readable text in the image."

### "All slots busy" / turn never starts

The dashboard ran an old turn that never released its server slot. Hit
"Stop" or restart the dashboard. Worst case, restart `llama-server`.

### Disk filling up

Per-turn artefacts (`logs/sessions/<turn>/`) are ~200 KB each. At
~hundreds of turns per day that's tens of MB. Trim:
```bash
find ~/jarvis-lab/logs/sessions/ -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +
```

---

## Recovery from a hard crash

```bash
# 1. Kill everything
pkill -x llama-server
pkill -f 'scripts/jarvis_voice.py'
pkill -f 'ffmpeg.*-i /dev/video0'

# 2. Confirm GPU is free
sudo cat /sys/kernel/debug/nvmap/iovmm/clients   # total ~0K

# 3. Bring back per "Start" above
```

The repo working tree on the Jetson is the only persistent state that
matters — models and artefacts are re-fetchable per
[`DEPLOY.md`](DEPLOY.md).
