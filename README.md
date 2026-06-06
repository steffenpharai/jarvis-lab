# Jarvis Lab

Wearable, always-on conversational AI with on-demand vision, running entirely
on an NVIDIA Jetson Orin Nano Super (8 GB) in a backpack. Powered by a
local Qwen2.5-VL-3B vision-language model on GPU and a local voice loop —
no cloud calls in the default path.

```
microphone ──► whisper.cpp tiny.en ──┐
                                      ├──► Qwen2.5-VL-3B ──► Piper TTS ──► speakers / Bluetooth
camera ────► live MJPEG ──────────────┘
```

You talk to Jarvis like Jarvis from Iron Man. Ask it what it sees — a
storefront, a person, a car, a sign — and it answers in a couple of
seconds.

This repo is the application stack — *not* the mobile-robot stack at
`steffenpharai/zip`. Different use case (wearable, not on a robot),
different model (VLM, not LLM), different runtime (native `llama.cpp`,
not `ollama`).

---

## Status

`v2` — full Round 1+2 roadmap shipped. Verified on a Jetson Orin Nano
Super (8 GB) running JetPack 6.2.x in multi-user.target.

| Capability                          | Status |
|-------------------------------------|--------|
| Live camera dashboard               | ✅ working, 640×480 @ 10 fps MJPEG |
| Text / SNAP / voice / point input   | ✅ all four modes |
| VLM streaming responses             | ✅ SSE token-by-token, ~22 tok/s |
| Sentence-streaming TTS              | ✅ Web Audio API gapless playback with sentence highlight |
| openWakeWord "Hey Jarvis"           | ✅ shared AudioBus, auto-fires talk turn |
| Markdown rendering in replies       | ✅ marked.js |
| Conversation memory (in-process)    | ✅ 3 turn pairs for VLM context |
| SQLite persistence + FTS search     | ✅ Memory tab, pin survives restart |
| Frame pHash cache + scene-change gate | ✅ live mode skips redundant VLM calls |
| Continuous Live Mode                | ✅ auto-narration with scene gate |
| Per-message latency breakdown bar   | ✅ stacked colored segments |
| Persona presets (4)                 | ✅ focused / inspector / companion / curator |
| Command palette + keyboard shortcuts| ✅ Enter / Space / L / W / P / R / 1-4 / ? |
| Anti-hallucination prompt           | ✅ verified on dark scenes |
| systemd auto-start units            | ✅ install once via systemctl enable |
| Backpack power + Bluetooth audio    | 🛠 hardware integration pending |
| Tool use + cloud-frontier escalation | 🛠 see docs/NEXT_SESSION.md |
| Long-term semantic memory           | 🛠 see docs/NEXT_SESSION.md |
| Multi-modal agentic loop            | 🛠 see docs/NEXT_SESSION.md |

---

---

## What it does

### Dashboard (`http://<jetson>:8085/`)

- **Live camera feed** dominates the left of the window with a glass HUD
  overlay showing model name, FPS, SoC temperature, free RAM, and VLM
  health.
- **Conversation panel** on the right scrolls independently with
  markdown-rendered bubbles, per-message hover actions (copy / regenerate),
  attached frame thumbnails, and an auto-playing audio bubble per reply.
- **Composer** at the bottom-right: type a question, hit Enter; click the
  voice orb to talk; press Space to SNAP-describe the scene; press `/`
  for the command palette.
- **Settings drawer** (gear icon) edits the system prompt, max response
  length, temperature, and voice-record duration — all live, no restart.

### Inputs

| Mode  | How                              | Latency target |
|-------|----------------------------------|----------------|
| Text  | type, Enter                      | ~2 s           |
| SNAP  | click "What do you see" / Space  | ~3 s           |
| Voice | click voice orb / `/voice`       | ~3 s + record  |

### Verified throughput

| Stage                          | Time                |
|-------------------------------|---------------------|
| Camera frame ready             | ≤ 100 ms            |
| Whisper STT (5 s audio)        | ~1 s, CPU only      |
| VLM vision encoder (512×384)   | 580–1054 ms, GPU    |
| VLM generation                 | **22.7 tok/s**, GPU |
| Piper TTS                      | 3.4× realtime, CPU  |

Whole turn: ~2.5–3 s for text, ~3 s for SNAP, ~9 s for voice with 6 s record.

---

## Hardware

| Component        | Spec                                          |
|------------------|-----------------------------------------------|
| Compute          | NVIDIA Jetson Orin Nano Super 8 GB            |
| OS               | JetPack 6.2.x, Ubuntu 22.04 jammy, multi-user.target |
| Camera + mic     | Logitech C615 USB UVC (MJPEG 1280×720, mono mic) |
| Audio out (dev)  | PC speakers via browser (WAV served by /audio/) |
| Audio out (prod) | Bluetooth A2DP to Pixel Buds 2 (TODO)         |
| Power            | 100 Wh USB-PD pack (~6 h typical)             |
| Mount            | Backpack with forward-facing strap camera     |

---

## Repo layout

```
jarvis-lab/
├── README.md             # this file
├── ARCHITECTURE.md       # system design, decisions, alternatives rejected
├── CHANGELOG.md          # per-commit ship log
├── LICENSE               # MIT
├── docs/
│   ├── DEPLOY.md         # fresh-Jetson install + bring-up
│   ├── OPERATIONS.md     # start/stop/monitor/troubleshoot
│   ├── API.md            # HTTP endpoint reference
│   └── PROMPT_DESIGN.md  # anti-hallucination ground rules
└── scripts/
    ├── jarvis_voice.py        # the single-file orchestrator + web UI
    ├── start_vlm_native.sh    # CMA preflight + native llama-server
    ├── capture_frame.sh       # ffmpeg one-shot capture (single-frame path)
    ├── ask_vlm.py             # CLI helper for benchmarking
    ├── benchmark.sh           # three-image timing run
    ├── post_build_bench.sh    # autonomous post-build benchmark
    ├── bringup.sh             # autonomous download/build orchestrator
    └── Modelfile.jarvis-vlm   # legacy ollama Modelfile (not used)
```

Gitignored (re-fetched at install time): `models/`, `build/`, `piper/`,
`logs/`, `test_images/`.

---

## Quick start

See [`docs/DEPLOY.md`](docs/DEPLOY.md) for a fresh-Jetson walkthrough. To
bring an already-installed instance back up:

```bash
ssh zip-jetson
~/jarvis-lab/scripts/start_vlm_native.sh &     # CMA preflight + VLM on :8080
python3 ~/jarvis-lab/scripts/jarvis_voice.py & # dashboard on :8085
```

Then open `http://<jetson-ip>:8085/` in any browser.

---

## License

MIT. See [`LICENSE`](LICENSE).
