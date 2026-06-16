# Jarvis Lab — Architecture

Wearable always-on conversational AI with on-demand vision, on a Jetson
Orin Nano Super 8 GB. This document captures the system as built and the
decisions behind it.

---

## 1. Use case

You are wearing a backpack with a Jetson and a USB camera. You talk to
"Jarvis" naturally. You can ask it about anything the camera sees — a
storefront, a person, a sign, a dog — and it responds in a few seconds
through your audio output. The whole loop is local by default; cloud
escalation is opt-in (TODO).

This stack is **separate** from:

- `steffenpharai/zip` — the mobile-robot stack (Elegoo car + brain). Same
  Jetson host previously, different application.
- `~/.openclaw` — the earlier OpenClaw conversational agent attempt that
  was strategically rejected as "too heavy for 8 GB" (see project memory).

---

## 2. Hardware

| Component        | Spec                                                |
|------------------|-----------------------------------------------------|
| Compute          | NVIDIA Jetson Orin Nano Super 8 GB shared LPDDR5    |
| GPU              | Ampere, 1024 CUDA cores, compute capability 8.7     |
| OS               | JetPack 6.2.x, L4T R36.4, Ubuntu 22.04, multi-user.target |
| Camera + mic     | Logitech C615 USB UVC (MJPEG 1280×720, mono mic via ALSA card 0) |
| Audio out (dev)  | PC speakers via browser (WAV served by `/audio/`)   |
| Audio out (prod) | Bluetooth A2DP to Pixel Buds 2 — TODO               |
| Power (target)   | 100 Wh USB-PD pack, ~6 h typical                    |

---

## 3. Resource budget

After cleanup + multi-user.target the Jetson has **~6.8 GB RAM
available** and a clean GPU. Peak measurements with the full stack
running:

| Component                                | GPU VRAM | RAM      |
|------------------------------------------|----------|----------|
| `llama-server` (Qwen2.5-VL-3B Q4_K_M + Q8 mmproj + KV) | **3.86 GB** | (negligible) |
| `jarvis_voice.py` orchestrator           | 0        | ~30 MB   |
| `ffmpeg` camera streamer (always-on)     | 0        | ~50 MB   |
| `whisper.cpp` on demand (tiny.en, CPU)   | 0        | ~150 MB peak |
| `piper` on demand (CPU)                  | 0        | ~120 MB peak |
| Kernel + OS                              | (negligible) | ~700 MB |

End state: **~3 GB RAM available** while idle, **3.86 GB GPU resident**.
This is the budget every future addition (wake-word, escalation, etc.)
plans against.

### Cleanup that got us here

1. Stop the `splat-lab/live_stream.py` workload that previously held the
   camera.
2. Drop unused Ollama models (`gemma4`) and unused Docker images
   (`dustynv/l4t-pytorch`).
3. **Drop to `multi-user.target`** — `systemctl set-default
   multi-user.target && systemctl stop gdm`. This is **load-bearing**,
   not optional. With the GUI running, even ~141 MB of GPU usage by
   `gnome-shell + Xorg` fragments the iGPU NvMap memory enough that the
   mmproj 800 MB contiguous CUDA allocation fails with
   `NvMapMemAllocInternalTagged error 12 (ENOMEM)`.

---

## 4. The VLM choice — Qwen2.5-VL-3B-Instruct Q4_K_M, native llama.cpp CUDA

Selected after a frontier survey of the 2026 small-VLM landscape on
Orin Nano Super.

| Candidate                  | Why considered | Why rejected (if rejected) |
|----------------------------|----------------|----------------------------|
| **Qwen2.5-VL-3B-Instruct** | Best sub-7B OCR (OCRBench 78.4), open weights, mature llama.cpp support | **selected** |
| Qwen3-VL-2B / 4B           | Strongest paper, newer arch | llama.cpp kernels not ready in mid-2026 (0.53 tok/s, vLLM OOM) |
| Moondream 3 Preview        | Edge-specialist tiny VLM | BSL-1.1 non-commercial, 9B-MoE active set OOMs in 8 GB |
| NVILA-Lite-2B / 3B / 8B    | NVIDIA-tuned for Jetson | CC-BY-NC-4.0 license, weak-vs-Qwen at sub-3B |
| InternVL3-2B (fallback)    | MIT license, strong DocVQA | smaller MMMU than Qwen; kept as named fallback |
| LLaVA-OneVision-7B         | Top-tier | 0.57 tok/s on Orin Nano, unusable for conversation |
| SmolVLM2-2.2B              | Fastest (12.9 tok/s) | OCR + visual QA visibly weaker; storefront-sign reading suffers |
| Phi-4-Multimodal           | Microsoft | No published Orin Nano INT4 path, too big at 5.6 B / 15 B |
| Florence-2                 | Tiny, fast | Caption/detect specialist, not conversational |

Verified throughput on real C615 frames (build `b1-308f61c` from
`ggml-org/llama.cpp` upstream `main`, compiled with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_F16=ON`):

| Frame size       | Vision encoder | Total prefill | Generation |
|------------------|----------------|---------------|------------|
| 512×384 (default)| 580–1054 ms    | ~1.0 s        | 22.7 tok/s |
| 1280×720 (HD)    | 5359–5516 ms   | ~5.6 s        | 22.5 tok/s |

The 512×384 capture is the default because it sits comfortably under
the 1.2 s "feels conversational" cold-first-token budget.

### Why **not** Ollama

Ollama is installed on this Jetson but its `qwen2.5vl:3b` integration
runs the **vision encoder on CPU** — measured `image slice encoded in
47571 ms` for a single 1280×720 frame, vs. **580 ms** on the native
llama.cpp GPU path. The LLM half runs on GPU in both cases (~21 tok/s).
Generation is fine; the encoder is the gap. Ollama remains stopped;
restart with `sudo systemctl start ollama` if you want text-only models
for unrelated work.

### Why **not** `dustynv/llama_cpp:b5283`

The dustynv container is the obvious-looking path, but build `b5283`
sits in a dead zone — `llama-server` had multimodal support removed
during the mtmd refactor and only re-added in a later upstream build.
The `--mmproj` flag literally doesn't exist on that container's
`llama-server`. The `llama-mtmd-cli` works but only as a CLI (no HTTP).
We build llama.cpp from upstream `main` on the device instead.

---

## 5. The Tegra CMA preflight (load-bearing)

On Jetson Orin Nano 8 GB the iGPU shares LPDDR5 with the CPU. Contiguous
CUDA buffers > 500 MB require contiguous physical pages, allocated from
the kernel's CMA pool. With other GPU clients (gnome-shell, Xorg, prior
allocator residue) fragmenting CMA, `CmaFree` drops far below the 800 MB
the Qwen2.5-VL mmproj needs, even when `free -h` shows plenty of memory.

Symptom:
```
NvMapMemAllocInternalTagged: 1075072515 error 12  (ENOMEM)
GGML_ASSERT(buffer) failed
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 805.66 MiB on device 0: cudaMalloc failed
alloc_tensor_range: failed to allocate CUDA0 buffer of size 844796928
```

Fix (always run before starting the VLM):
```bash
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
echo 1 | sudo tee /proc/sys/vm/compact_memory
```

Measured impact: `CmaFree` jumps 6948 kB → 219204 kB (33×). After this,
mmproj allocates cleanly and the server boots in ~37 s.

This is baked into `scripts/start_vlm_native.sh`. **Do not skip it.**

---

## 6. Voice loop

| Stage             | Choice                              | Why                                  |
|-------------------|-------------------------------------|--------------------------------------|
| Microphone        | C615 USB ALSA `plughw:CARD=C615,DEV=0` | co-located with the camera, mono is fine for ASR |
| USB power policy  | autosuspend disabled via udev rule  | C615 disconnects mid-arecord at 2 s otherwise |
| Mic gain          | 100% via `amixer` (+33 dB)          | small mic, far user                  |
| Capture           | `ffmpeg -f alsa` 16 kHz mono        | more robust than `arecord` on USB underrun |
| STT               | whisper.cpp `tiny.en`, CPU          | 5× realtime, frees GPU for VLM       |
| Reasoner / vision | Qwen2.5-VL-3B (single model for both text and vision) | Mode A — VLM is the only reasoner    |
| TTS               | Piper `en_US-amy-medium`            | 3.4× realtime, CPU, clean voice      |
| Audio out (dev)   | WAV served by `/audio/<turn>.wav`, browser auto-plays | works through SSH tunnel on PC |
| Audio out (prod)  | Bluetooth A2DP to Pixel Buds 2      | TODO; HFP mic path won't be used     |

Whisper, Piper, and the camera streamer all run on CPU. The Jetson's
6 Cortex-A78AE cores have plenty of headroom alongside the VLM (which
saturates the iGPU during prefill but is idle the rest of the time).

---

## 7. Frontend — single-file Python HTTP server

`scripts/jarvis_voice.py` (~1770 lines) is a single file containing:

- A camera streamer thread that owns `/dev/video0` continuously and
  publishes the latest JPEG into a shared buffer.
- The orchestrator that runs each turn: record → STT → capture →
  VLM (streamed) → TTS.
- A `BaseHTTPRequestHandler` exposing the HTTP API.
- The entire single-page web UI inline (CSS, HTML, JS), using
  Geist Sans/Mono from Google Fonts and marked.js from a CDN.

### Why single-file

- One artefact to deploy, one journal stream to watch.
- No build step on the Jetson.
- Easy to read end-to-end while the project is still small.

When it stops fitting in one file we'll split it; not before.

### Streaming protocol (SSE)

```
POST /turn {kind, text?, seconds?}    → {turn_id}
GET  /events/{turn_id}                → text/event-stream
       data: {"phase":"recording", ...}
       data: {"phase":"transcribing"}
       data: {"phase":"capturing"}
       data: {"phase":"thinking"}
       data: {"phase":"token","delta":"Hello"}
       data: {"phase":"token","delta":" world"}
       data: {"phase":"speaking"}
       data: {"phase":"done","result":{...}}
POST /turn/{turn_id}/stop             → {ok:true}  // cancellation
```

Each turn runs in its own thread. The SSE endpoint drains a per-turn
queue and closes when the turn completes. Cancellation sets a
`threading.Event` that the VLM streaming loop checks per chunk and
breaks out cleanly.

See [`docs/API.md`](docs/API.md) for the full endpoint reference.

---

## 8. UI design language

After surveying frontier VLM/chat UIs in mid-2026, the design mental
model is:

> "Linear's chrome wrapped around Claude's voice orb, with the camera
> feed treated like a Vision Pro environment panel — one glass HUD
> floating over a calm dark room, one warm accent color, instrument
> telemetry in the corner like a Tesla nav screen."

Concrete tokens (see `:root` in the HTML):

| Token              | Choice                                           |
|--------------------|--------------------------------------------------|
| Typeface           | Geist Sans + Geist Mono (Google Fonts CDN)       |
| Accent             | `#c15f3c` Claude-rust, single accent             |
| Surfaces           | `#0a0a0b` → `#101113` → `#16181d`                |
| Borders            | `rgba(255,255,255,0.06)`                         |
| Icons              | Inline Lucide SVG, no emoji                      |
| Glass              | One `backdrop-filter: blur(20px)` layer (HUD)    |
| Radius scale       | 8 / 12 / 16 / 20 / 999 px                        |
| Motion             | 150 ms ease-out, 240 ms drawer, 1.6 s breathing  |

**Anti-patterns we explicitly avoided:**

- No emoji in chrome
- No three-column always-visible layouts
- No multi-accent rainbows
- No skeumorphic voice mode
- No modal settings dialogs that block the feed
- No "typing…" three-dot indicator (streaming cursor is the indicator)
- No drop shadows on dark backgrounds (borders only)
- No always-on waveform when idle (orb breathes, doesn't analyze)

---

## 9. Conversation memory

A per-process `collections.deque` holds the last **3 user/assistant
pairs** (text only — frames are not retained). Each turn:

1. Includes the system prompt + history + current user turn + current
   frame in the request.
2. On success, appends the (user, assistant) pair to history.
3. On cancellation/error, history is **not** mutated.

The "Clear conversation" button (top-right) issues `DELETE /history`.

Frames are not retained because each image adds ~256–1226 vision tokens
into the prefill cost — including history frames would multiply prefill
time. The model gets the current frame plus the text of prior turns,
which is usually enough for follow-ups like "and what color is it?".

---

## 10. Prompt design (anti-hallucination)

The system prompt enforces five numbered ground rules that override any
other instruction in the conversation. All chip prompts are phrased to
allow "I don't see it" as a valid answer. Temperature is 0.2.

See [`docs/PROMPT_DESIGN.md`](docs/PROMPT_DESIGN.md) for the full
rationale and verified counter-examples (e.g. the dark-scene "read
text" case that previously hallucinated "Please do not touch the
screen.").

---

## 11. Decisions deferred / future

| Layer                       | Plan                                              |
|-----------------------------|---------------------------------------------------|
| Wake word "Hey Jarvis"      | openWakeWord on CPU (TFLite), VAD-gated           |
| Cloud frontier escalation   | MCP tool-server: `tell_me_more(question, image)` over phone tether to Claude / GPT |
| Bluetooth audio out         | Use built-in `bluez` + PipeWire for A2DP; HFP mic explicitly rejected (8 kHz NB-SCO wrecks ASR) |
| systemd auto-start          | Two units (`jarvis-vlm.service`, `jarvis-voice.service`) with the CMA preflight in `ExecStartPre` |
| Frame K/V cache across turns| Reuse vision encoder K/V for follow-ups on the same scene (pHash gate) |
| Region-of-interest queries  | Browser-side crop rect → server crops frame before sending to VLM |
| Streaming TTS               | Sentence-split the VLM stream, synthesize each sentence as it arrives |

---

## 12. v3 — Vision, Agent, World-Model & Operational HUD (2026-06)

Everything from §1–§11 shipped; this section documents the layers added on top
to turn the conversational VLM into an Iron-Man-style agent with a
Palantir/Anduril-grade interface. All of it is local-by-default and fits the
8 GB budget; the engineering to make it fit is the interesting part.

### 12.1 Dual-loop + VLM serialization (the core constraint)

Two loops share one `llama-server`:

- **Interactive loop** (priority): user turns, `investigate`, agent tool calls.
- **Ambient loop** (yields): the visual-memory captioner, the watcher, live
  narration.

A single non-reentrant lock, `jarvis_tools.VLM_BUSY`, serializes **all** VLM
image inference. `stream_vlm(..., priority=True)` blocks to acquire it;
background callers pass `priority=False` and **skip** (returning
`{"skipped": True}`) when it's held. This was added after concurrent mmproj
image calls were found to SIGSEGV the server on the 8 GB box — the single most
important stability fix. The ambient captioner also gates on
`LAST_USER_ACTIVITY` so it never competes during a burst of interaction.

### 12.2 The `investigate` pipeline (vision drill-down)

`jarvis_tools.run_investigate()` is a deterministic SSE pipeline (not left to the
3B to chain), emitting a phase per step so the HUD can animate it:

```
capture (HD) → measure luma → low-light ENHANCE (auto-tuned) →
LOCATE (point-tap | explicit region | open-vocab grid-cell voting | OWL box) →
digital ZOOM (crop + enhance + Lanczos upscale, up to 4×) →
fine-grained IDENTIFY (species/model/brand + confidence) →
WEB lookup (cleaned query → Wikipedia + scraped web results) → done
```

Localization without a detector uses **grid-cell voting** (ask the VLM which
cell of a 4×3 grid holds the subject — far more reliable on a 3B than pixel
regression), padded generously. Tap-to-investigate is the reliable hero path.
The web query is cleaned and junk-guarded (a failed "unknown" identification
never web-searches the literal word). Endpoint: `POST /investigate` → SSE on
`/events/<id>`; artifacts served from `/inv/<id>/*.jpg`.

### 12.3 Visual memory & world model

An ambient, scene-gated captioner (`VisualMemory`) writes a keyframe + one-line
caption + object list to the `visual_memory` table (+ FTS5) whenever the scene
changes (pHash gate) and the user is idle. From that single stream we derive:

- **Recall** — `recall_visual("where did I last see X")` over FTS.
- **Entity registry** — `vmem_entities()` aggregates per-frame object lists into
  distinct entities (label, sighting count, last-seen, frame).
- **Co-occurrence graph** — `vmem_graph()` / `entity_detail()` link entities that
  appear in the same keyframes (the Palantir object-linking core); the inspector
  lets you pivot between linked entities.

No second model, no GPU cost beyond the captioner's own VLM call.

### 12.4 Proactive watcher

Natural-language watch rules ("alert me if X") are evaluated by a single batched
VLM YES/NO call per scene-change keyframe, debounced per rule, firing the
existing notification system (toast + Piper TTS). The Ambient.ai-Pulsar pattern:
cheap trigger → VLM reasons → alert.

### 12.5 Tool registry + ReAct loop

`jarvis_tools.ToolRegistry` holds 90+ tools across vision / web / reason /
productivity / memory / self / smart-home, each with a JSON schema and safety
level. `agentic_loop()` runs plan → act → observe (parsing `<tool_call>` tags,
dispatching, feeding results back) until a natural answer. Cloud-frontier tools
(`ask_claude`/`ask_gpt`/`ask_gemini`/`escalate`) are registered only when
`JARVIS_ALLOW_CLOUD=1` — **off by default**.

### 12.6 NanoOWL sidecar + the co-residency wall

`owl_sidecar.py` runs OWL-ViT patch32 + TensorRT in the `dustynv/nanoowl`
container, exposing `POST /detect` (open-vocab boxes). The engine builds at
~60 qps. **But it does not reliably co-reside with the full VLM on 8 GB** —
the container OOM-kills (exit 137) or fails PyTorch CUDA/NVML init when
`llama-server` holds ~3.8 GB. So `jarvis-owl.service` is installed but **disabled
by default**; `investigate`/`detect_objects` degrade gracefully to grid-cell
localization when the sidecar is down. Real-time detection co-residency wants an
Orin NX 16 GB, or running OWL with the VLM stopped.

### 12.7 The COP HUD (`jarvis_ui.html`)

A single self-contained file. Layout: full-bleed camera **world view** →
central **companion orb** → ephemeral conversation captions → collapsible
right **intel rail** → floating **command dock**.

- **Transparency console** (Anduril/Palantir): SYSTEMS rail with live link-health
  dots + resources + knowledge counts + **telemetry sparklines** + a live
  **OPERATION** readout + a **REASONING** stream (plan→act→observe), plus an
  Activity tool-call ledger and a Tools capability catalog.
- **Visualizations** (three.js / canvas, all on the *viewer's* GPU): audio-reactive
  fresnel **orb**, edge-glow **voice presence**, **link-chart** (force-directed
  entity graph), **3D point-cloud** (depth-from-luminance scene scan), **timeline
  scrubber**, **entity tracks** on the feed, **radar sweep**, targeting frame, and
  a **boot/power-on sequence**.
- **⌘K spotlight** global search across entities/tools/memory.
- Rendering split per 2026 research: crisp chrome (reticles, text, panels) in
  SVG/CSS; only glow-dependent 3D FX in WebGL. three.js loads from CDN (vendor
  locally for offline/AP). Screenshots of the live page are blocked by its
  persistent SSE + RAF (network/compositor never idle) — verify via DOM.

### 12.8 Persona & voice

Default persona is **J.A.R.V.I.S.** — refined, addresses the user as *sir*, dry
British wit, concise, with anti-hallucination ground rules retained. TTS voice is
**Piper `en_GB-alan-medium`** (British male).

### 12.9 Camera controls

`CameraStreamer` re-asserts the C615's auto modes on every capture
(`focus_automatic_continuous=1`, `auto_exposure=3`, `white_balance_automatic=1`,
`sharpness=160`) — a manual-focus webcam was the root cause of blurry,
misread labels; this survives reboots.

### 12.10 Research basis

The design follows the 2026 frontier convergence (researched, cited in project
memory): Figure Helix **dual-loop**, Anduril Lattice **common-operating-picture**
+ entity model, Palantir Gotham **object-centric cross-linking**, Project Astra
**spatial-temporal memory**, Ambient.ai Pulsar **proactive monitoring**, and the
universal edge recipe *cheap-detector → keyframed VLM → local memory → cloud
only when asked*.

---

## 13. v4 — Data engine, full diagnostics, self-healing, perception mode (2026-06)

Layers added to push the on-device frontier and harden the 8 GB engineering.

### 13.1 Training-dataset export ("robotics data engine")
`export_dataset()` ([jarvis_voice.py](scripts/jarvis_voice.py)) turns the
`visual_memory` keyframes (auto-captioned at the edge) + grounded `turns` into a
portable, **Open-X / LeRobot-friendly** bundle: `data/vision_language.jsonl`
(image + caption + objects), `data/visual_qa.jsonl` (image + Q&A), copied frames,
a machine-readable `meta/info.json` feature schema, a `meta/consent.json`
provenance record, and a human-readable `DATASET_CARD.md`. Served at
`/dataset/export|exports|card|dl`. This is **observational vision-language data**
(`action=null`) — its value is being *auto-annotated on-device at zero marginal
cost*; action-conditioned (VLA) trajectories require wiring the same loop into a
robot's drive path (`frame → commanded velocity` = the action label).

### 13.2 Full Nano diagnostics + turbo (`TegraStats`)
A background thread parses a persistent `tegrastats` stream into a jtop-grade
snapshot — per-core CPU load+freq, GPU (GR3D), EMC, every thermal zone +
throttle headroom, INA3221 power rails (now/avg), disk, network — plus nvpmodel
mode + governor. Exposed at `/nano`; the headline fields fold into `/metrics`.
`jetson_clocks` **turbo** (`/nano/jetson_clocks`) locks clocks to max for peak
throughput (stores pre-boost state for a clean restore). Read-only, zero GPU
cost; degrades gracefully if `tegrastats` is absent. The HUD renders it as the
NANO panel with live GPU/PWR/TEMP trend graphs.

### 13.3 Self-healing VLM memory watchdog
The 8 GB ceiling is the real limiter: when `MemAvailable` collapses, llama.cpp
stalls on the ~800 MB-contiguous mmproj vision prefill and returns nothing. A
watchdog auto-refreshes `jarvis-vlm` (reclaims ~3 GB) when memory stays low —
**only when idle, never mid-inference (holds `VLM_BUSY`), and rate-limited**.
This required decoupling the dashboard from VLM restarts: `jarvis-voice.service`
`Requires=`→`Wants=jarvis-vlm` (boot ordering kept via `After=` + a `/health`
`ExecStartPre`), so a refresh no longer cascade-restarts the UI. Tunable via
`/nano/autorefresh`.

### 13.4 VLM KV-prefix cache + inference telemetry
The VLM request sets `cache_prompt` so llama.cpp reuses the cached
system+history prefix across turns (less prefill). The streamed `timings` /
`usage` are parsed into a rolling perf snapshot (tok/s, TTFT, prefill ms,
cache-reuse), surfaced at `/nano`, `/metrics`, and on the HUD. **Gotcha:**
`stream_options.include_usage` makes llama.cpp emit a final chunk with
`"choices": []`; indexing it naively (`get("choices",[{}])[0]`) raised
`IndexError` and crashed the whole stream — guarded with `(… or [{}])[0]`.

### 13.5 Perception mode (real-time NanoOWL ⊕ VLM)
On 8 GB the VLM (~3.8 GB) and OWL (~1 GB + TRT) cannot co-reside, so perception
mode is a **systemd-enforced mode switch**, not co-residency: `jarvis-owl.service`
declares `Conflicts=jarvis-vlm` (was `Wants=`, a co-residency leftover that pulled
the VLM back and OOM'd OWL). `/perception {on}` runs a threaded transition (stop
VLM → CMA preflight → start OWL, and back); the UI shows a "VLM PAUSED" banner and
polls `detect_objects` for live boxes. The watchdog is suppressed while perception
is on. The detector's tuning is an open follow-up; the swap orchestration is solid.

### 13.6 Fork-free image capture (the un-obvious 8 GB fix)
With free RAM routinely <100 MB, `subprocess`-forking the multi-GB Python process
for ffmpeg (pHash, crop/scale, captioner) **stalls for seconds** (copy-on-write
page-table setup under pressure) and was breaking every interactive turn at the
capture phase. `phash_frame`, `capture_frame_for_vlm`, and `VisualMemory.capture_now`
now decode/crop/scale **in-process with PIL** — no fork, ~1 ms, robust. (Piper TTS
is still a binary fork; `on_token()` is wrapped so a TTS failure can't abort the
VLM stream.)

### 13.7 Camera resolution + coordinate fidelity
The stream was bumped 640×480 → **1280×720** (the C615 does MJPEG to 1080p) so
investigate crops are ~3× sharper; VLM turns stay fast because
`capture_frame_for_vlm` downscales to 512×384 regardless. The feed `<img>` is
`object-fit: cover`, so tap-to-crop and the targeting reticle apply an explicit
**cover transform** (image-px ↔ element-%) — otherwise the crop and the box point
at different pixels when the window isn't 16:9.
