# JARVIS Lab

**An offline, Iron-Man-style AI — vision, voice, agentic tools, and a
Palantir/Anduril-grade operational HUD — running entirely on a single
8 GB NVIDIA Jetson Orin Nano Super.**

No cloud in the default path. You talk to a companion orb like Tony Stark talks
to JARVIS; point a camera at the world and ask what it sees; tap anything to
zoom in, identify it, and look it up; and watch every subsystem, capability and
reasoning step in a live command-center interface.

```
              ┌─────────── INTERACTIVE LOOP (priority) ───────────┐
 mic ─► whisper.cpp ─► ┐                                           │
                       ├─► Qwen2.5-VL-3B (GPU) ─► Piper TTS ─► 🔊   │
 camera ─► MJPEG ──────┘        │  90+ tools (ReAct)               │
                                ▼                                  │
              ┌──────── AMBIENT LOOP (yields the GPU) ─────────────┤
 scene-change gate ─► keyframe caption ─► SQLite world-model       │
                   └─► natural-language watch rules ─► alerts ──────┘
```

> Built on the 2026 edge-AI recipe the frontier labs converged on:
> **cheap trigger → keyframed VLM → local vector/FTS memory → cloud only when asked.**
> Everything else is engineering to make that fit in 8 GB.

---

## What it feels like

- **Talk to it.** "Hey Jarvis" wake word → a refined, British-voiced persona that
  addresses you as *sir*, answers in ~2–3 s, and stays grounded in what the
  camera actually sees.
- **Ask about the world.** *"What's that?"* → *"a bird"* → *"what kind?"* and it
  **auto-locates the subject, digitally zooms (low-light enhance + upscale),
  identifies it as specifically as the image allows, and looks it up on the web.**
- **Tap to enhance.** Click anywhere on the live feed — the Iron-Man "enhance
  there" gesture — and JARVIS investigates that spot.
- **It remembers.** A persistent visual memory of everything it has seen — ask
  *"where did I last see my keys?"* and scrub back through time.
- **It watches.** *"Alert me if someone's at the door"* → a proactive monitor
  that pings you (toast + voice) when the condition becomes true.
- **You can see everything.** A live operational console shows subsystem health,
  resources, the capability catalog, a tool-call ledger, and JARVIS's
  plan → act → observe reasoning as it happens.

---

## The interface — a command-center HUD

The camera is the full-bleed **world view**; a holographic **companion orb** is
the thing you talk to; the chat recedes into ephemeral captions. Around it:

| Element | What it is |
|---|---|
| **Companion orb** | three.js audio-reactive fresnel orb + particle halo; reacts to your voice |
| **Edge-glow presence** | full-viewport border that breathes/idles and shifts color by state (listening/thinking/speaking/alert) |
| **SYSTEMS rail** | live link-health of every subsystem (VLM/CAM/STT/TTS/DETECT/WAKE/AGENT/LIVE), resources, knowledge counts, **live telemetry sparklines** |
| **OPERATION + REASONING** | what JARVIS is doing right now + a live plan→act→observe stream |
| **Intel panel** | Talk · Entities · Activity (tool-call ledger) · Tools (capability catalog) · Seen · Memory |
| **Entity inspector** | object-centric dossiers + **co-occurrence links** you can pivot between (Palantir Gotham core) |
| **Link-chart** | full-screen force-directed graph of entities and their links |
| **3D point-cloud** | rotating depth-from-luminance scan of the live scene |
| **Timeline scrubber** | scrub back through what JARVIS saw over time |
| **Targeting frame + radar sweep** | always-on Iron-Man HUD chrome |
| **⌘K spotlight** | global search across entities, capabilities, and memory |

Design language: Palantir Blueprint density + Apple "Liquid Glass" + restraint.
A single accent (rust) with semantic state colors, Geist + Geist Mono with
tabular numerals, physics-based motion. All eye-candy renders on the *viewer's*
GPU (three.js via CDN) — zero extra load on the Jetson.

---

## Capabilities

**Vision** — open-vocabulary VQA · `investigate` (locate → low-light enhance →
digital zoom → fine-grained identify → web lookup) · tap/point queries ·
read-all-text (OCR) · barcode lookup · depth-of-point · multi-frame compare ·
optional **NanoOWL** open-vocab detection (opt-in; see *8 GB reality*).

**Voice** — openWakeWord "Hey Jarvis" · whisper.cpp `tiny.en` STT · Qwen2.5-VL
streaming · Piper `en_GB-alan` (British male) sentence-streamed, gapless Web-Audio
playback with sentence highlight.

**Memory & world model** — SQLite persistence with FTS5 search · ambient,
scene-gated **visual memory** captioner · **entity registry** + co-occurrence
graph · semantic-ish recall ("where did I last see X").

**Proactive agent** — natural-language **watch rules** evaluated on scene-change
keyframes, debounced, firing toast + TTS alerts (the Ambient.ai Pulsar pattern).

**Agentic tools** — a ReAct plan→act→observe loop over **90+ local tools**: web
(search, Wikipedia, weather, DNS, HN, arXiv…), reasoning (math, units, regex,
crypto), productivity (notes, todos, reminders, bookmarks, journal), vision,
memory, self-management, and smart-home (Hue). Cloud-frontier escalation
(Claude/GPT/Gemini) exists but is **gated off by default** — fully local unless
you opt in with a key.

Full HTTP surface in [`docs/API.md`](docs/API.md).

---

## Architecture

A single-process, multi-threaded Python orchestrator — **no web framework, no
heavy deps**.

| Layer | Implementation |
|---|---|
| VLM serving | native `llama.cpp` `llama-server`, Qwen2.5-VL-3B-Instruct Q4_K_M + Q8_0 mmproj, CUDA sm_87, FlashAttention, full GPU offload, on `:8080` |
| Orchestrator + UI | [`scripts/jarvis_voice.py`](scripts/jarvis_voice.py) — `ThreadingHTTPServer` on `:8085`, SSE turn streaming, camera/audio buses, live mode, captioner, watcher |
| Tools | [`scripts/jarvis_tools.py`](scripts/jarvis_tools.py) — registry, ReAct loop, the `investigate` pipeline, web/vision/etc. tools |
| Dashboard | [`scripts/jarvis_ui.html`](scripts/jarvis_ui.html) — single self-contained file; SVG/CSS HUD + three.js FX |
| State | SQLite (`turns`, `visual_memory`+FTS, `watch_rules`, `tool_calls`, notes/todos/reminders…) |
| Detection sidecar (opt-in) | NanoOWL (OWL-ViT patch32 + TensorRT) in a container on `:8086` |

The **dual-loop** is the key idea: interactive turns hold a global `VLM_BUSY`
lock; background work (captioner, watcher, live narration) *yields* the GPU when
you're interacting. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design,
the rejected alternatives, and the research it's based on.

---

## The 8 GB reality (the interesting part)

Shipping a frontier-feeling assistant on 8 GB of *unified* memory is mostly a
fight against that ceiling. The hard-won findings:

- **Tegra CMA preflight is mandatory.** The ~1.8 GB mmproj needs a large
  contiguous CUDA allocation; without `drop_caches` + `compact_memory` first it
  fails with `NvMapMemAllocInternalTagged error 12`. Baked into the VLM unit.
- **Serialize VLM inference.** Concurrent mmproj image calls (an interactive
  investigate *while* the ambient captioner fires) spike memory and SIGSEGV the
  server. A single `VLM_BUSY` lock — interactive blocks, background skips — fixed
  it (0 crashes under load after).
- **One VLM + small engines is the real budget.** Running NanoOWL co-resident
  with the *full* VLM OOM-kills / fails CUDA init. So open-vocab detection is
  opt-in, and continuous awareness is delivered via the VLM captioner instead.
  (Real-time detection co-residency wants an Orin NX 16 GB.)
- **Camera autofocus matters more than the model.** A webcam left in manual
  focus made it misread a Nerds box as Skittles; re-asserting auto
  focus/exposure/white-balance was a bigger accuracy win than any prompt change.
- **Headless `multi-user.target`** reclaims the GPU the desktop fragments.
- **`httpx` streaming with a wall-clock deadline** is the only reliable way to
  stream + cancel LLM HTTP (urllib/requests both have edge-case failures).

---

## Hardware

| Component | Spec |
|---|---|
| Compute | NVIDIA Jetson Orin Nano Super 8 GB (JetPack 6.2.x, Ubuntu 22.04, multi-user.target) |
| Camera + mic | Logitech C615 USB UVC (MJPEG, autofocus) |
| Audio out | browser Web Audio (dev) · Bluetooth A2DP (planned) |
| Reach | USB-C (192.168.55.1) or Wi-Fi; dashboard on `:8085` |

---

## Quick start

Full fresh-install walkthrough in [`docs/DEPLOY.md`](docs/DEPLOY.md). Once
installed, the systemd units bring everything up on boot:

```bash
sudo systemctl enable --now jarvis-vlm jarvis-voice   # VLM (:8080) + dashboard (:8085)
# open http://<jetson-ip>:8085/
```

Optional open-vocab detector (8 GB-tight, opt-in):

```bash
sudo systemctl start jarvis-owl                        # NanoOWL sidecar (:8086)
```

Operations, troubleshooting, and the CMA preflight details:
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

---

## Repo layout

```
jarvis-lab/
├── README.md            # this file
├── ARCHITECTURE.md      # full system design + decisions + research
├── CHANGELOG.md         # per-commit ship log
├── docs/
│   ├── DEPLOY.md        # fresh-Jetson install
│   ├── OPERATIONS.md    # start/stop/monitor/troubleshoot
│   ├── API.md           # HTTP endpoint reference
│   └── PROMPT_DESIGN.md # persona + anti-hallucination ground rules
└── scripts/
    ├── jarvis_voice.py        # orchestrator + dashboard server
    ├── jarvis_tools.py        # tool registry + ReAct loop + investigate
    ├── jarvis_ui.html         # single-file HUD dashboard
    ├── owl_sidecar.py         # NanoOWL open-vocab detector service
    ├── run_owl_sidecar.sh     # build engine + run the OWL container
    ├── start_vlm_native.sh    # CMA preflight + native llama-server
    └── jarvis-{vlm,voice,owl}.service   # systemd units
```

Gitignored (re-fetched/built at install): `models/`, `build/`, `piper/`, `logs/`.
Secrets (optional cloud keys) live at `~/.config/jarvis/keys.json`, never in-repo.

---

## License

MIT — see [`LICENSE`](LICENSE).

*Separate from the mobile-robot stack at `steffenpharai/zip`: different use case
(assistant, not robot), model (VLM, not LLM), and runtime (native llama.cpp).*
