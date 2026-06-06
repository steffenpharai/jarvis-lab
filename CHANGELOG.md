# Changelog

All notable changes shipped to this stack.

Project history is captured per-commit; this is the human-readable
roll-up.

## 2026-06-05 — initial six commits

### `447b31d` — fix: anti-hallucination across system prompt + chip prompts

User reported Jarvis fabricated text (`"Please do not touch the screen."`)
on a near-pitch-dark frame with zero text visible. Fixed across four
levers:

- `SYSTEM_PROMPT` rewritten with five numbered GROUND RULES that
  override any other instruction (never invent, prefer "I don't see"
  over guessing, no fabricated text/counts/names/identifications).
- Default `temperature` 0.4 → 0.2 to reduce creative completions.
- Chip prompts (`read`, `count`, `identify`, `find`) reworded to
  explicitly allow null answers ("no readable text", "zero", "not
  visible", "too dark to identify").
- `snap` default prompt also allows "too dark or empty to describe".

Verified on the original dark frame: `read` → "I don't see any readable
text in the image.", `snap` → "The image is very dark, making it
difficult to discern specific details.", `count` → "I can't tell from
this image."

### `22be162` — gitignore: exclude piper/

`piper/` directory holds the downloaded Piper aarch64 binary and the
`en_US-amy-medium` voice ONNX (~75 MB). Re-fetched at install time per
`docs/DEPLOY.md`. No reason to track it.

### `a94ed48` — layout: full-window two-column on ≥ 980 px

Old shell was `max-width: 980px` centered, leaving the dashboard as a
narrow strip in the middle of wide screens. Now:

- ≥ 980 px viewports: two-column grid. Left column (1.45 fr) is the
  camera feed taking full height. Right column (1 fr, min 380 px) is
  the conversation feed + sticky composer.
- < 980 px: stacks vertically with camera at 38 vh on mobile.

Also fixed a source-order bug: the wide media query was emitted before
the base `.feed` rules in the source, so the cascade was reverting to
`aspect-ratio: 16 / 9` + `max-height: 48vh`. Moved the media query to
the bottom of the CSS where it belongs.

### `628fd14` — v3 frontend: frontier-grade UI

Frontier-grade refactor of the dashboard based on a survey of June 2026
VLM/chat UIs (ChatGPT Advanced Voice, Claude, Gemini Live, Open-WebUI,
NVIDIA Live VLM WebUI, Vision Pro app surfaces). The design mental
model: "Linear's chrome wrapped around Claude's voice orb, with the
camera feed treated like a Vision Pro environment panel — one glass HUD
floating over a calm dark room, one warm accent color, instrument
telemetry in the corner like a Tesla nav screen."

Server adds:
- SSE streaming endpoint (`POST /turn` returns `turn_id`, `GET
  /events/<id>` streams phase events + token deltas).
- `POST /turn/<id>/stop` cancels mid-generation.
- `/settings` GET/PUT for live tuning of system prompt, max_tokens,
  temperature, record_seconds.
- `/history` DELETE.
- Regenerate kind reuses the last user turn after popping the last
  assistant reply.
- Turn registry with 10-min GC.

Frontend (single page, no build step):
- Geist Sans + Mono via Google Fonts CDN.
- marked.js for markdown rendering, deferred code blocks.
- Single rust accent `#c15f3c`, inline Lucide SVG icons, no emoji.
- Glass HUD pills on the camera feed (live state, telemetry, model
  label).
- Composer pattern: orb + textarea + rust send button in one rounded
  surface.
- Breathing voice orb (CSS animation).
- `/` command palette with keyboard navigation.
- Slide-in right settings drawer with system prompt editor + sliders.
- Per-message hover actions (copy, regenerate, replay audio).
- Streaming cursor + token fade-in via requestAnimationFrame batching.
- Lightbox for frame thumbnails.
- Toast notifications for ok/error feedback.

Verified end-to-end:
- ~2.75 s for SNAP turn including TTS.
- Markdown lists streaming token-by-token.
- Conversation memory 3 pairs working.
- All chips and commands functional.

### `61f6039` — voice loop v1

Wires the existing VLM to a full conversational loop:

- C615 USB autosuspend disabled persistently via udev rule
  `/etc/udev/rules.d/90-jarvis-c615.rules` (autosuspend was killing
  arecord mid-stream after 2 s).
- `whisper.cpp` cloned from `ggml-org/whisper.cpp`, built CPU-only with
  CUDA explicitly disabled to keep the GPU dedicated to the VLM. `tiny.en`
  model downloaded.
- Piper TTS aarch64 binary installed, `en_US-amy-medium` voice
  downloaded.
- Single-file orchestrator `scripts/jarvis_voice.py` (289 lines at this
  stage) with a tiny web UI on port 8085.

Verified end-to-end:
- whisper-cli: 1.05 s on 5 s audio = 5× realtime.
- Vision query: Qwen2.5-VL-3B, ~3.5 s with full system prompt.
- Piper: 3.4× realtime.

### `2daeee1` — jarvis-lab v1 (initial Qwen2.5-VL bring-up)

Brings up the native llama.cpp build with Qwen2.5-VL-3B on the Jetson:

- Cleanup pass: stop the splat-lab `live_stream.py` workload, drop
  unused Ollama models (`gemma4`), prune unused Docker images
  (`dustynv/l4t-pytorch`), switch to `multi-user.target` (free ~250 MB
  RAM and — crucially — un-fragment iGPU CMA).
- Identified that `dustynv/llama_cpp:b5283` lacks `--mmproj` (dead-zone
  build between mtmd refactor and re-add), and that Ollama's
  `qwen2.5vl:3b` runs the vision encoder on CPU (47 s per 1280×720 frame).
- Native `llama.cpp` build from upstream `main`, CUDA sm_87,
  FlashAttention, with full mmproj GPU offload.
- Tegra CMA preflight (`drop_caches` + `compact_memory`) added to
  `scripts/start_vlm_native.sh` — load-bearing for the 800 MB mmproj
  contiguous allocation.

Verified throughput on real C615 frames:

| Frame size       | Vision encoder | Total prefill | Generation  |
|------------------|----------------|---------------|-------------|
| 512×384 (default)| 580–1054 ms    | ~1.0 s        | 22.7 tok/s  |
| 1280×720 (HD)    | 5.4 s          | ~5.6 s        | 22.5 tok/s  |

Quality: read "BEWARE" from a small sign in the test frame.

ARCHITECTURE.md, README.md, and bring-up scripts all committed.
