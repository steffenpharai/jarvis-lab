## 2026-06-06 — session 2 (full roadmap shipped)

This session took the v1 dashboard from "polished MVP" to feature-rich
frontier-grade product. Every queued feature from `docs/ARCHITECTURE.md
§11` that didn't require new hardware was shipped + verified.

### `20b0e03` — fix: wake-word triggered turn now renders in the UI

When openWakeWord detected "Hey Jarvis" the server correctly spawned a
talk-mode turn but the browser only showed a toast — the rest of the
turn ran invisibly because the events SSE was only opened by a
client-side `doTurn()` call.

Extracted the SSE attach logic into `attachTurn(turn_id, jmsg, kind)`.
The wake event handler on `/live/stream` now also calls `attachTurn`
with the server-emitted `turn_id`, materializing a user/Jarvis bubble
pair and subscribing to the existing event stream.

### `022aa25` — feat: full Round 1+2 roadmap

Big commit (+3206 / -2480 lines, 4 files) delivering the entire
forward-looking roadmap.

**SQLite persistence** (`Memory` class). Every turn recorded to
`logs/jarvis.db` with FTS5 over question+reply. WAL mode, threadsafe
connection. Pin/unpin survives restart. Endpoints: `GET /memory/recent`,
`GET /memory/search?q=...`, `GET /memory/pinned`, `POST
/memory/<tid>/pin`, `POST /memory/<tid>/unpin`.

**Sentence-streaming TTS** (`StreamingTTS` class). As VLM tokens stream
in, a regex splits on sentence boundaries and a single worker thread
synthesizes each sentence via Piper, preserving order. Each segment is
saved as `logs/sessions/<tid>/seg_NNN.wav` and emitted as an
`audio_segment` SSE event. Client uses Web Audio API to decode and
play gaplessly. Perceived TTS latency drops from waiting-for-full-reply
(~2.5 s) to first-segment-ready (~500 ms).

**Shared AudioBus** (`AudioBus` class). Single `arecord` process owns
the mic continuously; raw int16 PCM in 80 ms chunks fanned out to
subscribers via per-subscriber `queue.Queue`. Eliminates mic contention
between wake word listener, voice recorder, and the audio meter.
Switched from `ffmpeg` to `arecord` because ffmpeg buffered stdout
in a way that delayed chunks indefinitely.

**openWakeWord "Hey Jarvis"** (`WakeWordListener` class). Pretrained
`hey_jarvis_v0.1.onnx` model, ONNX runtime. Subscribes to AudioBus,
runs `predict()` per 80 ms chunk, threshold 0.55 sustained over 6
inference frames triggers a callback. Cooldown 3 s. On detection,
auto-fires a talk-mode turn and emits a wake event on `/live/stream`.

**Perceptual frame hash** (`phash_frame` + `capture_frame_for_vlm`).
Resizes captured frame to 8×8 grayscale and bit-packs around mean.
Stored in `_FRAME_CACHE` alongside the encoded VLM-input path. On next
capture, if Hamming distance to cached hash is `<= PHASH_REUSE_DIST`
(6 bits) AND caller passed `allow_reuse=True` AND no crop, copy the
cached frame instead of re-encoding. Live mode uses the same hash with
`PHASH_LIVE_GATE` (4 bits) — if the scene is essentially unchanged,
the live tick SKIPS the VLM call entirely, saving ~3 s of GPU and not
spamming the Live tab with duplicate descriptions.

**Voice recorder** now uses AudioBus (no more `ffmpeg-arecord`
subprocess per turn). Simple RMS-based VAD: stop on 1.2 s of silence
after at least 1 s of audio, hard cap at `max_seconds`.

Updated `/metrics` with `memory_count`, `wake_enabled`, `wake_score`.
Updated `/settings` with `scene_cache_enabled`, `wake_word_enabled`.
HTML moved out to `scripts/jarvis_ui.html` (edited independently).

**Frontend** (`jarvis_ui.html`): four tabs now (Conversation / Live /
**Memory** / Pinned). Memory tab shows persistent history with FTS
search box; Pinned tab is now persistent in SQLite (not in-memory).
Wake-word listening ring around the voice orb. Web Audio API for
gapless sentence playback with sentence highlighted as it plays.
Scene-change pulse on feed when live mode emits a *changed* observation.
New shortcut: `W` toggles wake word. Tab switching now `1/2/3/4`.

**systemd** units: `scripts/jarvis-vlm.service` runs `start_vlm_native.sh`
with the required CMA preflight as `ExecStartPre`.
`scripts/jarvis-voice.service` runs `jarvis_voice.py`, depends on
jarvis-vlm, waits for VLM `/health` before starting.

**Dependencies added**: apt `python3-httpx 0.22.0`; pip --user
`openwakeword 0.6.0` + `numpy<2` (pinned 1.26.4 for system scipy
compat); `hey_jarvis_v0.1.onnx` pretrained model.

### `7975e12` — fix: live-mode deadlock + single-column layout + httpx streaming

**The bug:** `LiveMode.start()` and `LiveMode.stop()` both held
`self.lock` while calling `self._broadcast()` which also tried to
acquire `self.lock`. `threading.Lock` is non-reentrant. The very first
call deadlocked the LiveMode forever, which is why we saw exactly one
observation and then nothing.

Took an instrumented log on the worker thread to find it: "VLM done"
printed, but "observation appended" — which sits two lines below the
`_broadcast()` call — never did. The deadlock was inside `_broadcast`
on the line after the VLM returned.

**The fix:**
- Drop the lock around start/stop entirely; they're single-caller from
  the HTTP handler.
- `_broadcast()` snapshots subscribers under the lock then releases it
  before doing per-queue `put_nowait()`.
- `subscribe()` similarly: snapshot recent observations outside the
  lock, append to subscribers under the lock, seed the new queue
  outside the lock.

**Also in this commit:** switched from urllib to httpx for the VLM
streaming + health calls (same primitive used by OpenAI/Anthropic
Python SDKs). Removed the watchdog-thread hack from the previous
attempt. Layout: removed the two-column wide-screen layout — frontier
pattern for camera + chat is camera-as-hero on top, conversation in a
focused single column below.

### `a724301` — fix: keyboard shortcuts fire when textarea is empty

UX fix: shortcuts now fire when the focused text field is empty (so
user can press L right away without first clicking outside the
autofocused composer). Modifier keys (Ctrl/Meta/Alt) explicitly
skipped so native browser shortcuts work normally.

### `4257079` — fix: live-mode watchdog + deadline (partial)

The first attempt at fixing live mode hangs. Watchdog thread now
properly fires on deadline (was checking wrong condition). Replaced
shortly afterward by the proper deadlock fix in `7975e12`.

### `37a0f5b` — feat: deep frontier features

LiveMode (continuous narration), point queries (click-on-feed crop +
ask), AudioMonitor (RMS broadcast over SSE), system prompt presets
(focused / inspector / companion / curator), pin/unpin, Markdown
export, per-turn latency breakdown bar.

### `b078ace` — fix: prompt overcorrected to refusal + reset temp bug

The anti-hallucination rules (`447b31d`) overcorrected. On a dimly-lit
but clearly-describable scene Jarvis kept replying "The scene is too
dark to make out details" verbatim — the example phrase I put in rule
2 became a self-fulfilling pattern. Rewrote the prompt to allow
description of dim scenes while still preventing fabrication of
text/brands/exact counts. Also fixed `/settings/reset` setting
temperature to 0.4 (drift) instead of 0.2.

### `447b31d` — fix: anti-hallucination across system prompt + chip prompts

Hallucinated text on a near-pitch-dark frame. SYSTEM_PROMPT rewritten
with 5 numbered GROUND RULES that override other instructions. Default
temperature 0.4 → 0.2. Chip prompts (`read`, `count`, `identify`,
`find`) reworded to explicitly allow null answers.

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
