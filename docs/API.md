# HTTP API reference

`jarvis_voice.py` is a single-process Python HTTP server on port 8085.
All endpoints are unauthenticated (assumes a trusted LAN/loopback).

---

## Dashboard

### `GET /`

Returns the single-page web UI as `text/html; charset=utf-8`.

---

## Conversation turns

A "turn" is one user-and-Jarvis exchange. Three modes:

| `kind`       | What happens                                       |
|--------------|----------------------------------------------------|
| `text`       | Skip mic + STT; use `text` as the question. Captures one frame, asks the VLM, synthesizes TTS. |
| `snap`       | Skip mic + STT; ask the canonical "describe what you see" prompt against a fresh frame. |
| `talk`       | Record `seconds` from the mic, whisper-transcribe, then proceed as `text`. |
| `regenerate` | Pop the last assistant message from history and rerun the last user message against a fresh frame. |

### `POST /turn`

Start a turn. Returns immediately with a `turn_id`; results stream via
SSE.

Request body:
```json
{
  "kind": "text" | "snap" | "talk" | "regenerate",
  "text": "...",        // required for kind=text, ignored otherwise
  "seconds": 6          // optional, kind=talk only, clamped 2..15
}
```

Response:
```json
{ "turn_id": "20260605-220317-50088c" }
```

Status 400 on invalid body or missing required field.

### `GET /events/{turn_id}`

Server-Sent Events stream of the turn's lifecycle.

`Content-Type: text/event-stream`

Each event is a single `data: <json>\n\n` line. Phase sequence:

| Phase           | When                              | Extra fields                |
|-----------------|-----------------------------------|-----------------------------|
| `recording`     | only `kind=talk`                  | `seconds`                   |
| `transcribing`  | only `kind=talk`                  | —                           |
| `capturing`     | always                            | `transcription`, `question` |
| `thinking`      | VLM call started                  | —                           |
| `token`         | per output token from the VLM     | `delta` (string fragment)   |
| `speaking`      | TTS started                       | —                           |
| `done`          | full result ready                 | `result` (see below)        |
| `cancelled`     | a `/stop` arrived mid-stream      | —                           |
| `error`         | something failed                  | `error` (message)           |

Final `done` payload `result`:
```json
{
  "turn_id":        "20260605-220317-50088c",
  "kind":           "text",
  "question":       "describe the scene",
  "transcription":  "",
  "reply":          "A dark room with...",
  "cancelled":      false,
  "timings": {
    "record_s":      0.0,
    "transcribe_s":  0.0,
    "capture_s":     0.17,
    "vlm_s":         1.78,
    "tts_s":         0.80
  },
  "audio_url": "/audio/20260605-220317-50088c.wav",
  "frame_url": "/frame/20260605-220317-50088c.jpg"
}
```

Connection closes automatically after `done` or `error`.

### `POST /turn/{turn_id}/stop`

Cancel an in-flight turn. The VLM call breaks at the next token boundary
and the SSE stream emits a `cancelled` phase. History is **not** mutated
on cancel.

Response:
```json
{ "ok": true }
```

404 if the turn doesn't exist.

---

## History

### `GET /history`

Current conversation memory (last 3 user/assistant pairs, text only).

```json
{
  "messages": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

### `DELETE /history`

Clear conversation memory.

```json
{ "ok": true }
```

---

## Settings

### `GET /settings`

Current effective settings (everything tunable via the drawer).

```json
{
  "system_prompt":   "...",
  "max_tokens":      240,
  "temperature":     0.2,
  "record_seconds":  6
}
```

### `PUT /settings`

Partial update. Any subset of the four keys above. Takes effect on the
next turn.

```json
{
  "temperature": 0.3,
  "max_tokens":  300
}
```

Response: `{"ok": true}`.

### `POST /settings/reset`

Restore the canonical defaults (including the anti-hallucination
system prompt).

---

## Live frames

### `GET /stream.mjpeg`

`multipart/x-mixed-replace; boundary=jarvis` MJPEG stream from the
camera at ~10 fps. Browsers render this directly in an `<img>`.

```html
<img src="/stream.mjpeg">
```

### `GET /snapshot.jpg`

Single JPEG, latest frame from the camera buffer.

503 if no frame is available within 3 s (camera streamer not yet
producing — usually only at startup).

---

## Telemetry

### `GET /metrics`

System + service health, cached 2 s.

```json
{
  "ram_total_mb":  7620,
  "ram_avail_mb":  2200,
  "ram_used_pct":  71.1,
  "soc_temp_c":    53.0,
  "vlm_up":        true,
  "whisper_ok":    true,
  "piper_ok":      true,
  "cam_fps":       9.5,
  "history_pairs": 1
}
```

---

## Per-turn artefacts

### `GET /audio/{turn_id}.wav`

Synthesized TTS WAV for a completed turn.

### `GET /frame/{turn_id}.jpg`

The 512×384 frame that was actually fed to the VLM for that turn.

404 if the turn doesn't exist or never reached the capture phase.

---

## Conventions

- All POST/PUT bodies are JSON with `Content-Type: application/json`.
- All errors return JSON `{"error": "..."}` with a `4xx` or `5xx`
  status.
- Turn IDs are `YYYYMMDD-HHMMSS-<6 hex>`.
- The server is single-process, multi-threaded
  (`ThreadingHTTPServer`). Each turn runs in its own daemon thread.
  Concurrent turns are technically allowed but compete for the GPU; in
  practice the client serializes.

---

## v3 endpoints (vision, memory, agent, transparency)

### Vision / investigate
| Method | Path | Purpose |
|---|---|---|
| POST | `/investigate` | start the locate→enhance→zoom→identify→web pipeline. Body: `{subject?, point?:[x,y], region?:[x,y,w,h], web?}` → `{turn_id}`; stream phases via `GET /events/<turn_id>` (capturing, enhancing, locating, located, zooming, zoomed, identifying, identified, researching, researched, done) |
| GET | `/inv/<id>/<file>.jpg` | investigate artifacts (`full.jpg`, `zoom.jpg`) |
| GET | `/snapshot.jpg[?t=…]` | latest camera JPEG (tolerates a cache-buster query) |

### Visual memory & entities (the world model)
| Method | Path | Purpose |
|---|---|---|
| GET | `/memory/visual/recent` | recent keyframes `{items:[{id,ts,caption,objects,frame_url,source}], enabled, count}` |
| GET | `/memory/visual/search?q=` | FTS over captions/objects |
| POST | `/memory/visual/capture` | force-capture a keyframe now |
| POST | `/memory/visual/{enable,disable}` | toggle the ambient captioner |
| GET | `/memory/entities` | entity registry (top by count+recency) |
| GET | `/memory/entity?label=` | entity dossier: sightings, first/last seen, co-occurring (linked) entities |
| GET | `/memory/graph` | co-occurrence graph `{nodes,edges}` for the link-chart |
| GET | `/vmem/<file>.jpg` | a visual-memory keyframe |

### Proactive watch
| Method | Path | Purpose |
|---|---|---|
| GET | `/watch/rules` | list rules |
| POST | `/watch/rules` | add `{text}` |
| POST | `/watch/rules/<id>/{toggle,delete}` | manage a rule |
| POST | `/watch/test` | force-evaluate all rules against the current view |

### Tools & agent (transparency)
| Method | Path | Purpose |
|---|---|---|
| GET | `/tools` | capability catalog (name, description, schema, category, safety) |
| GET | `/tools/calls?limit=` | tool-call audit ledger |
| POST | `/tool/<name>?confirm=1` | invoke a single tool. Body: `{args:{…}}` |
| POST | `/agent` | run the ReAct loop. Body: `{question, max_steps?, use_frame?}` |

### Telemetry
`GET /metrics` now also returns: `owl_up`, `watch_count`, `uptime_s`,
`vmem_count`, `vmem_enabled`, `tool_count`, `agent_mode_enabled`.

---

## v4 endpoints (diagnostics, perception, dataset export)

### Nano diagnostics & power
| Method | Path | Purpose |
|---|---|---|
| GET | `/nano` | full Jetson telemetry snapshot from a persistent `tegrastats` stream: `{cpu:[{load,freq}], gpu_util, emc_util, temps:{...}, temp_max_c, throttle_headroom_c, power:{VDD_IN:{now,avg},...}, power_total_mw, ram/swap, net_rx_kbs, net_tx_kbs, meta:{power_mode, governor, jetson_clocks, disk_*}, vlm:{tok_s,ttft_ms,prefill_ms,prompt_n}, autorefresh:{enabled,refreshes,min_avail_mb}, ok}` |
| POST | `/nano/jetson_clocks` | ⚡ turbo. Body `{on:bool}` — locks all clocks to max (or restores DVFS). Stores pre-boost state so `--restore` is clean |
| POST | `/nano/autorefresh` | tune the self-healing watchdog. Body `{enabled?:bool, min_avail_mb?:int}` |

`GET /metrics` also gains: `gpu_util`, `emc_util`, `power_w`, `cpu_load_avg`,
`power_mode`, `jetson_clocks`, `throttle_headroom_c`, `vlm_tok_s`, `vlm_ttft_ms`,
`vlm_autorefresh`, `vlm_refreshes`, `perception_mode`, `perception_transition`.

### Perception mode (real-time NanoOWL ⊕ VLM)
| Method | Path | Purpose |
|---|---|---|
| POST | `/perception` | Body `{on:bool}`. Threaded mode switch — stops the VLM, CMA-preflights, starts OWL (and back). Mutually exclusive on 8 GB (`jarvis-owl` `Conflicts=jarvis-vlm`); the dashboard survives (voice `Wants=` not `Requires=`). Poll `perception_mode`/`owl_up` in `/metrics`; draw live boxes via `POST /tool/detect_objects` |

### Power management (FULL / ECO / OFF)
| Method | Path | Purpose |
|---|---|---|
| POST | `/power` | Body `{action: "eco"\|"wake"\|"shutdown"\|"reboot"}`. **eco** stops the VLM + pauses the captioner (mic/wake-word stay alive, ~7.6 W); **wake** restarts the VLM (~50-90 s) and resets the idle timer; **shutdown**/**reboot** fire `systemctl poweroff`/`reboot` after a short grace. Eco/wake run in a thread — poll `power_state` in `/metrics` |

`/metrics` adds `power_state` (`full`\|`eco`) and `power_transition` (e.g.
`"waking"`, `"entering eco"`, `"shutting down"`). Voice equivalents: the STT
phrases "wake up", "eco mode", "shut down", "reboot" route to `/power` *before*
the VLM (so they work in eco). Auto-eco fires after 15 min idle. **Boot-to-eco**:
jarvis-vlm is disabled from auto-start; jarvis-voice boots standalone and detects
state via `systemctl is-active jarvis-vlm`. **OFF can only be undone by the
physical button** — no voice/network wake.

### Visual-memory capture (live ticker)
`POST /memory/visual/capture` accepts `{bg:true}` → background caption
(`priority=False`: yields to interactive turns, skips if the VLM is busy). Used by
the live detection ticker to keep the current-view objects fresh.

### Training-dataset export ("robotics data engine")
| Method | Path | Purpose |
|---|---|---|
| POST | `/dataset/export` | build a portable dataset from visual memory + grounded Q&A. Body `{since_days?, limit?, include_frames?, include_turns?, consent?:{...}}` → `{ok, name, counts, bytes, card_url, download_url}`. Writes Open-X/LeRobot-friendly JSONL (`vision_language.jsonl`, `visual_qa.jsonl`) + copied frames + `meta/info.json` + `meta/consent.json` + `DATASET_CARD.md` |
| GET | `/dataset/exports` | list prior exports with counts |
| GET | `/dataset/card/<name>` | the dataset card (markdown) |
| GET | `/dataset/dl/<name>.zip` | download the bundle as a zip (zipped on demand) |
