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
