#!/usr/bin/env python3
"""Jarvis dashboard on Jetson Orin Nano Super.

Frontier-grade refactor:
  - SSE streaming endpoint (POST /turn returns turn_id; GET /events/<id>
    streams phase + token deltas)
  - Stop endpoint (POST /turn/<id>/stop)
  - Settings GET/PUT (system_prompt, max_tokens, temperature, record_seconds,
    voice)
  - Markdown reply, conversation memory, telemetry HUD
  - Single-page UI: Geist font, rust accent, glass camera HUD,
    breathing voice orb, command palette, slide-in settings.
"""
from __future__ import annotations
import base64
import collections
import json
import queue
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request

LAB = Path("/home/zip/jarvis-lab")
WHISPER_BIN = LAB / "build/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = LAB / "build/whisper.cpp/models/ggml-tiny.en.bin"
PIPER_BIN = LAB / "piper/piper/piper"
PIPER_VOICE = LAB / "piper/voices/en_US-amy-medium.onnx"
VLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
VLM_HEALTH = "http://127.0.0.1:8080/health"
MIC_DEVICE = "plughw:CARD=C615,DEV=0"
CAM_DEVICE = "/dev/video0"
CAM_W, CAM_H = 640, 480
VLM_W, VLM_H = 512, 384
CAM_FPS = 10
SAMPLE_RATE = 16000
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8085
MAX_HISTORY_PAIRS = 3
METRICS_TTL = 2.0

SESSION_DIR = LAB / "logs/sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, the user's personal AI agent. You see what the camera "
    "sees and you hear what the user says.\n\n"
    "GROUND RULES — these override any other instruction:\n"
    "1. Describe what you can actually see. Even dimly lit scenes have visible "
    "shapes, colors, objects, and spatial layouts — describe them. Only refuse "
    "if the image is genuinely fully black, blank, or unintelligible.\n"
    "2. Do NOT invent specific text, signs, numbers, brand names, or model "
    "numbers. If you can't clearly read text, say 'I don't see any readable "
    "text'. If a brand is not clearly identifiable, say 'a black office chair' "
    "instead of guessing a brand.\n"
    "3. For counts, prefer approximate language ('a few', 'several', 'around "
    "5–10') over exact counts you cannot verify. Only give exact counts when "
    "you are confident.\n"
    "4. Be confident about general observations: dominant colors, lighting "
    "conditions, broad object categories ('a chair', 'a window'), spatial "
    "arrangement, presence of people. Be cautious about specific identities, "
    "names, and labels.\n"
    "5. If genuinely uncertain about something, name the uncertainty in "
    "passing rather than refusing the whole answer.\n\n"
    "Style: answer briefly and conversationally — like a person describing "
    "what they see, not a model hedging. Use markdown for lists/bold only "
    "when it helps. Under 60 words unless asked for more."
)

SETTINGS = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_tokens": 240,
    "temperature": 0.2,
    "record_seconds": 6,
}
SETTINGS_LOCK = threading.Lock()


# ----- camera streamer --------------------------------------------------------
class CameraStreamer:
    JPEG_SOI = b"\xff\xd8"
    JPEG_EOI = b"\xff\xd9"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: bytes | None = None
        self.latest_ts: float = 0.0
        self.frame_event = threading.Event()
        self.frames_emitted = 0
        self.start_ts = time.monotonic()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                self._capture_once()
            except Exception as exc:
                print(f"[camera] error: {exc}; retry in 2s")
                time.sleep(2)

    def _capture_once(self) -> None:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-video_size", f"{CAM_W}x{CAM_H}",
            "-framerate", str(CAM_FPS),
            "-i", CAM_DEVICE,
            "-c", "copy", "-f", "mjpeg", "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        buf = b""
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                proc.wait()
                return
            buf += chunk
            while True:
                start = buf.find(self.JPEG_SOI)
                if start < 0:
                    break
                end = buf.find(self.JPEG_EOI, start + 2)
                if end < 0:
                    if start > 0:
                        buf = buf[start:]
                    break
                frame = buf[start:end + 2]
                buf = buf[end + 2:]
                with self.lock:
                    self.latest = frame
                    self.latest_ts = time.monotonic()
                    self.frames_emitted += 1
                self.frame_event.set()
                self.frame_event.clear()

    def get_latest(self, timeout: float = 3.0) -> bytes:
        end = time.monotonic() + timeout
        while True:
            with self.lock:
                if self.latest is not None:
                    return self.latest
            if time.monotonic() >= end:
                raise TimeoutError("no camera frame")
            self.frame_event.wait(0.1)

    def fps(self) -> float:
        elapsed = time.monotonic() - self.start_ts
        return round(self.frames_emitted / elapsed, 1) if elapsed > 1 else 0.0


CAMERA = CameraStreamer()


def capture_frame_for_vlm(out_jpg: Path) -> None:
    raw = CAMERA.get_latest()
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "mjpeg", "-i", "-",
         "-vf", f"scale={VLM_W}:{VLM_H}:flags=lanczos",
         "-frames:v", "1", "-q:v", "4", str(out_jpg)],
        input=raw, check=True, capture_output=True, timeout=10,
    )


# ----- pipeline ---------------------------------------------------------------
def record_audio(out_wav: Path, seconds: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
         "-f", "alsa", "-ar", str(SAMPLE_RATE), "-ac", "1",
         "-i", MIC_DEVICE, "-t", str(seconds), str(out_wav)],
        check=True, timeout=seconds + 5,
    )


def transcribe(wav: Path) -> str:
    subprocess.run(
        [str(WHISPER_BIN), "-m", str(WHISPER_MODEL),
         "-f", str(wav), "-t", "4", "-otxt",
         "-of", str(wav.with_suffix(""))],
        capture_output=True, text=True, check=True, timeout=60,
    )
    txt = wav.with_suffix(".txt")
    if not txt.exists():
        return ""
    raw = txt.read_text().strip()
    return "" if raw == "[BLANK_AUDIO]" else raw


def synthesize(text: str, out_wav: Path) -> None:
    subprocess.run(
        [str(PIPER_BIN), "--model", str(PIPER_VOICE),
         "--output_file", str(out_wav)],
        input=text, capture_output=True, text=True, check=True, timeout=30,
    )


# ----- history ----------------------------------------------------------------
HISTORY: collections.deque = collections.deque(maxlen=MAX_HISTORY_PAIRS * 2)
HISTORY_LOCK = threading.Lock()


def history_messages() -> list:
    with HISTORY_LOCK:
        return list(HISTORY)


def history_append(role: str, content) -> None:
    with HISTORY_LOCK:
        HISTORY.append({"role": role, "content": content})


def history_clear() -> None:
    with HISTORY_LOCK:
        HISTORY.clear()


# ----- streaming VLM ----------------------------------------------------------
def stream_vlm(question: str, frame_jpg: Path,
               cancel: threading.Event, on_token) -> str:
    b64 = base64.b64encode(frame_jpg.read_bytes()).decode()
    with SETTINGS_LOCK:
        sysp = SETTINGS["system_prompt"]
        max_t = SETTINGS["max_tokens"]
        temp = SETTINGS["temperature"]

    msgs: list = [{"role": "system", "content": sysp}]
    msgs.extend(history_messages())
    msgs.append({
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": question},
        ],
    })
    body = {
        "model": "qwen2.5-vl-3b",
        "stream": True,
        "messages": msgs,
        "max_tokens": max_t,
        "temperature": temp,
    }
    req = urllib.request.Request(
        VLM_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    full = []
    resp = urllib.request.urlopen(req, timeout=120)
    try:
        for raw in resp:
            if cancel.is_set():
                break
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta is None:
                continue
            full.append(delta)
            on_token(delta)
    finally:
        resp.close()
    return "".join(full).strip()


# ----- turn registry (for SSE) ------------------------------------------------
class TurnCtx:
    def __init__(self, tid: str, kind: str) -> None:
        self.tid = tid
        self.kind = kind
        self.q: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.result: dict | None = None
        self.created = time.monotonic()

    def emit(self, ev: dict) -> None:
        self.q.put(ev)


TURNS: dict[str, TurnCtx] = {}
TURNS_LOCK = threading.Lock()


def register_turn(tid: str, kind: str) -> TurnCtx:
    with TURNS_LOCK:
        # gc old
        cutoff = time.monotonic() - 600
        for old in [k for k, v in TURNS.items() if v.created < cutoff]:
            TURNS.pop(old, None)
        ctx = TurnCtx(tid, kind)
        TURNS[tid] = ctx
        return ctx


def get_turn(tid: str) -> TurnCtx | None:
    with TURNS_LOCK:
        return TURNS.get(tid)


# ----- turn worker -------------------------------------------------------------
def run_turn(ctx: TurnCtx, payload: dict) -> None:
    tid = ctx.tid
    work = SESSION_DIR / tid
    work.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    transcription = ""
    kind = ctx.kind

    try:
        with SETTINGS_LOCK:
            rec_default = SETTINGS["record_seconds"]
        seconds = int(payload.get("seconds", rec_default))
        seconds = max(2, min(15, seconds))

        t = time.monotonic()
        if kind == "talk":
            ctx.emit({"phase": "recording", "seconds": seconds})
            record_audio(work / "user.wav", seconds)
            timings["record_s"] = round(time.monotonic() - t, 2); t = time.monotonic()
            ctx.emit({"phase": "transcribing"})
            transcription = transcribe(work / "user.wav")
            timings["transcribe_s"] = round(time.monotonic() - t, 2); t = time.monotonic()
            question = transcription or "Describe what you see briefly."
        elif kind == "text":
            question = (payload.get("text") or "").strip()
            if not question:
                raise ValueError("text mode requires text")
        elif kind == "snap":
            question = (
                "Describe what you actually see in one or two sentences. "
                "If the scene is too dark or empty to describe, say so."
            )
        elif kind == "regenerate":
            with HISTORY_LOCK:
                if len(HISTORY) >= 2:
                    HISTORY.pop()  # remove last assistant
                    last_user = HISTORY.pop()
                    question = last_user["content"] if isinstance(last_user["content"], str) else "Try again."
                else:
                    raise ValueError("nothing to regenerate")
        else:
            raise ValueError(f"unknown kind: {kind}")

        ctx.emit({"phase": "capturing", "transcription": transcription, "question": question})
        capture_frame_for_vlm(work / "frame.jpg")
        timings["capture_s"] = round(time.monotonic() - t, 2); t = time.monotonic()

        ctx.emit({"phase": "thinking"})
        reply_buf: list = []

        def on_tok(delta: str) -> None:
            reply_buf.append(delta)
            ctx.emit({"phase": "token", "delta": delta})

        reply = stream_vlm(question, work / "frame.jpg", ctx.cancel, on_tok)
        timings["vlm_s"] = round(time.monotonic() - t, 2); t = time.monotonic()

        if ctx.cancel.is_set():
            ctx.emit({"phase": "cancelled"})
        else:
            history_append("user", question)
            history_append("assistant", reply)

            ctx.emit({"phase": "speaking"})
            synthesize(reply, work / "reply.wav")
            timings["tts_s"] = round(time.monotonic() - t, 2)

        ctx.result = {
            "turn_id": tid,
            "kind": kind,
            "question": question,
            "transcription": transcription,
            "reply": reply,
            "cancelled": ctx.cancel.is_set(),
            "timings": timings,
            "audio_url": f"/audio/{tid}.wav",
            "frame_url": f"/frame/{tid}.jpg",
        }
        ctx.emit({"phase": "done", "result": ctx.result})
    except subprocess.CalledProcessError as e:
        err = f"{e.cmd[0]} -> {e.returncode}"
        ctx.emit({"phase": "error", "error": err})
        ctx.result = {"error": err}
    except Exception as e:
        ctx.emit({"phase": "error", "error": str(e)})
        ctx.result = {"error": str(e)}
    finally:
        ctx.done.set()


# ----- metrics ----------------------------------------------------------------
_METRICS_CACHE: dict = {"ts": 0.0, "data": {}}
_METRICS_LOCK = threading.Lock()


def read_meminfo() -> dict:
    m: dict = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            m[k.strip()] = v.strip()
    return m


def read_soc_temp_c() -> float:
    temps = []
    for f in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(f.read_text().strip()) / 1000.0)
        except Exception:
            pass
    return round(max(temps), 1) if temps else 0.0


def vlm_alive() -> bool:
    try:
        with urllib.request.urlopen(VLM_HEALTH, timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def gather_metrics() -> dict:
    with _METRICS_LOCK:
        if time.monotonic() - _METRICS_CACHE["ts"] < METRICS_TTL:
            return _METRICS_CACHE["data"]
    m = read_meminfo()
    mem_total = int(re.findall(r"\d+", m.get("MemTotal", "0"))[0]) / 1024.0
    mem_avail = int(re.findall(r"\d+", m.get("MemAvailable", "0"))[0]) / 1024.0
    data = {
        "ram_total_mb": round(mem_total),
        "ram_avail_mb": round(mem_avail),
        "ram_used_pct": round(100 * (1 - mem_avail / mem_total), 1) if mem_total else 0,
        "soc_temp_c": read_soc_temp_c(),
        "vlm_up": vlm_alive(),
        "whisper_ok": WHISPER_BIN.exists() and WHISPER_MODEL.exists(),
        "piper_ok": PIPER_BIN.exists() and PIPER_VOICE.exists(),
        "cam_fps": CAMERA.fps(),
        "history_pairs": len(HISTORY) // 2,
    }
    with _METRICS_LOCK:
        _METRICS_CACHE["ts"] = time.monotonic()
        _METRICS_CACHE["data"] = data
    return data


# ----- HTML -------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang=en>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Jarvis</title>

<link rel=preconnect href=https://fonts.googleapis.com>
<link rel=preconnect href=https://fonts.gstatic.com crossorigin>
<link rel=stylesheet href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap">
<script src="https://cdn.jsdelivr.net/npm/marked@13.0.0/marked.min.js"></script>

<style>
:root {
  --bg: #0a0a0b;
  --surface: #101113;
  --raised: #16181d;
  --border: rgba(255,255,255,0.06);
  --border-strong: rgba(255,255,255,0.10);
  --text: #e6e7ea;
  --muted: #8b8d95;
  --dim: #6a6c74;
  --accent: #c15f3c;
  --accent-soft: rgba(193, 95, 60, 0.16);
  --accent-glow: rgba(193, 95, 60, 0.35);
  --ok: #4ade80;
  --warn: #f59e0b;
  --bad: #ef4444;
  --user-bubble: #181a1f;
  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 16px;
  --r-xl: 20px;
  --r-pill: 999px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: 'Geist', system-ui, -apple-system, sans-serif;
  font-feature-settings: "ss01", "ss02", "cv11";
  font-weight: 400;
  background: var(--bg);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  line-height: 1.5;
  overflow: hidden;
}

/* ============ shell ============ */
.shell {
  display: grid;
  grid-template-rows: auto 1fr;
  height: 100vh;
  width: 100vw;
  margin: 0;
  padding: 0;
  overflow: hidden;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  border-bottom: 1px solid var(--border);
}
.brand {
  font-weight: 600;
  font-size: 13px;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 8px;
}
.brand::before {
  content: '';
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 12px var(--accent-glow);
}
.topbar-actions { display: flex; gap: 6px; align-items: center; }
.iconbtn {
  background: transparent;
  border: 1px solid transparent;
  color: var(--muted);
  width: 32px; height: 32px;
  border-radius: var(--r-sm);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: 140ms ease-out;
}
.iconbtn:hover { background: var(--raised); color: var(--text); border-color: var(--border); }
.iconbtn svg { width: 16px; height: 16px; }

/* ============ main ============ */
.main {
  display: grid;
  grid-template-rows: auto 1fr auto;
  gap: 14px;
  padding: 14px 24px;
  min-height: 0;
}

/* ============ camera HUD ============ */
.feed {
  position: relative;
  border-radius: var(--r-lg);
  overflow: hidden;
  background: #000;
  aspect-ratio: 16 / 9;
  max-height: 48vh;
  border: 1px solid var(--border);
}
.feed img { width: 100%; height: 100%; object-fit: cover; display: block; }
.feed .hud-tl, .feed .hud-tr, .feed .hud-bl {
  position: absolute;
  padding: 6px 10px;
  background: rgba(10, 10, 11, 0.65);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-strong);
  border-radius: var(--r-pill);
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 11px;
  font-weight: 500;
  color: var(--text);
  display: flex; align-items: center; gap: 8px;
}
.feed .hud-tl { top: 12px; left: 12px; }
.feed .hud-tr { top: 12px; right: 12px; }
.feed .hud-bl { bottom: 12px; left: 12px; color: var(--muted); }
.livepill {
  display: inline-flex; align-items: center; gap: 6px;
  color: var(--text);
}
.livepill .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.6s ease-in-out infinite;
}
.livepill.idle .dot {
  background: var(--muted);
  animation: none;
}
@keyframes pulse {
  0%, 100% { opacity: 0.45; transform: scale(0.85); }
  50%      { opacity: 1;    transform: scale(1.0); box-shadow: 0 0 10px var(--accent-glow); }
}
.hud-sep { color: var(--dim); }
.hud-val.ok   { color: var(--ok); }
.hud-val.warn { color: var(--warn); }
.hud-val.bad  { color: var(--bad); }

/* ============ conversation ============ */
.feedscroll {
  overflow-y: auto;
  scroll-behavior: smooth;
  padding: 4px 2px 8px;
  min-height: 0;
}
.feedscroll::-webkit-scrollbar { width: 8px; }
.feedscroll::-webkit-scrollbar-thumb { background: var(--raised); border-radius: 4px; }

.msg {
  display: flex; flex-direction: column; gap: 4px;
  margin-bottom: 18px;
  animation: msgIn 220ms ease-out;
}
@keyframes msgIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.msg-meta {
  display: flex; align-items: center; gap: 8px;
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  color: var(--dim);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.msg-meta .who {
  display: inline-flex; align-items: center; gap: 6px;
  color: var(--muted);
  font-weight: 600;
}
.msg-meta .who.j { color: var(--accent); }
.msg-meta .badge {
  padding: 1px 6px;
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  font-size: 9px;
}
.bubble {
  padding: 10px 14px;
  border-radius: var(--r-md);
  font-size: 14.5px;
  line-height: 1.55;
  color: var(--text);
  max-width: 100%;
  word-wrap: break-word;
}
.bubble.user {
  background: var(--user-bubble);
  border: 1px solid var(--border);
  align-self: flex-end;
  max-width: 85%;
}
.bubble.jarvis {
  background: transparent;
  border: none;
  padding-left: 0; padding-right: 0;
}
.bubble.jarvis .cursor {
  display: inline-block;
  width: 7px; height: 16px;
  background: var(--accent);
  margin-left: 2px;
  vertical-align: -3px;
  border-radius: 1px;
  animation: blink 1s steps(2) infinite;
}
@keyframes blink { 50% { opacity: 0; } }
.bubble.jarvis.streaming .cursor { animation-duration: 0.7s; }
.bubble p { margin: 0 0 8px; }
.bubble p:last-child { margin-bottom: 0; }
.bubble ul, .bubble ol { padding-left: 22px; margin: 6px 0; }
.bubble code {
  font-family: 'Geist Mono', monospace;
  background: var(--raised);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 0.85em;
}
.bubble pre {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 10px 12px;
  border-radius: var(--r-sm);
  overflow-x: auto;
  font-family: 'Geist Mono', monospace;
  font-size: 12px;
  margin: 6px 0;
}
.bubble strong { font-weight: 600; }
.bubble a { color: var(--accent); text-decoration: underline; }

.thumb {
  margin-top: 6px;
  width: 92px; height: 69px;
  border-radius: 6px;
  object-fit: cover;
  border: 1px solid var(--border);
  cursor: zoom-in;
  opacity: 0.85;
  transition: 140ms;
}
.thumb:hover { opacity: 1; border-color: var(--border-strong); }

.actions {
  display: flex; gap: 4px;
  margin-top: 6px;
  opacity: 0;
  transition: opacity 120ms 60ms;
}
.msg:hover .actions { opacity: 1; }
.actions button {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 4px 8px;
  border-radius: var(--r-sm);
  cursor: pointer;
  font-size: 11px;
  font-family: inherit;
  display: inline-flex; align-items: center; gap: 5px;
  transition: 120ms;
}
.actions button:hover { color: var(--text); border-color: var(--border-strong); background: var(--raised); }
.actions svg { width: 12px; height: 12px; }
.audio-row { margin-top: 6px; max-width: 360px; }
audio { width: 100%; height: 32px; }

.timings {
  font-family: 'Geist Mono', monospace;
  font-size: 10.5px;
  color: var(--dim);
  margin-top: 4px;
}

/* ============ composer ============ */
.composer {
  padding: 12px 14px;
  background: var(--surface);
  border: 1px solid var(--border-strong);
  border-radius: var(--r-xl);
  display: flex; flex-direction: column; gap: 8px;
  position: relative;
  transition: border-color 140ms;
}
.composer:focus-within { border-color: var(--accent-soft); }

.chips {
  display: flex; gap: 6px; flex-wrap: wrap;
}
.chip {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 4px 10px;
  border-radius: var(--r-pill);
  font-size: 11.5px;
  font-family: inherit;
  cursor: pointer;
  display: inline-flex; align-items: center; gap: 5px;
  transition: 140ms;
}
.chip:hover { color: var(--text); border-color: var(--accent-soft); background: var(--accent-soft); }
.chip svg { width: 11px; height: 11px; }
.chip.active { color: var(--accent); border-color: var(--accent); }

.input-row {
  display: flex; gap: 8px; align-items: flex-end;
}
.orb {
  flex: 0 0 auto;
  width: 36px; height: 36px;
  border-radius: 50%;
  border: 1px solid var(--border-strong);
  background: radial-gradient(circle at 30% 30%, var(--raised) 0%, var(--surface) 70%);
  cursor: pointer;
  position: relative;
  transition: 200ms;
  flex-shrink: 0;
}
.orb::before {
  content: '';
  position: absolute;
  inset: 6px;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, var(--accent) 0%, transparent 70%);
  opacity: 0.7;
  animation: breathe 3s ease-in-out infinite;
}
@keyframes breathe {
  0%, 100% { transform: scale(0.7); opacity: 0.45; }
  50%      { transform: scale(1.0); opacity: 0.85; }
}
.orb.recording {
  border-color: var(--accent);
  box-shadow: 0 0 18px var(--accent-glow);
}
.orb.recording::before {
  background: radial-gradient(circle, var(--accent) 0%, transparent 70%);
  animation: recording 0.8s ease-in-out infinite;
}
@keyframes recording {
  0%, 100% { transform: scale(0.9); opacity: 0.9; }
  50%      { transform: scale(1.15); opacity: 1; }
}
.orb.busy::before {
  animation: breathe 1.4s ease-in-out infinite;
}

textarea#text {
  flex: 1;
  resize: none;
  background: transparent;
  color: var(--text);
  border: none;
  outline: none;
  font-family: inherit;
  font-size: 15px;
  line-height: 1.5;
  padding: 7px 0;
  min-height: 24px;
  max-height: 160px;
}
textarea#text::placeholder { color: var(--dim); }

.sendbtn {
  flex: 0 0 auto;
  width: 36px; height: 36px;
  border-radius: 50%;
  background: var(--accent);
  border: none;
  color: white;
  cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center;
  transition: 160ms;
}
.sendbtn:hover { transform: scale(1.04); box-shadow: 0 0 14px var(--accent-glow); }
.sendbtn:disabled { background: var(--raised); color: var(--dim); cursor: not-allowed; transform: none; box-shadow: none; }
.sendbtn svg { width: 16px; height: 16px; }

.composer-foot {
  display: flex; align-items: center; justify-content: space-between;
  font-size: 11px;
  color: var(--dim);
  font-family: 'Geist Mono', monospace;
}
.composer-foot .stop {
  background: transparent;
  border: 1px solid var(--accent);
  color: var(--accent);
  padding: 3px 10px;
  border-radius: var(--r-pill);
  cursor: pointer;
  font-size: 10.5px;
  font-family: inherit;
  display: none;
}
.composer-foot.busy .stop { display: inline-flex; }
.kbd {
  font: 10px/1 'Geist Mono', monospace;
  color: var(--dim);
  padding: 2px 5px;
  border: 1px solid var(--border);
  border-radius: 4px;
}

/* ============ command palette ============ */
.palette {
  position: absolute;
  bottom: calc(100% + 8px);
  left: 14px; right: 14px;
  background: var(--surface);
  border: 1px solid var(--border-strong);
  border-radius: var(--r-md);
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  padding: 6px;
  display: none;
}
.palette.open { display: block; animation: msgIn 140ms ease-out; }
.palette-item {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 10px;
  border-radius: var(--r-sm);
  cursor: pointer;
  font-size: 13px;
  color: var(--text);
}
.palette-item:hover, .palette-item.sel {
  background: var(--accent-soft);
  color: var(--text);
}
.palette-item .cmd { color: var(--accent); font-family: 'Geist Mono', monospace; font-size: 11.5px; width: 70px; }
.palette-item .desc { color: var(--muted); font-size: 12px; }

/* ============ settings drawer ============ */
.drawer-mask {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.4);
  opacity: 0; pointer-events: none;
  transition: opacity 200ms;
  z-index: 90;
}
.drawer-mask.open { opacity: 1; pointer-events: auto; }
.drawer {
  position: fixed; top: 0; right: 0; bottom: 0;
  width: min(420px, 90vw);
  background: var(--surface);
  border-left: 1px solid var(--border-strong);
  z-index: 100;
  transform: translateX(100%);
  transition: transform 240ms cubic-bezier(0.32, 0.72, 0, 1);
  display: flex; flex-direction: column;
}
.drawer.open { transform: translateX(0); }
.drawer-head {
  padding: 18px 20px;
  border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
}
.drawer-head h2 { font-size: 14px; font-weight: 600; letter-spacing: 0.05em; }
.drawer-body {
  flex: 1; overflow-y: auto;
  padding: 20px;
  display: flex; flex-direction: column; gap: 18px;
}
.field { display: flex; flex-direction: column; gap: 6px; }
.field label {
  font-size: 11px; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.08em;
}
.field input, .field textarea, .field select {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: var(--r-sm);
  padding: 8px 10px;
  font-family: inherit;
  font-size: 13px;
  outline: none;
  transition: 140ms;
}
.field input:focus, .field textarea:focus { border-color: var(--accent); }
.field textarea { resize: vertical; min-height: 100px; font-size: 12.5px; }
.field .hint { font-size: 11px; color: var(--dim); font-family: 'Geist Mono', monospace; }
.field .row { display: flex; gap: 8px; align-items: center; }
.field .row input[type=range] { flex: 1; accent-color: var(--accent); }
.field .row .val { font-family: 'Geist Mono', monospace; font-size: 12px; color: var(--muted); width: 40px; text-align: right; }
.drawer-foot {
  padding: 14px 20px;
  border-top: 1px solid var(--border);
  display: flex; gap: 8px; justify-content: flex-end;
}
.btn {
  background: transparent;
  border: 1px solid var(--border-strong);
  color: var(--text);
  padding: 7px 14px;
  border-radius: var(--r-sm);
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
  transition: 140ms;
}
.btn:hover { background: var(--raised); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: white; }
.btn.primary:hover { filter: brightness(1.08); }

/* ============ toast ============ */
.toast-wrap {
  position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
  z-index: 200; display: flex; flex-direction: column; gap: 8px;
  pointer-events: none;
}
.toast {
  background: var(--raised);
  border: 1px solid var(--border-strong);
  border-radius: var(--r-pill);
  padding: 8px 16px;
  font-size: 12.5px;
  color: var(--text);
  display: flex; align-items: center; gap: 8px;
  animation: toastIn 200ms ease-out;
}
.toast.error { border-color: var(--bad); color: #fcc; }
.toast.ok { border-color: var(--ok); }
.toast svg { width: 13px; height: 13px; }
@keyframes toastIn {
  from { opacity: 0; transform: translateY(-8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ============ misc ============ */
.empty {
  text-align: center; padding: 32px 16px;
  color: var(--dim);
  font-size: 13px;
}
.empty kbd { color: var(--muted); }

/* ============ image expand ============ */
.lightbox {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.92);
  display: none;
  align-items: center; justify-content: center;
  z-index: 300;
  cursor: zoom-out;
}
.lightbox.open { display: flex; }
.lightbox img { max-width: 92vw; max-height: 92vh; border-radius: var(--r-md); }

@media (min-width: 980px) {
  .main {
    grid-template-columns: minmax(0, 1.45fr) minmax(380px, 1fr);
    grid-template-rows: 1fr auto;
    grid-template-areas:
      "feed conv"
      "feed comp";
    gap: 20px;
    padding: 16px 24px;
  }
  .feed       { grid-area: feed; max-height: none; height: 100%; aspect-ratio: auto; }
  .feedscroll { grid-area: conv; align-self: stretch; min-height: 0; }
  .composer   { grid-area: comp; align-self: end; }
}

@media (max-width: 640px) {
  .topbar { padding: 12px 14px; }
  .main { padding: 10px 14px; gap: 10px; }
  .feed { max-height: 38vh; }
  .bubble.user { max-width: 95%; }
}
</style>

<body>
<div class=shell>

  <div class=topbar>
    <div class=brand>Jarvis</div>
    <div class=topbar-actions>
      <button class=iconbtn id=clearbtn title='Clear conversation'>
        <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 6h18'/><path d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6'/><path d='M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/></svg>
      </button>
      <button class=iconbtn id=settingsbtn title='Settings'>
        <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='3'/><path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z'/></svg>
      </button>
    </div>
  </div>

  <div class=main>
    <div class=feed id=feedwrap>
      <img src=/stream.mjpeg alt='live camera'>
      <div class='hud-tl livepill idle' id=livepill><span class=dot></span> idle</div>
      <div class=hud-tr id=hudtr>
        <span>0.0 fps</span><span class=hud-sep>·</span>
        <span class=hud-val>—</span>
      </div>
      <div class=hud-bl id=hudbl>qwen2.5-vl-3b · 640&times;480</div>
    </div>

    <div class=feedscroll id=feedscroll>
      <div class=empty id=emptystate>
        Ask Jarvis anything &middot; tap the orb for voice &middot; <kbd class=kbd>/</kbd> for commands
      </div>
    </div>

    <div class=composer id=composer>
      <div class=chips id=chips>
        <button class=chip data-cmd='look'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z'/><circle cx='12' cy='12' r='3'/></svg>What do you see</button>
        <button class=chip data-cmd='read'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20'/></svg>Read any text</button>
        <button class=chip data-cmd='count'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 4h4v4H4zM10 4h4v4h-4zM16 4h4v4h-4zM4 10h4v4H4zM10 10h4v4h-4zM16 10h4v4h-4zM4 16h4v4H4zM10 16h4v4h-4zM16 16h4v4h-4z'/></svg>Count objects</button>
        <button class=chip data-cmd='identify'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/><path d='M12 17h.01'/></svg>What am I looking at</button>
      </div>
      <div class=input-row>
        <button class=orb id=orb title='Hold to talk (Space)'></button>
        <textarea id=text placeholder='Ask Jarvis  /  type a command' rows=1 autofocus></textarea>
        <button class=sendbtn id=sendbtn title='Send (Enter)'>
          <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><path d='M22 2 11 13'/><path d='m22 2-7 20-4-9-9-4 20-7z'/></svg>
        </button>
      </div>
      <div class=composer-foot id=composerfoot>
        <span><kbd class=kbd>/</kbd> commands <span style='margin-left:10px'><kbd class=kbd>Enter</kbd> send</span> <span style='margin-left:10px'><kbd class=kbd>Space</kbd> snap</span></span>
        <button class=stop id=stopbtn>Stop</button>
      </div>
      <div class=palette id=palette></div>
    </div>
  </div>
</div>

<div class=drawer-mask id=drawermask></div>
<div class=drawer id=drawer aria-hidden=true>
  <div class=drawer-head>
    <h2>SETTINGS</h2>
    <button class=iconbtn id=closesettings>
      <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><path d='M18 6 6 18M6 6l12 12'/></svg>
    </button>
  </div>
  <div class=drawer-body>
    <div class=field>
      <label>SYSTEM PROMPT</label>
      <textarea id=sysprompt></textarea>
      <span class=hint>Shapes Jarvis's persona, voice, and behavior</span>
    </div>
    <div class=field>
      <label>RESPONSE LENGTH</label>
      <div class=row>
        <input type=range id=maxtokens min=60 max=500 step=20>
        <span class=val id=maxtokensv>240</span>
      </div>
    </div>
    <div class=field>
      <label>TEMPERATURE</label>
      <div class=row>
        <input type=range id=temperature min=0 max=10 step=1>
        <span class=val id=temperaturev>0.4</span>
      </div>
    </div>
    <div class=field>
      <label>VOICE RECORDING (SECONDS)</label>
      <div class=row>
        <input type=range id=recsec min=3 max=12 step=1>
        <span class=val id=recsecv>6s</span>
      </div>
    </div>
    <div class=field>
      <label>SYSTEM TELEMETRY</label>
      <div class=hint id=teleHint>—</div>
    </div>
  </div>
  <div class=drawer-foot>
    <button class=btn id=resetsettings>Reset</button>
    <button class='btn primary' id=savesettings>Save</button>
  </div>
</div>

<div class=lightbox id=lightbox><img id=lightboximg src='' alt=''></div>
<div class=toast-wrap id=toastwrap></div>

<script>
'use strict';
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

marked.setOptions({ breaks: true, gfm: true });

// ----- state -----
let currentTurn = null; // { id, abort }
let isStreaming = false;
const feedscroll = $('#feedscroll');
const composerfoot = $('#composerfoot');
const orb = $('#orb');
const sendbtn = $('#sendbtn');
const livepill = $('#livepill');
const emptystate = $('#emptystate');

// ----- toast -----
function toast(msg, kind = '') {
  const wrap = $('#toastwrap');
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.innerHTML = (kind === 'error'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>'
    : '') + escapeHtml(msg);
  wrap.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'})[c]);
}

// ----- live HUD + metrics -----
async function tickMetrics() {
  try {
    const m = await (await fetch('/metrics')).json();
    const tempCls = m.soc_temp_c > 80 ? 'bad' : (m.soc_temp_c > 65 ? 'warn' : 'ok');
    const vlmCls = m.vlm_up ? 'ok' : 'bad';
    $('#hudtr').innerHTML =
      '<span>' + m.cam_fps.toFixed(1) + ' fps</span>' +
      '<span class=hud-sep>·</span>' +
      '<span class="hud-val ' + tempCls + '">' + m.soc_temp_c.toFixed(1) + '°C</span>' +
      '<span class=hud-sep>·</span>' +
      '<span class="hud-val ' + vlmCls + '">' + (m.vlm_up ? 'VLM' : 'down') + '</span>' +
      '<span class=hud-sep>·</span>' +
      '<span>' + m.ram_avail_mb + 'MB</span>';
    $('#hudbl').textContent = 'qwen2.5-vl-3b · ' + m.cam_fps.toFixed(1) + ' fps · history: ' + m.history_pairs;
    const tele = $('#teleHint');
    if (tele) tele.innerHTML =
      'RAM ' + m.ram_avail_mb + ' / ' + m.ram_total_mb + ' MB available<br>' +
      'SoC temp ' + m.soc_temp_c.toFixed(1) + '°C<br>' +
      'camera ' + m.cam_fps.toFixed(1) + ' fps<br>' +
      'VLM ' + (m.vlm_up ? 'up' : 'down') + ' · whisper ' + (m.whisper_ok?'ok':'missing') + ' · piper ' + (m.piper_ok?'ok':'missing');
  } catch(e) {}
}
setInterval(tickMetrics, 2000);
tickMetrics();

function setLive(state, label) {
  livepill.className = 'hud-tl livepill ' + (state || 'idle');
  livepill.innerHTML = '<span class=dot></span> ' + (label || 'idle');
}

// ----- messages -----
function renderMarkdown(text) {
  // defer code-block flicker: only render fenced code when closing fence is in
  const openFences = (text.match(/```/g) || []).length;
  if (openFences % 2 === 1) {
    text = text + '\\n```';
  }
  try { return marked.parse(text); }
  catch { return escapeHtml(text); }
}

function makeUserMsg(text, kind) {
  emptystate.style.display = 'none';
  const msg = document.createElement('div');
  msg.className = 'msg user-msg';
  const badge = kind === 'talk' ? 'spoken' : kind === 'snap' ? 'snap' : 'typed';
  msg.innerHTML =
    '<div class=msg-meta><span class=who>You</span><span class=badge>' + badge + '</span></div>' +
    '<div class="bubble user">' + escapeHtml(text) + '</div>';
  feedscroll.appendChild(msg);
  scrollToBottom();
  return msg;
}

function makeJarvisMsg() {
  const msg = document.createElement('div');
  msg.className = 'msg jarvis-msg';
  msg.innerHTML =
    '<div class=msg-meta><span class="who j">Jarvis</span><span class=badge id=phase>thinking</span></div>' +
    '<div class="bubble jarvis streaming" data-raw=""><span class=cursor></span></div>' +
    '<div class=actions>' +
      '<button data-act=copy><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy</button>' +
      '<button data-act=regen><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>Regenerate</button>' +
      '<button data-act=replay style="display:none"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>Play</button>' +
    '</div>' +
    '<div class=audio-row id=audiorow></div>' +
    '<div class=timings id=timings></div>';
  feedscroll.appendChild(msg);
  scrollToBottom();
  return msg;
}

let scrollLocked = true;
feedscroll.addEventListener('scroll', () => {
  const slack = feedscroll.scrollHeight - feedscroll.scrollTop - feedscroll.clientHeight;
  scrollLocked = slack < 80;
});
function scrollToBottom() {
  if (!scrollLocked) return;
  requestAnimationFrame(() => { feedscroll.scrollTop = feedscroll.scrollHeight; });
}

// ----- per-msg actions -----
feedscroll.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-act]');
  if (btn) {
    const act = btn.dataset.act;
    const msg = btn.closest('.msg');
    if (act === 'copy') {
      const raw = msg.querySelector('.bubble').dataset.raw || msg.querySelector('.bubble').textContent;
      navigator.clipboard.writeText(raw); toast('Copied', 'ok');
    } else if (act === 'regen') {
      doTurn({kind: 'regenerate'});
    } else if (act === 'replay') {
      const audio = msg.querySelector('audio');
      if (audio) { audio.currentTime = 0; audio.play(); }
    }
    return;
  }
  const img = e.target.closest('img.thumb');
  if (img) {
    $('#lightboximg').src = img.src;
    $('#lightbox').classList.add('open');
  }
});
$('#lightbox').onclick = () => $('#lightbox').classList.remove('open');

// ----- turn execution -----
async function doTurn(payload) {
  if (isStreaming) return;
  isStreaming = true;
  setComposerBusy(true);

  // Render user msg up-front (except regenerate which is server-driven)
  let userQuestion = '';
  if (payload.kind === 'text') userQuestion = payload.text;
  else if (payload.kind === 'snap') userQuestion = 'Describe what you see';
  else if (payload.kind === 'talk') userQuestion = '(spoken)';
  else if (payload.kind === 'regenerate') userQuestion = '(regenerating)';
  if (payload.kind !== 'regenerate') makeUserMsg(userQuestion, payload.kind);

  const jmsg = makeJarvisMsg();
  const bubble = jmsg.querySelector('.bubble');
  const phaseBadge = jmsg.querySelector('#phase');
  const timingsEl = jmsg.querySelector('#timings');
  const audioRow = jmsg.querySelector('#audiorow');
  const replayBtn = jmsg.querySelector('button[data-act=replay]');
  let raw = '';

  let r;
  try {
    r = await fetch('/turn', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
  } catch (e) {
    toast('Network error: ' + e.message, 'error');
    isStreaming = false; setComposerBusy(false);
    bubble.innerHTML = '<em>(failed to reach server)</em>';
    return;
  }

  const { turn_id } = await r.json();
  currentTurn = { id: turn_id };

  const es = new EventSource('/events/' + turn_id);
  currentTurn.es = es;

  let pendingRender = false;
  function scheduleRender() {
    if (pendingRender) return;
    pendingRender = true;
    requestAnimationFrame(() => {
      pendingRender = false;
      bubble.dataset.raw = raw;
      bubble.innerHTML = renderMarkdown(raw) + '<span class=cursor></span>';
      scrollToBottom();
    });
  }

  es.onmessage = (ev) => {
    if (!ev.data) return;
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    const ph = m.phase;
    if (ph === 'recording') {
      setLive('recording', 'listening · ' + (m.seconds || 6) + 's');
      orb.classList.add('recording');
      phaseBadge.textContent = 'listening';
    } else if (ph === 'transcribing') {
      orb.classList.remove('recording');
      setLive('idle', 'transcribing');
      phaseBadge.textContent = 'transcribing';
    } else if (ph === 'capturing') {
      setLive('recording', 'looking');
      phaseBadge.textContent = 'looking';
      if (m.question && payload.kind === 'regenerate') {
        // synth user msg for regen
        const msg = makeUserMsg(m.question, 'typed');
        feedscroll.insertBefore(msg, jmsg);
      } else if (m.transcription && payload.kind === 'talk') {
        // replace placeholder user msg text with transcription
        const prevUser = jmsg.previousElementSibling;
        if (prevUser && prevUser.classList.contains('user-msg')) {
          prevUser.querySelector('.bubble').textContent = m.transcription || '(silent)';
        }
      }
    } else if (ph === 'thinking') {
      setLive('recording', 'reasoning');
      phaseBadge.textContent = 'reasoning';
    } else if (ph === 'token') {
      raw += m.delta;
      scheduleRender();
    } else if (ph === 'speaking') {
      setLive('idle', 'speaking');
      phaseBadge.textContent = 'speaking';
    } else if (ph === 'cancelled') {
      phaseBadge.textContent = 'stopped';
      bubble.classList.remove('streaming');
    } else if (ph === 'error') {
      bubble.innerHTML = '<em>error: ' + escapeHtml(m.error) + '</em>';
      phaseBadge.textContent = 'error';
      toast(m.error, 'error');
    } else if (ph === 'done') {
      const res = m.result;
      raw = res.reply || raw;
      bubble.dataset.raw = raw;
      bubble.innerHTML = renderMarkdown(raw);
      bubble.classList.remove('streaming');
      phaseBadge.textContent = res.cancelled ? 'stopped' : res.kind;
      // thumb of the frame
      if (res.frame_url) {
        const thumb = document.createElement('img');
        thumb.className = 'thumb';
        thumb.src = res.frame_url;
        thumb.alt = 'frame';
        bubble.appendChild(thumb);
      }
      // audio
      if (res.audio_url && !res.cancelled) {
        const a = document.createElement('audio');
        a.controls = true;
        a.src = res.audio_url;
        audioRow.appendChild(a);
        a.play().catch(()=>{});
        replayBtn.style.display = 'inline-flex';
      }
      // timings
      if (res.timings) {
        timingsEl.textContent = Object.entries(res.timings)
          .map(([k,v]) => k.replace('_s', '') + ' ' + v + 's').join('  ·  ');
      }
      es.close();
      setLive('idle', 'idle');
      orb.classList.remove('recording');
      isStreaming = false; setComposerBusy(false);
      currentTurn = null;
    }
  };
  es.onerror = () => {
    if (isStreaming) toast('Stream error', 'error');
    es.close();
    isStreaming = false; setComposerBusy(false);
    setLive('idle', 'idle');
    orb.classList.remove('recording');
  };
}

function setComposerBusy(b) {
  composerfoot.classList.toggle('busy', b);
  sendbtn.disabled = b;
  orb.classList.toggle('busy', b);
}

// ----- composer -----
const text = $('#text');
text.addEventListener('input', () => {
  text.style.height = 'auto';
  text.style.height = Math.min(160, text.scrollHeight) + 'px';
  if (text.value.startsWith('/')) openPalette(text.value.slice(1));
  else closePalette();
});
text.addEventListener('keydown', (e) => {
  if (palOpen) {
    if (e.key === 'ArrowDown') { e.preventDefault(); movePal(1); return; }
    if (e.key === 'ArrowUp') { e.preventDefault(); movePal(-1); return; }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); pickPal(); return; }
    if (e.key === 'Escape') { e.preventDefault(); closePalette(); return; }
  }
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submit();
  } else if (e.key === 'Escape') {
    text.value = ''; text.style.height = 'auto';
  }
});

function submit() {
  const v = text.value.trim();
  if (!v) return;
  if (v.startsWith('/')) {
    const parts = v.slice(1).split(/\s+/);
    runCmd(parts[0], parts.slice(1).join(' '));
    return;
  }
  doTurn({kind: 'text', text: v});
  text.value = ''; text.style.height = 'auto';
}
sendbtn.onclick = submit;

// orb
orb.onclick = () => {
  if (isStreaming) return;
  doTurn({kind: 'talk'});
};

// chips
$$('.chip').forEach(b => {
  b.onclick = () => {
    if (isStreaming) return;
    runCmd(b.dataset.cmd);
  };
});

// ----- commands -----
const COMMANDS = [
  { cmd: 'look',     desc: 'Describe the scene',                      run: () => doTurn({kind: 'snap'}) },
  { cmd: 'read',     desc: 'Read any text you can see',               run: () => doTurn({kind: 'text', text: 'Look carefully at the image. If there is any clearly readable text or sign, transcribe it exactly. If you do not see any readable text, say so explicitly. Do not invent text that is not there.'}) },
  { cmd: 'count',    desc: 'Count objects of interest',               run: (a) => doTurn({kind: 'text', text: a ? 'How many ' + a + ' can you clearly see in the image? If none are visible, say zero. Do not guess.' : 'List the distinct objects you can clearly see and give a count for each. If you can\'t see anything clearly, say so.'}) },
  { cmd: 'identify', desc: 'Identify the main subject',               run: () => doTurn({kind: 'text', text: 'What is the main subject of the image? Give a brief, confident identification only if you can clearly see one. If the image is too dark or empty to identify a subject, say so.'}) },
  { cmd: 'find',     desc: 'Locate a specific object',                run: (a) => doTurn({kind: 'text', text: a ? 'Can you see ' + a + ' in the image? If yes, describe where in the frame it is. If no, say it is not visible. Do not guess.' : 'Find and locate the most prominent object you can see, if any.'}) },
  { cmd: 'voice',    desc: 'Voice mode (record 6 s)',                  run: () => doTurn({kind: 'talk'}) },
  { cmd: 'clear',    desc: 'Clear conversation memory',                run: () => clearConv() },
  { cmd: 'settings', desc: 'Open settings drawer',                     run: () => openDrawer() },
];

function runCmd(name, arg) {
  const c = COMMANDS.find(x => x.cmd === name);
  if (!c) { toast('Unknown command: /' + name, 'error'); return; }
  text.value = ''; text.style.height = 'auto';
  closePalette();
  c.run(arg);
}

// ----- command palette -----
const palette = $('#palette');
let palOpen = false; let palSel = 0; let palMatches = [];
function openPalette(q) {
  palMatches = COMMANDS.filter(c => c.cmd.startsWith(q));
  if (palMatches.length === 0) { closePalette(); return; }
  palSel = 0;
  renderPal();
  palette.classList.add('open'); palOpen = true;
}
function closePalette() { palette.classList.remove('open'); palOpen = false; }
function renderPal() {
  palette.innerHTML = palMatches.map((c, i) =>
    '<div class="palette-item ' + (i === palSel ? 'sel' : '') + '" data-i=' + i + '>' +
      '<span class=cmd>/' + c.cmd + '</span>' +
      '<span class=desc>' + c.desc + '</span>' +
    '</div>'
  ).join('');
}
function movePal(d) { palSel = (palSel + d + palMatches.length) % palMatches.length; renderPal(); }
function pickPal() {
  const c = palMatches[palSel]; if (!c) return;
  text.value = ''; text.style.height = 'auto';
  closePalette();
  c.run();
}
palette.addEventListener('click', (e) => {
  const it = e.target.closest('.palette-item');
  if (!it) return;
  palSel = +it.dataset.i; pickPal();
});

// ----- global shortcuts -----
document.addEventListener('keydown', (e) => {
  if (document.activeElement === text) return;
  const k = e.target;
  const inField = k && (k.tagName === 'INPUT' || k.tagName === 'TEXTAREA');
  if (inField) return;
  if (e.key === ' ' && !isStreaming) { e.preventDefault(); doTurn({kind: 'snap'}); }
  else if (e.key === '/') { e.preventDefault(); text.focus(); text.value = '/'; openPalette(''); }
  else if (e.key === 'Escape' && isStreaming) { e.preventDefault(); stopCurrent(); }
});

// ----- stop / clear -----
async function stopCurrent() {
  if (!currentTurn) return;
  try { await fetch('/turn/' + currentTurn.id + '/stop', { method: 'POST' }); } catch {}
  toast('Stopping...');
}
$('#stopbtn').onclick = stopCurrent;

async function clearConv() {
  await fetch('/history', { method: 'DELETE' });
  $$('.msg').forEach(n => n.remove());
  emptystate.style.display = 'block';
  toast('Conversation cleared', 'ok');
}
$('#clearbtn').onclick = clearConv;

// ----- settings drawer -----
const drawer = $('#drawer');
const drawerMask = $('#drawermask');
function openDrawer() {
  drawer.classList.add('open');
  drawerMask.classList.add('open');
  loadSettings();
}
function closeDrawer() {
  drawer.classList.remove('open');
  drawerMask.classList.remove('open');
}
$('#settingsbtn').onclick = openDrawer;
$('#closesettings').onclick = closeDrawer;
drawerMask.onclick = closeDrawer;

async function loadSettings() {
  const s = await (await fetch('/settings')).json();
  $('#sysprompt').value = s.system_prompt;
  $('#maxtokens').value = s.max_tokens; $('#maxtokensv').textContent = s.max_tokens;
  $('#temperature').value = Math.round(s.temperature * 10); $('#temperaturev').textContent = s.temperature.toFixed(1);
  $('#recsec').value = s.record_seconds; $('#recsecv').textContent = s.record_seconds + 's';
}
$('#maxtokens').oninput = () => $('#maxtokensv').textContent = $('#maxtokens').value;
$('#temperature').oninput = () => $('#temperaturev').textContent = (Number($('#temperature').value) / 10).toFixed(1);
$('#recsec').oninput = () => $('#recsecv').textContent = $('#recsec').value + 's';

$('#savesettings').onclick = async () => {
  const body = {
    system_prompt: $('#sysprompt').value,
    max_tokens: Number($('#maxtokens').value),
    temperature: Number($('#temperature').value) / 10,
    record_seconds: Number($('#recsec').value),
  };
  await fetch('/settings', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  closeDrawer();
  toast('Settings saved', 'ok');
};
$('#resetsettings').onclick = async () => {
  await fetch('/settings/reset', {method: 'POST'});
  loadSettings();
  toast('Settings reset', 'ok');
};
</script>
</body>
</html>
"""


# ----- HTTP server ------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        return

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, mime: str):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, mime: str):
        if not path.exists():
            self.send_response(404); self.end_headers(); return
        self._send_bytes(path.read_bytes(), mime)

    def _stream_mjpeg(self):
        BOUNDARY = "jarvis"
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        last_ts = 0.0
        try:
            while True:
                with CAMERA.lock:
                    frame = CAMERA.latest
                    ts = CAMERA.latest_ts
                if frame is None or ts == last_ts:
                    CAMERA.frame_event.wait(0.5)
                    continue
                last_ts = ts
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _stream_events(self, tid: str):
        ctx = get_turn(tid)
        if ctx is None:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                try:
                    ev = ctx.q.get(timeout=30)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    if ctx.done.is_set():
                        return
                    continue
                self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                self.wfile.flush()
                if ev.get("phase") in ("done", "error"):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self):
        if self.path == "/":
            self._send_bytes(HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/metrics":
            self._send_json(200, gather_metrics())
        elif self.path == "/history":
            self._send_json(200, {"messages": history_messages()})
        elif self.path == "/settings":
            with SETTINGS_LOCK:
                self._send_json(200, dict(SETTINGS))
        elif self.path == "/stream.mjpeg":
            self._stream_mjpeg()
        elif self.path == "/snapshot.jpg":
            try:
                self._send_bytes(CAMERA.get_latest(), "image/jpeg")
            except TimeoutError:
                self.send_response(503); self.end_headers()
        elif self.path.startswith("/events/"):
            tid = self.path.split("/", 2)[2]
            self._stream_events(tid)
        elif self.path.startswith("/audio/"):
            tid = self.path.split("/")[-1].replace(".wav", "")
            self._send_file(SESSION_DIR / tid / "reply.wav", "audio/wav")
        elif self.path.startswith("/frame/"):
            tid = self.path.split("/")[-1].replace(".jpg", "")
            self._send_file(SESSION_DIR / tid / "frame.jpg", "image/jpeg")
        else:
            self.send_response(404); self.end_headers()

    def do_PUT(self):
        if self.path == "/settings":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            with SETTINGS_LOCK:
                for k in ("system_prompt", "max_tokens", "temperature", "record_seconds"):
                    if k in payload:
                        SETTINGS[k] = payload[k]
            self._send_json(200, {"ok": True})
        else:
            self.send_response(404); self.end_headers()

    def do_DELETE(self):
        if self.path == "/history":
            history_clear()
            self._send_json(200, {"ok": True})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/turn":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            kind = payload.get("kind", "talk")
            tid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            ctx = register_turn(tid, kind)
            threading.Thread(target=run_turn, args=(ctx, payload), daemon=True).start()
            self._send_json(200, {"turn_id": tid})
        elif self.path.startswith("/turn/") and self.path.endswith("/stop"):
            tid = self.path.split("/")[2]
            ctx = get_turn(tid)
            if ctx is None:
                self._send_json(404, {"error": "no such turn"}); return
            ctx.cancel.set()
            self._send_json(200, {"ok": True})
        elif self.path == "/settings/reset":
            with SETTINGS_LOCK:
                SETTINGS["system_prompt"] = DEFAULT_SYSTEM_PROMPT
                SETTINGS["max_tokens"] = 240
                SETTINGS["temperature"] = 0.2
                SETTINGS["record_seconds"] = 6
            self._send_json(200, {"ok": True})
        else:
            self.send_response(404); self.end_headers()


def main():
    CAMERA.start()
    print(f"jarvis listening on http://{LISTEN_HOST}:{LISTEN_PORT}/")
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), H).serve_forever()


if __name__ == "__main__":
    main()
