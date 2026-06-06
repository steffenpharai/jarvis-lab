#!/usr/bin/env python3
"""Jarvis dashboard on Jetson Orin Nano Super.

Frontier-feature build:
  - SSE streaming turns with phase + token deltas + cancellation
  - Continuous Live Mode (auto-snap + narrate every N seconds, separate
    from conversation memory)
  - Click-on-feed point queries (crop region around tap point, ask VLM
    about just that area)
  - Audio waveform / VAD meter while recording (server-side RMS over SSE)
  - System prompt PRESETS (Focused / Inspector / Companion / Curator)
  - Conversation export to Markdown
  - Pinned replies
  - Per-turn latency breakdown JSON
  - Frame inspector metadata
  - Keyboard shortcut overlay
  - Token / context counters
"""
from __future__ import annotations
import base64
import collections
import contextlib
import io
import json
import queue
import re
import struct
import subprocess
import threading
import time
import uuid
import wave
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
PIN_FILE = LAB / "logs/pinned.json"


# ----- system prompt presets --------------------------------------------------
PRESETS = {
    "focused": (
        "You are Jarvis, the user's personal AI agent. You see what the "
        "camera sees and you hear what the user says.\n\n"
        "GROUND RULES:\n"
        "1. Describe what you can actually see. Even dimly lit scenes have "
        "visible shapes, colors, objects, and spatial layouts — describe them. "
        "Only refuse if the image is genuinely fully black, blank, or "
        "unintelligible.\n"
        "2. Do NOT invent specific text, signs, numbers, brand names, or "
        "model numbers. If you can't clearly read text, say 'I don't see any "
        "readable text'. If a brand is not clearly identifiable, say 'a black "
        "office chair' instead of guessing.\n"
        "3. For counts, prefer approximate language ('a few', 'several', "
        "'around 5-10'). Only give exact counts when you are confident.\n"
        "4. Be confident about general observations (lighting, dominant "
        "colors, broad object categories, spatial arrangement, presence of "
        "people). Be cautious about specific identities, names, and labels.\n"
        "5. If genuinely uncertain about something, name the uncertainty "
        "rather than refusing the whole answer.\n\n"
        "Style: brief, conversational, like a person describing what they "
        "see. Use markdown for lists/bold only when it helps. Under 60 words "
        "unless asked for more."
    ),
    "inspector": (
        "You are Jarvis in Inspector mode — a forensic visual analyst. The "
        "user wants precise, structured observations of the scene.\n\n"
        "PROTOCOL:\n"
        "- Lead with a one-line summary.\n"
        "- Follow with a bulleted breakdown: subjects, environment, lighting, "
        "notable details.\n"
        "- Use precise language. No filler.\n"
        "- Never invent: if you cannot read text, say so. If you cannot "
        "identify a brand or model, say so. Approximate counts only when "
        "exact would be a guess.\n"
        "- Note anything anomalous or noteworthy.\n\n"
        "Be terse. Markdown lists are encouraged. Under 100 words."
    ),
    "companion": (
        "You are Jarvis, a warm conversational companion. The user is "
        "wearing you in a backpack and walks through the world with you.\n\n"
        "Style: friendly, natural, like talking with a thoughtful friend. "
        "Use 'I' freely. Show curiosity about what you see. Ask the user "
        "follow-up questions occasionally if it fits.\n\n"
        "GROUND RULES (still):\n"
        "- Describe what is actually visible.\n"
        "- Don't fabricate text, brands, or specific identities.\n"
        "- For counts, prefer approximate language.\n"
        "- Name uncertainty in passing rather than refusing wholesale.\n\n"
        "Under 50 words usually; warmer than brief."
    ),
    "curator": (
        "You are Jarvis in Curator mode — caption things like a museum or "
        "magazine writer. Brief, evocative, observational.\n\n"
        "- One short paragraph, maybe two.\n"
        "- Sensory and specific where you can be (color, light, texture, "
        "spatial relationships).\n"
        "- Don't fabricate text or specific identifications.\n"
        "- Tone: considered, present-tense, third-person-omniscient.\n\n"
        "Under 60 words."
    ),
}

DEFAULT_SYSTEM_PROMPT = PRESETS["focused"]

SETTINGS = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "preset": "focused",
    "max_tokens": 240,
    "temperature": 0.2,
    "record_seconds": 6,
    "live_interval_s": 8,
    "live_max_observations": 50,
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


def capture_frame_for_vlm(out_jpg: Path, crop: tuple | None = None) -> None:
    """Pull latest frame; if `crop` is (nx, ny, nw, nh) in 0..1 normalized,
    crop to that region then scale to VLM_W x VLM_H. Otherwise scale full
    frame."""
    raw = CAMERA.get_latest()
    vf_parts = []
    if crop is not None:
        nx, ny, nw, nh = crop
        cx = max(0, int(nx * CAM_W))
        cy = max(0, int(ny * CAM_H))
        cw = max(64, min(CAM_W - cx, int(nw * CAM_W)))
        ch = max(64, min(CAM_H - cy, int(nh * CAM_H)))
        vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
    vf_parts.append(f"scale={VLM_W}:{VLM_H}:flags=lanczos")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "mjpeg", "-i", "-",
         "-vf", ",".join(vf_parts),
         "-frames:v", "1", "-q:v", "4", str(out_jpg)],
        input=raw, check=True, capture_output=True, timeout=10,
    )


# ----- pipeline ---------------------------------------------------------------
class AudioMonitor:
    """Broadcasts RMS levels of the latest mic chunk to subscribers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.subscribers: list[queue.Queue] = []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=20)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            with contextlib.suppress(ValueError):
                self.subscribers.remove(q)

    def push(self, level: float) -> None:
        with self.lock:
            for q in list(self.subscribers):
                with contextlib.suppress(queue.Full):
                    q.put_nowait(level)


AUDIO_MONITOR = AudioMonitor()


def record_audio_with_meter(out_wav: Path, seconds: int) -> None:
    """Record from ALSA while broadcasting RMS levels every 50ms."""
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
         "-f", "alsa", "-ar", str(SAMPLE_RATE), "-ac", "1",
         "-i", MIC_DEVICE, "-t", str(seconds),
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
    )
    chunk_samples = SAMPLE_RATE // 20  # 50ms
    chunk_bytes = chunk_samples * 2
    pcm = bytearray()
    try:
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            pcm.extend(data)
            samples = struct.unpack(f"{len(data) // 2}h", data)
            n = len(samples)
            if n:
                rms = (sum(s * s for s in samples) / n) ** 0.5
                AUDIO_MONITOR.push(min(1.0, rms / 8000.0))
    finally:
        proc.wait(timeout=seconds + 5)

    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(pcm))


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


# ----- pins -------------------------------------------------------------------
def load_pins() -> list:
    if PIN_FILE.exists():
        try:
            return json.loads(PIN_FILE.read_text())
        except Exception:
            return []
    return []


def save_pins(pins: list) -> None:
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(json.dumps(pins, indent=2))


PINS: list = load_pins()
PINS_LOCK = threading.Lock()


# ----- streaming VLM ----------------------------------------------------------
def stream_vlm(question: str, frame_jpg: Path,
               cancel: threading.Event, on_token,
               include_history: bool = True,
               deadline_s: float = 60.0) -> tuple[str, dict]:
    b64 = base64.b64encode(frame_jpg.read_bytes()).decode()
    with SETTINGS_LOCK:
        sysp = SETTINGS["system_prompt"]
        max_t = SETTINGS["max_tokens"]
        temp = SETTINGS["temperature"]

    msgs: list = [{"role": "system", "content": sysp}]
    if include_history:
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
    usage = {}
    resp = urllib.request.urlopen(req, timeout=15)

    # Watchdog: if cancel fires (user clicked Stop) or the deadline
    # passes (llama-server got stuck mid-stream), force-close the
    # response so the urllib read() inside the for-loop fails out
    # with an exception we can swallow. urllib's timeout=... only
    # applies to the initial connect / header read; per-chunk reads
    # are NOT interrupted, hence the explicit watchdog.
    closed = {"flag": False}
    def _watchdog() -> None:
        # wait UP TO deadline_s for cancel; close the socket on either
        # outcome so the urllib read loop breaks out
        cancel.wait(deadline_s)
        closed["flag"] = True
        try:
            resp.close()
        except Exception:
            pass
    threading.Thread(target=_watchdog, daemon=True).start()
    deadline_at = time.monotonic() + deadline_s

    try:
        for raw in resp:
            if cancel.is_set():
                break
            if time.monotonic() > deadline_at:
                cancel.set()
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
            if obj.get("usage"):
                usage = obj["usage"]
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta is None:
                continue
            full.append(delta)
            on_token(delta)
    except Exception:
        # Watchdog closed the socket; treat as a clean cancellation
        if not closed["flag"]:
            raise
    finally:
        try:
            resp.close()
        except Exception:
            pass
    return "".join(full).strip(), usage


# ----- turn registry (SSE) ----------------------------------------------------
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

        crop = payload.get("crop")
        if crop and len(crop) == 4:
            crop = tuple(float(c) for c in crop)
        else:
            crop = None

        t = time.monotonic()
        if kind == "talk":
            ctx.emit({"phase": "recording", "seconds": seconds})
            record_audio_with_meter(work / "user.wav", seconds)
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
                "If the scene is genuinely empty or fully black, say so."
            )
        elif kind == "point":
            point = payload.get("point")
            if not point or len(point) != 2:
                raise ValueError("point mode requires point [x, y] 0..1")
            px, py = float(point[0]), float(point[1])
            crop = (max(0.0, px - 0.15), max(0.0, py - 0.15),
                    min(0.30, 1.0 - max(0.0, px - 0.15)),
                    min(0.30, 1.0 - max(0.0, py - 0.15)))
            question = (
                "The user tapped a specific point in the image. Describe "
                "what is at this point or in this region. Be specific and "
                "brief. If unclear, say so."
            )
        elif kind == "regenerate":
            with HISTORY_LOCK:
                if len(HISTORY) >= 2:
                    HISTORY.pop()
                    last_user = HISTORY.pop()
                    question = last_user["content"] if isinstance(last_user["content"], str) else "Try again."
                else:
                    raise ValueError("nothing to regenerate")
        else:
            raise ValueError(f"unknown kind: {kind}")

        ctx.emit({"phase": "capturing", "transcription": transcription,
                  "question": question, "crop": crop})
        capture_frame_for_vlm(work / "frame.jpg", crop=crop)
        timings["capture_s"] = round(time.monotonic() - t, 2); t = time.monotonic()

        ctx.emit({"phase": "thinking"})
        reply_buf: list = []

        def on_tok(delta: str) -> None:
            reply_buf.append(delta)
            ctx.emit({"phase": "token", "delta": delta})

        reply, usage = stream_vlm(question, work / "frame.jpg", ctx.cancel, on_tok)
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
            "usage": usage,
            "audio_url": f"/audio/{tid}.wav",
            "frame_url": f"/frame/{tid}.jpg",
            "crop": list(crop) if crop else None,
            "ts": time.time(),
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


# ----- LIVE MODE (auto-narration) ---------------------------------------------
class LiveMode:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.subscribers: list[queue.Queue] = []
        self.observations: collections.deque = collections.deque(maxlen=50)
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._broadcast({"event": "start", "ts": time.time()})
            return True

    def stop(self) -> bool:
        with self.lock:
            if not self.running:
                return False
            self.running = False
            self._stop_evt.set()
            self._broadcast({"event": "stop", "ts": time.time()})
            return True

    def is_running(self) -> bool:
        with self.lock:
            return self.running

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        with self.lock:
            self.subscribers.append(q)
            # Replay recent observations
            for obs in list(self.observations)[-10:]:
                with contextlib.suppress(queue.Full):
                    q.put_nowait({"event": "observation", **obs})
            q.put_nowait({"event": "state", "running": self.running})
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            with contextlib.suppress(ValueError):
                self.subscribers.remove(q)

    def _broadcast(self, ev: dict) -> None:
        with self.lock:
            for q in list(self.subscribers):
                with contextlib.suppress(queue.Full):
                    q.put_nowait(ev)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                tid = "live-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
                work = SESSION_DIR / tid
                work.mkdir(parents=True, exist_ok=True)
                capture_frame_for_vlm(work / "frame.jpg")
                cancel = threading.Event()

                def on_tok(_d: str) -> None:
                    return  # batch only

                reply, _ = stream_vlm(
                    "In one short sentence, describe what is happening or "
                    "visible in the scene RIGHT NOW. Be concrete. If nothing "
                    "has notably changed, say what is most salient.",
                    work / "frame.jpg", cancel, on_tok,
                    include_history=False, deadline_s=25.0,
                )
                if not reply:
                    # watchdog fired; treat as a skipped tick, do not stop
                    self._broadcast({
                        "event": "error",
                        "error": "live observation timed out",
                        "ts": time.time(),
                    })
                    continue
                obs = {
                    "ts": time.time(),
                    "turn_id": tid,
                    "text": reply,
                    "frame_url": f"/frame/{tid}.jpg",
                }
                self.observations.append(obs)
                self._broadcast({"event": "observation", **obs})
            except Exception as exc:
                self._broadcast({"event": "error", "error": str(exc), "ts": time.time()})
            # interval
            with SETTINGS_LOCK:
                interval = SETTINGS["live_interval_s"]
            for _ in range(max(1, int(interval * 4))):
                if self._stop_evt.wait(0.25):
                    return

    def list_observations(self) -> list:
        return list(self.observations)


LIVE = LiveMode()


# ----- shared state -----------------------------------------------------------
STATE_LOCK = threading.Lock()
CURRENT: dict = {"phase": "idle", "reply": ""}


def set_state(s: dict) -> None:
    with STATE_LOCK:
        CURRENT.clear()
        CURRENT.update(s)


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


# VLM health flag, refreshed by a background poller. Never call urllib
# synchronously from the metrics endpoint — when llama-server is busy
# generating, /health queues behind the long request and metrics hangs.
_VLM_UP = {"flag": False}


def _vlm_health_poller():
    while True:
        try:
            with urllib.request.urlopen(VLM_HEALTH, timeout=2) as r:
                _VLM_UP["flag"] = (r.status == 200)
        except Exception:
            _VLM_UP["flag"] = False
        time.sleep(3)


def gather_metrics() -> dict:
    # Metrics is pure-in-memory now — no synchronous network calls, no IO
    # beyond two small /proc reads. Safe under heavy concurrent load.
    m = read_meminfo()
    mem_total = int(re.findall(r"\d+", m.get("MemTotal", "0"))[0]) / 1024.0
    mem_avail = int(re.findall(r"\d+", m.get("MemAvailable", "0"))[0]) / 1024.0
    return {
        "ram_total_mb": round(mem_total),
        "ram_avail_mb": round(mem_avail),
        "soc_temp_c": read_soc_temp_c(),
        "vlm_up": _VLM_UP["flag"],
        "whisper_ok": WHISPER_BIN.exists() and WHISPER_MODEL.exists(),
        "piper_ok": PIPER_BIN.exists() and PIPER_VOICE.exists(),
        "cam_fps": CAMERA.fps(),
        "history_pairs": len(HISTORY) // 2,
        "pins_count": len(PINS),
        "live_running": LIVE.running,
        "live_observations": len(LIVE.observations),
    }


# ----- export -----------------------------------------------------------------
def export_markdown() -> str:
    lines = [
        "# Jarvis session", "",
        f"_exported {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]
    msgs = history_messages()
    if msgs:
        lines.append("## Conversation")
        lines.append("")
        for m in msgs:
            role = "User" if m["role"] == "user" else "Jarvis"
            content = m["content"] if isinstance(m["content"], str) else "(image)"
            lines.append(f"**{role}.** {content}")
            lines.append("")
    with PINS_LOCK:
        pins = list(PINS)
    if pins:
        lines.append("## Pinned")
        lines.append("")
        for p in pins:
            lines.append(f"- _{p.get('question','')}_  →  {p.get('reply','')}")
    return "\n".join(lines) + "\n"


# ----- HTML / UI --------------------------------------------------------------
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
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px; --r-xl: 20px; --r-pill: 999px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: 'Geist', system-ui, -apple-system, sans-serif;
  font-feature-settings: "ss01", "ss02", "cv11";
  background: var(--bg); color: var(--text);
  -webkit-font-smoothing: antialiased; line-height: 1.5; overflow: hidden;
}

.shell { display: grid; grid-template-rows: auto 1fr; height: 100vh; width: 100vw; }

/* topbar */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 11px 22px; border-bottom: 1px solid var(--border);
  gap: 14px;
}
.brand {
  font-weight: 600; font-size: 12px; letter-spacing: 0.22em; color: var(--muted);
  text-transform: uppercase; display: flex; align-items: center; gap: 9px;
}
.brand::before {
  content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 12px var(--accent-glow);
}
.topbar-actions { display: flex; gap: 4px; align-items: center; }
.iconbtn {
  background: transparent; border: 1px solid transparent;
  color: var(--muted);
  width: 32px; height: 32px; border-radius: var(--r-sm);
  display: inline-flex; align-items: center; justify-content: center;
  cursor: pointer; transition: 140ms ease-out;
  font-family: inherit;
}
.iconbtn:hover { background: var(--raised); color: var(--text); border-color: var(--border); }
.iconbtn.active { color: var(--accent); border-color: var(--accent-soft); }
.iconbtn svg { width: 16px; height: 16px; }

/* main grid: feed | (conv stack) */
.main {
  display: grid; gap: 14px; padding: 14px 22px; min-height: 0;
}

/* feed */
.feedwrap {
  position: relative; border-radius: var(--r-lg);
  overflow: hidden; background: #000; border: 1px solid var(--border);
}
.feedwrap img.live {
  width: 100%; height: 100%; object-fit: cover; display: block;
  cursor: crosshair;
}
.feedwrap svg.bbox-layer {
  position: absolute; inset: 0; width: 100%; height: 100%;
  pointer-events: none;
}
.feedwrap .tap-marker {
  position: absolute; width: 28px; height: 28px; border-radius: 50%;
  border: 2px solid var(--accent); box-shadow: 0 0 12px var(--accent-glow);
  transform: translate(-50%, -50%);
  animation: tap-pulse 1s ease-out forwards;
  pointer-events: none;
}
@keyframes tap-pulse {
  from { opacity: 1; width: 8px; height: 8px; }
  to   { opacity: 0; width: 80px; height: 80px; }
}

/* feed HUD */
.hud-pill {
  position: absolute; padding: 6px 10px;
  background: rgba(10,10,11,0.65);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-strong); border-radius: var(--r-pill);
  font: 500 11px/1 'Geist Mono', monospace; color: var(--text);
  display: flex; align-items: center; gap: 8px;
}
.hud-tl { top: 12px; left: 12px; }
.hud-tr { top: 12px; right: 12px; }
.hud-bl { bottom: 12px; left: 12px; color: var(--muted); }
.hud-br { bottom: 12px; right: 12px; }
.livepill .dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
  animation: pulse 1.6s ease-in-out infinite;
}
.livepill.idle .dot { background: var(--muted); animation: none; }
.livepill.live-on .dot { background: var(--ok); }
@keyframes pulse {
  0%,100% { opacity: 0.45; transform: scale(0.85); }
  50% { opacity: 1; transform: scale(1.0); box-shadow: 0 0 10px var(--accent-glow); }
}
.hud-sep { color: var(--dim); }
.hud-val.ok { color: var(--ok); }
.hud-val.warn { color: var(--warn); }
.hud-val.bad { color: var(--bad); }
.live-toggle {
  position: absolute; bottom: 12px; right: 12px;
  padding: 8px 14px; font: 600 11.5px/1 'Geist', sans-serif;
  letter-spacing: 0.08em; text-transform: uppercase;
  background: rgba(10,10,11,0.65); backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-strong); border-radius: var(--r-pill);
  color: var(--text); cursor: pointer;
  display: inline-flex; align-items: center; gap: 7px;
  transition: 140ms;
}
.live-toggle:hover { border-color: var(--accent); color: var(--accent); }
.live-toggle.on { background: var(--accent); border-color: var(--accent); color: white; }
.live-toggle .dotpip { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
.live-toggle.on .dotpip { background: white; animation: pulse 1.2s ease-in-out infinite; }

/* right column */
.rightcol {
  display: grid; grid-template-rows: auto 1fr auto;
  min-height: 0; gap: 10px;
}
.rightcol-tabs {
  display: flex; gap: 4px; padding: 2px;
  border-bottom: 1px solid var(--border);
}
.tab {
  flex: 1; padding: 7px 10px; font-size: 12px; font-weight: 500;
  background: transparent; color: var(--muted);
  border: none; border-radius: var(--r-sm); cursor: pointer;
  font-family: inherit; transition: 120ms;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); }
.tab-badge {
  display: inline-block; margin-left: 4px;
  background: var(--accent-soft); color: var(--accent);
  padding: 1px 6px; border-radius: var(--r-pill);
  font-size: 10px; font-family: 'Geist Mono', monospace;
}

.pane {
  overflow-y: auto; scroll-behavior: smooth;
  padding: 4px 2px 8px; min-height: 0;
}
.pane.hidden { display: none; }
.pane::-webkit-scrollbar { width: 8px; }
.pane::-webkit-scrollbar-thumb { background: var(--raised); border-radius: 4px; }

.empty {
  text-align: center; padding: 32px 16px;
  color: var(--dim); font-size: 13px;
}
.empty .kbd { color: var(--muted); }

/* conversation messages */
.msg {
  display: flex; flex-direction: column; gap: 4px;
  margin-bottom: 18px; animation: msgIn 220ms ease-out;
}
@keyframes msgIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.msg-meta {
  display: flex; align-items: center; gap: 8px;
  font: 600 10px/1 'Geist Mono', monospace;
  color: var(--dim); text-transform: uppercase; letter-spacing: 0.08em;
}
.msg-meta .who { color: var(--muted); }
.msg-meta .who.j { color: var(--accent); }
.msg-meta .badge {
  padding: 1px 6px; border: 1px solid var(--border);
  border-radius: var(--r-sm); font-size: 9px;
}
.msg-meta .pinmark { color: var(--accent); }

.bubble {
  padding: 10px 14px; border-radius: var(--r-md);
  font-size: 14.5px; line-height: 1.55; color: var(--text);
  max-width: 100%; word-wrap: break-word;
}
.bubble.user {
  background: var(--user-bubble); border: 1px solid var(--border);
  align-self: flex-end; max-width: 88%;
}
.bubble.jarvis {
  background: transparent; border: none;
  padding-left: 0; padding-right: 0;
}
.bubble.jarvis .cursor {
  display: inline-block; width: 7px; height: 16px;
  background: var(--accent); margin-left: 2px;
  vertical-align: -3px; border-radius: 1px;
  animation: blink 1s steps(2) infinite;
}
@keyframes blink { 50% { opacity: 0; } }
.bubble p { margin: 0 0 8px; }
.bubble p:last-child { margin-bottom: 0; }
.bubble ul, .bubble ol { padding-left: 22px; margin: 6px 0; }
.bubble code {
  font-family: 'Geist Mono', monospace; background: var(--raised);
  padding: 1px 6px; border-radius: 4px; font-size: 0.85em;
}
.bubble pre {
  background: var(--surface); border: 1px solid var(--border);
  padding: 10px 12px; border-radius: var(--r-sm); overflow-x: auto;
  font-family: 'Geist Mono', monospace; font-size: 12px; margin: 6px 0;
}
.bubble strong { font-weight: 600; }
.bubble a { color: var(--accent); text-decoration: underline; }

.thumb {
  margin-top: 6px; width: 96px; height: 72px;
  border-radius: 6px; object-fit: cover;
  border: 1px solid var(--border); cursor: zoom-in;
  opacity: 0.85; transition: 140ms;
}
.thumb:hover { opacity: 1; border-color: var(--border-strong); }

.actions {
  display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap;
  opacity: 0; transition: opacity 120ms 60ms;
}
.msg:hover .actions { opacity: 1; }
.actions button {
  background: transparent; border: 1px solid var(--border);
  color: var(--muted); padding: 4px 8px; border-radius: var(--r-sm);
  cursor: pointer; font-size: 11px; font-family: inherit;
  display: inline-flex; align-items: center; gap: 5px; transition: 120ms;
}
.actions button:hover { color: var(--text); border-color: var(--border-strong); background: var(--raised); }
.actions button.pinned { color: var(--accent); border-color: var(--accent-soft); }
.actions svg { width: 12px; height: 12px; }
.audio-row { margin-top: 6px; max-width: 360px; }
audio { width: 100%; height: 32px; }

.lat-bar {
  margin-top: 4px; height: 6px; border-radius: 3px; background: var(--raised);
  display: flex; overflow: hidden;
}
.lat-bar .seg {
  height: 100%; opacity: 0.85; min-width: 1px;
}
.lat-bar .seg.record { background: #5da6ff; }
.lat-bar .seg.transcribe { background: #71ddd7; }
.lat-bar .seg.capture { background: #b3e16f; }
.lat-bar .seg.vlm { background: var(--accent); }
.lat-bar .seg.tts { background: #c0a3f3; }
.timings {
  font: 500 10.5px/1.4 'Geist Mono', monospace;
  color: var(--dim); margin-top: 4px;
  display: flex; gap: 12px; flex-wrap: wrap;
}
.timings .t-tok { color: var(--muted); }

/* live observations pane */
.live-obs {
  display: flex; gap: 8px; padding: 8px 10px;
  border-left: 2px solid var(--ok); background: var(--surface);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  margin-bottom: 8px; animation: msgIn 200ms ease-out;
}
.live-obs img {
  width: 60px; height: 45px; object-fit: cover;
  border-radius: 4px; border: 1px solid var(--border);
  flex-shrink: 0;
}
.live-obs .body { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.live-obs .ts { font: 500 10px/1 'Geist Mono', monospace; color: var(--dim); }
.live-obs .text { font-size: 12.5px; color: var(--text); }

/* pinned card */
.pin-card {
  background: var(--surface); border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--r-sm); padding: 10px 12px;
  margin-bottom: 8px;
}
.pin-card .q { font-size: 11px; color: var(--muted); font-style: italic; margin-bottom: 4px; }
.pin-card .r { font-size: 13px; color: var(--text); }
.pin-card .unpin {
  font-size: 11px; color: var(--dim); margin-top: 6px;
  background: transparent; border: none; cursor: pointer; font-family: inherit;
}
.pin-card .unpin:hover { color: var(--accent); }

/* composer */
.composer {
  padding: 12px 14px; background: var(--surface);
  border: 1px solid var(--border-strong); border-radius: var(--r-xl);
  display: flex; flex-direction: column; gap: 8px; position: relative;
  transition: border-color 140ms;
}
.composer:focus-within { border-color: var(--accent-soft); }
.composer.recording { border-color: var(--accent); box-shadow: 0 0 18px var(--accent-glow); }

.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip {
  background: transparent; border: 1px solid var(--border);
  color: var(--muted); padding: 4px 10px; border-radius: var(--r-pill);
  font: 500 11.5px/1 'Geist', sans-serif; font-family: inherit; cursor: pointer;
  display: inline-flex; align-items: center; gap: 5px; transition: 140ms;
}
.chip:hover { color: var(--text); border-color: var(--accent-soft); background: var(--accent-soft); }
.chip svg { width: 11px; height: 11px; }
.chip.active { color: var(--accent); border-color: var(--accent); }

.input-row { display: flex; gap: 8px; align-items: flex-end; }
.orb {
  flex: 0 0 auto; width: 36px; height: 36px; border-radius: 50%;
  border: 1px solid var(--border-strong);
  background: radial-gradient(circle at 30% 30%, var(--raised) 0%, var(--surface) 70%);
  cursor: pointer; position: relative; transition: 200ms; flex-shrink: 0;
  overflow: hidden;
}
.orb::before {
  content: ''; position: absolute; inset: 6px; border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, var(--accent) 0%, transparent 70%);
  opacity: 0.7; animation: breathe 3s ease-in-out infinite;
}
@keyframes breathe {
  0%,100% { transform: scale(0.7); opacity: 0.45; }
  50%     { transform: scale(1.0); opacity: 0.85; }
}
.orb.recording { border-color: var(--accent); box-shadow: 0 0 18px var(--accent-glow); }
.orb canvas {
  position: absolute; inset: 2px; width: calc(100% - 4px); height: calc(100% - 4px);
  border-radius: 50%; display: none;
}
.orb.recording canvas { display: block; }
.orb.recording::before { animation: none; opacity: 0; }
.orb.busy::before { animation: breathe 1.4s ease-in-out infinite; }

textarea#text {
  flex: 1; resize: none; background: transparent; color: var(--text);
  border: none; outline: none; font-family: inherit;
  font-size: 15px; line-height: 1.5; padding: 7px 0;
  min-height: 24px; max-height: 160px;
}
textarea#text::placeholder { color: var(--dim); }

.sendbtn {
  flex: 0 0 auto; width: 36px; height: 36px; border-radius: 50%;
  background: var(--accent); border: none; color: white; cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center;
  transition: 160ms;
}
.sendbtn:hover { transform: scale(1.04); box-shadow: 0 0 14px var(--accent-glow); }
.sendbtn:disabled { background: var(--raised); color: var(--dim); cursor: not-allowed; transform: none; box-shadow: none; }
.sendbtn svg { width: 16px; height: 16px; }

.composer-foot {
  display: flex; align-items: center; justify-content: space-between;
  font: 500 11px/1.2 'Geist Mono', monospace; color: var(--dim);
}
.composer-foot .stop {
  background: transparent; border: 1px solid var(--accent);
  color: var(--accent); padding: 3px 10px; border-radius: var(--r-pill);
  cursor: pointer; font-size: 10.5px; font-family: inherit; display: none;
}
.composer-foot.busy .stop { display: inline-flex; }
.kbd {
  font: 10px/1 'Geist Mono', monospace; color: var(--dim);
  padding: 2px 5px; border: 1px solid var(--border); border-radius: 4px;
}

/* palette */
.palette {
  position: absolute; bottom: calc(100% + 8px);
  left: 14px; right: 14px;
  background: var(--surface); border: 1px solid var(--border-strong);
  border-radius: var(--r-md); box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  padding: 6px; display: none; z-index: 10;
}
.palette.open { display: block; animation: msgIn 140ms ease-out; }
.palette-item {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 10px; border-radius: var(--r-sm);
  cursor: pointer; font-size: 13px; color: var(--text);
}
.palette-item:hover, .palette-item.sel { background: var(--accent-soft); color: var(--text); }
.palette-item .cmd { color: var(--accent); font-family: 'Geist Mono', monospace; font-size: 11.5px; width: 80px; }
.palette-item .desc { color: var(--muted); font-size: 12px; }

/* drawer */
.drawer-mask {
  position: fixed; inset: 0; background: rgba(0,0,0,0.4);
  opacity: 0; pointer-events: none; transition: opacity 200ms;
  z-index: 90;
}
.drawer-mask.open { opacity: 1; pointer-events: auto; }
.drawer {
  position: fixed; top: 0; right: 0; bottom: 0;
  width: min(440px, 92vw); background: var(--surface);
  border-left: 1px solid var(--border-strong); z-index: 100;
  transform: translateX(100%); transition: transform 240ms cubic-bezier(0.32,0.72,0,1);
  display: flex; flex-direction: column;
}
.drawer.open { transform: translateX(0); }
.drawer-head {
  padding: 18px 20px; border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
}
.drawer-head h2 { font-size: 13px; font-weight: 600; letter-spacing: 0.16em; }
.drawer-body {
  flex: 1; overflow-y: auto; padding: 20px;
  display: flex; flex-direction: column; gap: 18px;
}
.field { display: flex; flex-direction: column; gap: 6px; }
.field label {
  font: 600 10.5px/1 'Geist', sans-serif;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em;
}
.field input, .field textarea, .field select {
  background: var(--bg); border: 1px solid var(--border);
  color: var(--text); border-radius: var(--r-sm);
  padding: 8px 10px; font-family: inherit; font-size: 13px;
  outline: none; transition: 140ms;
}
.field input:focus, .field textarea:focus { border-color: var(--accent); }
.field textarea { resize: vertical; min-height: 110px; font-size: 12.5px; line-height: 1.5; }
.field .hint { font: 500 11px/1.4 'Geist Mono', monospace; color: var(--dim); }
.field .row { display: flex; gap: 8px; align-items: center; }
.field .row input[type=range] { flex: 1; accent-color: var(--accent); }
.field .row .val {
  font: 500 12px/1 'Geist Mono', monospace; color: var(--muted);
  width: 50px; text-align: right;
}
.preset-row { display: flex; gap: 5px; flex-wrap: wrap; }
.preset-chip {
  background: var(--bg); border: 1px solid var(--border);
  color: var(--muted); padding: 5px 11px; border-radius: var(--r-pill);
  font: 500 11px/1 'Geist', sans-serif; cursor: pointer; transition: 120ms;
}
.preset-chip:hover { color: var(--text); }
.preset-chip.active { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }
.drawer-foot {
  padding: 14px 20px; border-top: 1px solid var(--border);
  display: flex; gap: 8px; justify-content: flex-end;
}
.btn {
  background: transparent; border: 1px solid var(--border-strong);
  color: var(--text); padding: 7px 14px; border-radius: var(--r-sm);
  cursor: pointer; font-family: inherit; font-size: 12px; transition: 140ms;
}
.btn:hover { background: var(--raised); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: white; }
.btn.primary:hover { filter: brightness(1.08); }

/* toast */
.toast-wrap {
  position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
  z-index: 200; display: flex; flex-direction: column; gap: 8px;
  pointer-events: none;
}
.toast {
  background: var(--raised); border: 1px solid var(--border-strong);
  border-radius: var(--r-pill); padding: 8px 16px; font-size: 12.5px;
  color: var(--text); display: flex; align-items: center; gap: 8px;
  animation: toastIn 200ms ease-out;
}
.toast.error { border-color: var(--bad); color: #fcc; }
.toast.ok { border-color: var(--ok); }
.toast svg { width: 13px; height: 13px; }
@keyframes toastIn {
  from { opacity: 0; transform: translateY(-8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* lightbox */
.lightbox {
  position: fixed; inset: 0; background: rgba(0,0,0,0.92);
  display: none; align-items: center; justify-content: center;
  z-index: 300; cursor: zoom-out;
}
.lightbox.open { display: flex; }
.lightbox img { max-width: 92vw; max-height: 92vh; border-radius: var(--r-md); }

/* shortcut modal */
.modal {
  position: fixed; inset: 0; background: rgba(0,0,0,0.6);
  display: none; align-items: center; justify-content: center;
  z-index: 250; padding: 20px;
}
.modal.open { display: flex; animation: msgIn 180ms ease-out; }
.modal-content {
  background: var(--surface); border: 1px solid var(--border-strong);
  border-radius: var(--r-lg); padding: 24px;
  max-width: 580px; width: 100%; max-height: 80vh; overflow-y: auto;
}
.modal-content h2 { font-size: 14px; font-weight: 600; letter-spacing: 0.14em; margin-bottom: 14px; color: var(--muted); text-transform: uppercase; }
.shortcut-grid {
  display: grid; grid-template-columns: max-content 1fr; gap: 8px 18px;
  font-size: 13px;
}
.shortcut-grid .keys { display: flex; gap: 4px; align-items: center; }
.shortcut-grid .desc { color: var(--muted); font-size: 12.5px; }

/* responsive: wide (>= 980px) */
@media (min-width: 980px) {
  .main {
    grid-template-columns: minmax(0, 1.45fr) minmax(380px, 1fr);
    grid-template-rows: 1fr;
    gap: 18px; padding: 14px 22px;
  }
  .rightcol { grid-row: 1; }
}
@media (max-width: 640px) {
  .topbar { padding: 11px 14px; }
  .main { padding: 10px 14px; gap: 10px; }
  .bubble.user { max-width: 95%; }
}
</style>

<body>
<div class=shell>

  <div class=topbar>
    <div class=brand>Jarvis</div>
    <div class=topbar-actions>
      <button class=iconbtn id=shortcutsbtn title='Shortcuts (?)' aria-label=shortcuts>
        <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/><path d='M12 17h.01'/></svg>
      </button>
      <button class=iconbtn id=exportbtn title='Export conversation as Markdown' aria-label=export>
        <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/><polyline points='7 10 12 15 17 10'/><line x1='12' y1='15' x2='12' y2='3'/></svg>
      </button>
      <button class=iconbtn id=clearbtn title='Clear conversation' aria-label=clear>
        <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 6h18'/><path d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6'/><path d='M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/></svg>
      </button>
      <button class=iconbtn id=settingsbtn title='Settings' aria-label=settings>
        <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='3'/><path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z'/></svg>
      </button>
    </div>
  </div>

  <div class=main>
    <!-- camera column -->
    <div class=feedwrap id=feedwrap>
      <img class=live id=liveimg src='/stream.mjpeg' alt='live camera'>
      <svg class='bbox-layer' id=bboxlayer viewBox='0 0 100 100' preserveAspectRatio=none></svg>
      <div class='hud-pill hud-tl livepill idle' id=livepill><span class=dot></span> idle</div>
      <div class='hud-pill hud-tr' id=hudtr><span>0.0 fps</span><span class=hud-sep>·</span><span class='hud-val'>—</span></div>
      <div class='hud-pill hud-bl' id=hudbl>qwen2.5-vl-3b · 640&times;480</div>
      <button class=live-toggle id=livetoggle title='Continuous live narration'>
        <span class=dotpip></span><span>Live mode</span>
      </button>
    </div>

    <!-- right column -->
    <div class=rightcol>
      <div class=rightcol-tabs role=tablist>
        <button class='tab active' id=tab-conv data-pane=conv role=tab>Conversation</button>
        <button class=tab id=tab-live data-pane=live role=tab>Live <span class=tab-badge id=live-count>0</span></button>
        <button class=tab id=tab-pins data-pane=pins role=tab>Pinned <span class=tab-badge id=pin-count>0</span></button>
      </div>

      <div class=pane id=pane-conv>
        <div class=empty id=emptystate>
          Ask Jarvis anything · tap orb to talk · click the feed to ask about a spot · <span class=kbd>/</span> for commands · <span class=kbd>?</span> for shortcuts
        </div>
      </div>
      <div class='pane hidden' id=pane-live>
        <div class=empty id=live-empty>
          Live mode is off. Press <span class=kbd>L</span> or tap the Live button on the feed to start. Jarvis will narrate the scene every few seconds.
        </div>
      </div>
      <div class='pane hidden' id=pane-pins>
        <div class=empty id=pins-empty>
          No pinned replies yet. Hover any Jarvis reply and click the bookmark to save it here.
        </div>
      </div>

      <div class=composer id=composer>
        <div class=chips id=chips>
          <button class=chip data-cmd=look><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z'/><circle cx='12' cy='12' r='3'/></svg>What do you see</button>
          <button class=chip data-cmd=read><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20'/></svg>Read any text</button>
          <button class=chip data-cmd=count><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 4h4v4H4zM10 4h4v4h-4zM16 4h4v4h-4zM4 10h4v4H4zM10 10h4v4h-4zM16 10h4v4h-4zM4 16h4v4H4zM10 16h4v4h-4zM16 16h4v4h-4z'/></svg>Count objects</button>
          <button class=chip data-cmd=identify><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/><path d='M12 17h.01'/></svg>What am I looking at</button>
        </div>
        <div class=input-row>
          <button class=orb id=orb title='Tap to record (Space)'>
            <canvas id=orbwave width=60 height=60></canvas>
          </button>
          <textarea id=text placeholder='Ask Jarvis · type / for commands' rows=1 autofocus></textarea>
          <button class=sendbtn id=sendbtn title='Send (Enter)'>
            <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><path d='M22 2 11 13'/><path d='m22 2-7 20-4-9-9-4 20-7z'/></svg>
          </button>
        </div>
        <div class=composer-foot id=composerfoot>
          <span><kbd class=kbd>/</kbd> <kbd class=kbd>Enter</kbd> <kbd class=kbd>Space</kbd> <kbd class=kbd>L</kbd> live <kbd class=kbd>?</kbd> help</span>
          <button class=stop id=stopbtn>Stop</button>
        </div>
        <div class=palette id=palette></div>
      </div>
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
      <label>PERSONA PRESETS</label>
      <div class=preset-row id=presetrow></div>
      <span class=hint>One-click persona swap · custom edits are auto-tagged</span>
    </div>
    <div class=field>
      <label>SYSTEM PROMPT</label>
      <textarea id=sysprompt></textarea>
    </div>
    <div class=field>
      <label>RESPONSE LENGTH (TOKENS)</label>
      <div class=row><input type=range id=maxtokens min=60 max=500 step=20><span class=val id=maxtokensv>240</span></div>
    </div>
    <div class=field>
      <label>TEMPERATURE</label>
      <div class=row><input type=range id=temperature min=0 max=10 step=1><span class=val id=temperaturev>0.2</span></div>
    </div>
    <div class=field>
      <label>VOICE RECORDING (SECONDS)</label>
      <div class=row><input type=range id=recsec min=3 max=12 step=1><span class=val id=recsecv>6s</span></div>
    </div>
    <div class=field>
      <label>LIVE MODE INTERVAL (SECONDS)</label>
      <div class=row><input type=range id=liveint min=4 max=30 step=1><span class=val id=liveintv>8s</span></div>
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

<div class=modal id=shortcuts-modal>
  <div class=modal-content>
    <h2>KEYBOARD &amp; COMMANDS</h2>
    <div class=shortcut-grid>
      <div class=keys><span class=kbd>Enter</span></div>          <div class=desc>Send typed question</div>
      <div class=keys><span class=kbd>Space</span></div>          <div class=desc>SNAP — describe scene now</div>
      <div class=keys><span class=kbd>/</span></div>              <div class=desc>Open command palette</div>
      <div class=keys><span class=kbd>L</span></div>              <div class=desc>Toggle Live Mode</div>
      <div class=keys><span class=kbd>P</span></div>              <div class=desc>Pin last Jarvis reply</div>
      <div class=keys><span class=kbd>R</span></div>              <div class=desc>Regenerate last reply</div>
      <div class=keys><span class=kbd>Esc</span></div>            <div class=desc>Cancel turn / close palette / clear input</div>
      <div class=keys><span class=kbd>?</span></div>              <div class=desc>This help</div>
      <div class=keys><span class=kbd>1</span> <span class=kbd>2</span> <span class=kbd>3</span></div>
      <div class=desc>Switch panes: Conversation / Live / Pinned</div>
    </div>
    <h2 style='margin-top:24px'>SLASH COMMANDS</h2>
    <div class=shortcut-grid>
      <div class=keys><code>/look</code></div>     <div class=desc>Describe the scene</div>
      <div class=keys><code>/read</code></div>     <div class=desc>Read any text you can see</div>
      <div class=keys><code>/count</code></div>    <div class=desc>Count objects of interest</div>
      <div class=keys><code>/identify</code></div> <div class=desc>Identify the main subject</div>
      <div class=keys><code>/find X</code></div>   <div class=desc>Locate a specific object</div>
      <div class=keys><code>/voice</code></div>    <div class=desc>Voice mode (record)</div>
      <div class=keys><code>/live</code></div>     <div class=desc>Toggle Live Mode</div>
      <div class=keys><code>/clear</code></div>    <div class=desc>Clear conversation</div>
      <div class=keys><code>/export</code></div>   <div class=desc>Download conversation as Markdown</div>
      <div class=keys><code>/settings</code></div> <div class=desc>Open settings</div>
    </div>
    <h2 style='margin-top:24px'>FEED INTERACTIONS</h2>
    <div class=shortcut-grid>
      <div class=keys><span class=kbd>Click feed</span></div>     <div class=desc>Ask Jarvis about a specific point</div>
      <div class=keys><span class=kbd>Live</span> button</div>    <div class=desc>Auto-narrate every N seconds</div>
    </div>
  </div>
</div>

<div class=toast-wrap id=toastwrap></div>

<script>
'use strict';
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
marked.setOptions({ breaks: true, gfm: true });

let currentTurn = null;
let isStreaming = false;
let lastJarvisMsg = null;
let liveES = null;
let liveOn = false;

const liveimg     = $('#liveimg');
const feedwrap    = $('#feedwrap');
const bboxlayer   = $('#bboxlayer');
const composer    = $('#composer');
const composerfoot= $('#composerfoot');
const orb         = $('#orb');
const orbwave     = $('#orbwave');
const sendbtn     = $('#sendbtn');
const livepill    = $('#livepill');
const livetoggle  = $('#livetoggle');
const emptystate  = $('#emptystate');
const paneConv    = $('#pane-conv');
const paneLive    = $('#pane-live');
const panePins    = $('#pane-pins');
const liveCount   = $('#live-count');
const pinCount    = $('#pin-count');

function escapeHtml(s) {
  return String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'})[c]);
}
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toTimeString().slice(0, 8);
}
function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = msg;
  $('#toastwrap').appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

/* ---- tabs ---- */
$$('.tab').forEach(t => {
  t.onclick = () => switchPane(t.dataset.pane);
});
function switchPane(name) {
  $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.pane === name));
  paneConv.classList.toggle('hidden', name !== 'conv');
  paneLive.classList.toggle('hidden', name !== 'live');
  panePins.classList.toggle('hidden', name !== 'pins');
}

/* ---- metrics tick ---- */
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
    $('#hudbl').textContent = 'qwen2.5-vl-3b · history: ' + m.history_pairs + ' · pins: ' + m.pins_count;
    pinCount.textContent = m.pins_count;
    liveCount.textContent = m.live_observations;
    const tele = $('#teleHint');
    if (tele) tele.innerHTML =
      'RAM ' + m.ram_avail_mb + ' / ' + m.ram_total_mb + ' MB<br>' +
      'SoC ' + m.soc_temp_c.toFixed(1) + '°C<br>' +
      'cam ' + m.cam_fps.toFixed(1) + ' fps<br>' +
      'VLM ' + (m.vlm_up ? 'up' : 'down') + ' · whisper ' + (m.whisper_ok?'ok':'missing') + ' · piper ' + (m.piper_ok?'ok':'missing') + '<br>' +
      'live ' + (m.live_running ? 'ON' : 'off') + ' · obs ' + m.live_observations;
    // sync live toggle if changed elsewhere
    if (m.live_running !== liveOn) {
      liveOn = m.live_running;
      livetoggle.classList.toggle('on', liveOn);
      livetoggle.querySelector('span:nth-child(2)').textContent = liveOn ? 'Live ON' : 'Live mode';
      if (liveOn && !liveES) attachLiveStream();
      if (!liveOn && liveES) detachLiveStream();
    }
  } catch(e) {}
}
setInterval(tickMetrics, 2000); tickMetrics();

function setLive(state, label) {
  livepill.className = 'hud-pill hud-tl livepill ' + (state || 'idle');
  livepill.innerHTML = '<span class=dot></span> ' + (label || 'idle');
}

/* ---- markdown rendering helper ---- */
function renderMarkdown(text) {
  const openFences = (text.match(/```/g) || []).length;
  if (openFences % 2 === 1) text = text + '\n```';
  try { return marked.parse(text); } catch { return escapeHtml(text); }
}

/* ---- conversation messages ---- */
function makeUserMsg(text, kind) {
  emptystate.style.display = 'none';
  const msg = document.createElement('div');
  msg.className = 'msg user-msg';
  const badge = kind === 'talk' ? 'spoken' : kind === 'snap' ? 'snap' : kind === 'point' ? 'point' : 'typed';
  msg.innerHTML =
    '<div class=msg-meta><span class=who>You</span><span class=badge>' + badge + '</span></div>' +
    '<div class="bubble user">' + escapeHtml(text) + '</div>';
  paneConv.appendChild(msg);
  scrollPane(paneConv);
  return msg;
}

function makeJarvisMsg(kind) {
  const msg = document.createElement('div');
  msg.className = 'msg jarvis-msg';
  msg.dataset.kind = kind || '';
  msg.innerHTML =
    '<div class=msg-meta><span class="who j">Jarvis</span><span class=badge id=phase>thinking</span></div>' +
    '<div class="bubble jarvis streaming" data-raw=""><span class=cursor></span></div>' +
    '<div class=actions>' +
      '<button data-act=copy><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy</button>' +
      '<button data-act=regen><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>Regenerate</button>' +
      '<button data-act=pin><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>Pin</button>' +
      '<button data-act=replay style="display:none"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>Play</button>' +
    '</div>' +
    '<div class=audio-row id=audiorow></div>' +
    '<div class=lat-bar id=latbar></div>' +
    '<div class=timings id=timings></div>';
  paneConv.appendChild(msg);
  scrollPane(paneConv);
  lastJarvisMsg = msg;
  return msg;
}

let scrollLocked = true;
function scrollPane(p) {
  if (!scrollLocked) return;
  requestAnimationFrame(() => { p.scrollTop = p.scrollHeight; });
}
paneConv.addEventListener('scroll', () => {
  const slack = paneConv.scrollHeight - paneConv.scrollTop - paneConv.clientHeight;
  scrollLocked = slack < 80;
});

/* ---- per-message actions ---- */
paneConv.addEventListener('click', async (e) => {
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
    } else if (act === 'pin') {
      const raw = msg.querySelector('.bubble').dataset.raw || '';
      const prevUser = msg.previousElementSibling;
      const question = prevUser && prevUser.classList.contains('user-msg') ? prevUser.querySelector('.bubble').textContent : '';
      const frameUrl = msg.querySelector('img.thumb')?.src;
      const r = await fetch('/pins', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question, reply: raw, frame_url: frameUrl})});
      if (r.ok) { btn.classList.add('pinned'); toast('Pinned', 'ok'); loadPins(); }
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

/* ---- turn execution ---- */
async function doTurn(payload) {
  if (isStreaming) return;
  isStreaming = true;
  setComposerBusy(true);

  let userText = '';
  if (payload.kind === 'text') userText = payload.text;
  else if (payload.kind === 'snap') userText = 'Describe what you see';
  else if (payload.kind === 'talk') userText = '(listening...)';
  else if (payload.kind === 'point') userText = 'What is at this spot?';
  else if (payload.kind === 'regenerate') userText = '(regenerating)';
  if (payload.kind !== 'regenerate') makeUserMsg(userText, payload.kind);

  const jmsg = makeJarvisMsg(payload.kind);
  const bubble = jmsg.querySelector('.bubble');
  const phaseBadge = jmsg.querySelector('#phase');
  const timingsEl = jmsg.querySelector('#timings');
  const audioRow = jmsg.querySelector('#audiorow');
  const replayBtn = jmsg.querySelector('button[data-act=replay]');
  const latBar = jmsg.querySelector('#latbar');
  let raw = '';
  switchPane('conv');

  let r;
  try {
    r = await fetch('/turn', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  } catch (e) {
    toast('Network error: ' + e.message, 'error');
    isStreaming = false; setComposerBusy(false);
    bubble.innerHTML = '<em>(failed to reach server)</em>'; return;
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
      scrollPane(paneConv);
    });
  }

  es.onmessage = (ev) => {
    if (!ev.data) return;
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    const ph = m.phase;
    if (ph === 'recording') {
      setLive('recording', 'listening · ' + (m.seconds || 6) + 's');
      orb.classList.add('recording'); composer.classList.add('recording');
      phaseBadge.textContent = 'listening'; attachAudioMeter();
    } else if (ph === 'transcribing') {
      orb.classList.remove('recording'); composer.classList.remove('recording');
      detachAudioMeter();
      setLive('idle', 'transcribing'); phaseBadge.textContent = 'transcribing';
    } else if (ph === 'capturing') {
      setLive('recording', 'looking'); phaseBadge.textContent = 'looking';
      if (m.question && payload.kind === 'regenerate') {
        const msg = makeUserMsg(m.question, 'typed');
        paneConv.insertBefore(msg, jmsg);
      } else if (m.transcription && payload.kind === 'talk') {
        const prevUser = jmsg.previousElementSibling;
        if (prevUser && prevUser.classList.contains('user-msg')) {
          prevUser.querySelector('.bubble').textContent = m.transcription || '(silent)';
        }
      }
    } else if (ph === 'thinking') {
      setLive('recording', 'reasoning'); phaseBadge.textContent = 'reasoning';
    } else if (ph === 'token') {
      raw += m.delta; scheduleRender();
    } else if (ph === 'speaking') {
      setLive('idle', 'speaking'); phaseBadge.textContent = 'speaking';
    } else if (ph === 'cancelled') {
      phaseBadge.textContent = 'stopped'; bubble.classList.remove('streaming');
    } else if (ph === 'error') {
      bubble.innerHTML = '<em>error: ' + escapeHtml(m.error) + '</em>';
      phaseBadge.textContent = 'error'; toast(m.error, 'error');
    } else if (ph === 'done') {
      const res = m.result;
      raw = res.reply || raw;
      bubble.dataset.raw = raw;
      bubble.innerHTML = renderMarkdown(raw);
      bubble.classList.remove('streaming');
      phaseBadge.textContent = res.cancelled ? 'stopped' : res.kind;
      if (res.frame_url) {
        const thumb = document.createElement('img');
        thumb.className = 'thumb'; thumb.src = res.frame_url; thumb.alt = 'frame';
        bubble.appendChild(thumb);
      }
      if (res.audio_url && !res.cancelled) {
        const a = document.createElement('audio');
        a.controls = true; a.src = res.audio_url;
        audioRow.appendChild(a);
        a.play().catch(()=>{});
        replayBtn.style.display = 'inline-flex';
      }
      if (res.timings) renderTimings(latBar, timingsEl, res.timings, res.usage);
      es.close();
      setLive('idle', 'idle');
      orb.classList.remove('recording'); composer.classList.remove('recording');
      detachAudioMeter();
      isStreaming = false; setComposerBusy(false);
      currentTurn = null;
    }
  };
  es.onerror = () => {
    if (isStreaming) toast('Stream error', 'error');
    es.close();
    isStreaming = false; setComposerBusy(false);
    setLive('idle', 'idle');
    orb.classList.remove('recording'); composer.classList.remove('recording');
    detachAudioMeter();
  };
}

function renderTimings(barEl, txtEl, timings, usage) {
  const order = [
    ['record', 'record'], ['transcribe', 'transcribe'],
    ['capture', 'capture'], ['vlm', 'vlm'], ['tts', 'tts']
  ];
  const segs = [];
  let total = 0;
  for (const [k, _cls] of order) {
    const v = timings[k + '_s'];
    if (v != null) { total += v; segs.push([k, v]); }
  }
  if (total > 0) {
    barEl.innerHTML = segs.map(([k, v]) =>
      '<div class="seg ' + k + '" style="flex:' + (v / total) + '" title="' + k + ' ' + v + 's"></div>'
    ).join('');
  }
  let line = segs.map(([k, v]) => k + ' ' + v + 's').join('  ·  ');
  if (usage && usage.prompt_tokens) {
    line += '  ·  <span class=t-tok>' + usage.prompt_tokens + ' in / ' + (usage.completion_tokens || 0) + ' out</span>';
  }
  txtEl.innerHTML = line;
}

function setComposerBusy(b) {
  composerfoot.classList.toggle('busy', b);
  sendbtn.disabled = b;
  orb.classList.toggle('busy', b);
}

/* ---- composer ---- */
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
    if (e.key === 'ArrowUp')   { e.preventDefault(); movePal(-1); return; }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); pickPal(); return; }
    if (e.key === 'Escape') { e.preventDefault(); closePalette(); return; }
  }
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  else if (e.key === 'Escape') { text.value = ''; text.style.height = 'auto'; }
});
function submit() {
  const v = text.value.trim(); if (!v) return;
  if (v.startsWith('/')) {
    const parts = v.slice(1).split(/\s+/);
    runCmd(parts[0], parts.slice(1).join(' ')); return;
  }
  doTurn({kind: 'text', text: v});
  text.value = ''; text.style.height = 'auto';
}
sendbtn.onclick = submit;
orb.onclick = () => { if (!isStreaming) doTurn({kind: 'talk'}); };
$$('.chip').forEach(b => { b.onclick = () => { if (!isStreaming) runCmd(b.dataset.cmd); }; });

/* ---- click-on-feed (point queries) ---- */
liveimg.addEventListener('click', (e) => {
  if (isStreaming) return;
  const rect = liveimg.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top) / rect.height;
  // marker
  const m = document.createElement('div');
  m.className = 'tap-marker';
  m.style.left = (nx * 100) + '%'; m.style.top = (ny * 100) + '%';
  feedwrap.appendChild(m);
  setTimeout(() => m.remove(), 1000);
  doTurn({kind: 'point', point: [nx, ny]});
});

/* ---- COMMANDS ---- */
const COMMANDS = [
  { cmd: 'look',     desc: 'Describe the scene',           run: () => doTurn({kind: 'snap'}) },
  { cmd: 'read',     desc: 'Read any text you can see',    run: () => doTurn({kind: 'text', text: 'Look carefully at the image. If there is any clearly readable text or sign, transcribe it exactly. If you do not see any readable text, say so explicitly. Do not invent text that is not there.'}) },
  { cmd: 'count',    desc: 'Count objects',                run: (a) => doTurn({kind: 'text', text: a ? 'How many ' + a + ' can you clearly see in the image? If none are visible, say zero. Do not guess.' : 'List the distinct objects you can clearly see and give an approximate count for each.'}) },
  { cmd: 'identify', desc: 'Identify the main subject',    run: () => doTurn({kind: 'text', text: 'What is the main subject of the image? Give a brief, confident identification only if you can clearly see one. If the image is too dark or empty, say so.'}) },
  { cmd: 'find',     desc: 'Locate a specific object',     run: (a) => doTurn({kind: 'text', text: a ? 'Can you see ' + a + ' in the image? If yes, describe where in the frame it is. If no, say it is not visible. Do not guess.' : 'Find and locate the most prominent object you can see, if any.'}) },
  { cmd: 'voice',    desc: 'Voice mode',                   run: () => doTurn({kind: 'talk'}) },
  { cmd: 'live',     desc: 'Toggle Live Mode',             run: () => toggleLive() },
  { cmd: 'clear',    desc: 'Clear conversation memory',    run: () => clearConv() },
  { cmd: 'export',   desc: 'Download conversation as Markdown', run: () => exportConv() },
  { cmd: 'settings', desc: 'Open settings drawer',         run: () => openDrawer() },
];
function runCmd(name, arg) {
  const c = COMMANDS.find(x => x.cmd === name);
  if (!c) { toast('Unknown command: /' + name, 'error'); return; }
  text.value = ''; text.style.height = 'auto'; closePalette();
  c.run(arg);
}

/* ---- palette ---- */
const palette = $('#palette');
let palOpen = false, palSel = 0, palMatches = [];
function openPalette(q) {
  palMatches = COMMANDS.filter(c => c.cmd.startsWith(q));
  if (palMatches.length === 0) { closePalette(); return; }
  palSel = 0; renderPal();
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
  text.value = ''; text.style.height = 'auto'; closePalette(); c.run();
}
palette.addEventListener('click', (e) => {
  const it = e.target.closest('.palette-item'); if (!it) return;
  palSel = +it.dataset.i; pickPal();
});

/* ---- audio meter ---- */
let audioES = null;
let levels = new Float32Array(40);
function attachAudioMeter() {
  if (audioES) return;
  audioES = new EventSource('/audio_meter');
  audioES.onmessage = (e) => {
    const lv = parseFloat(e.data); if (isNaN(lv)) return;
    levels.copyWithin(0, 1);
    levels[levels.length - 1] = lv;
    drawWave();
  };
}
function detachAudioMeter() {
  if (audioES) { audioES.close(); audioES = null; }
  levels = new Float32Array(40); drawWave();
}
function drawWave() {
  const ctx = orbwave.getContext('2d');
  const W = orbwave.width, H = orbwave.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#c15f3c'; ctx.lineWidth = 2; ctx.lineCap = 'round';
  ctx.beginPath();
  const n = levels.length;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * W;
    const y = H / 2 - levels[i] * (H / 2 - 4);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * W;
    const y = H / 2 + levels[i] * (H / 2 - 4);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

/* ---- live mode ---- */
function attachLiveStream() {
  if (liveES) return;
  liveES = new EventSource('/live/stream');
  liveES.onmessage = (e) => {
    let m; try { m = JSON.parse(e.data); } catch { return; }
    if (m.event === 'observation') appendLiveObservation(m);
    else if (m.event === 'state') {
      liveOn = !!m.running;
      livetoggle.classList.toggle('on', liveOn);
      livetoggle.querySelector('span:nth-child(2)').textContent = liveOn ? 'Live ON' : 'Live mode';
    } else if (m.event === 'error') {
      toast('Live: ' + m.error, 'error');
    }
  };
  liveES.onerror = () => { try { liveES.close(); } catch {} liveES = null; };
}
function detachLiveStream() {
  if (liveES) { try { liveES.close(); } catch {} liveES = null; }
}
function appendLiveObservation(m) {
  const empty = $('#live-empty'); if (empty) empty.style.display = 'none';
  const el = document.createElement('div');
  el.className = 'live-obs';
  el.innerHTML =
    '<img src="' + m.frame_url + '" alt="">' +
    '<div class=body>' +
      '<div class=ts>' + fmtTime(m.ts) + '</div>' +
      '<div class=text>' + escapeHtml(m.text) + '</div>' +
    '</div>';
  paneLive.appendChild(el);
  // trim
  while (paneLive.children.length > 50) paneLive.removeChild(paneLive.children[0]);
  paneLive.scrollTop = paneLive.scrollHeight;
}
async function toggleLive() {
  const action = liveOn ? '/live/stop' : '/live/start';
  const r = await fetch(action, {method: 'POST'});
  const j = await r.json();
  if (j.ok) {
    liveOn = !liveOn;
    livetoggle.classList.toggle('on', liveOn);
    livetoggle.querySelector('span:nth-child(2)').textContent = liveOn ? 'Live ON' : 'Live mode';
    if (liveOn) { attachLiveStream(); switchPane('live'); }
    toast(liveOn ? 'Live narration started' : 'Live narration stopped', 'ok');
  }
}
livetoggle.onclick = toggleLive;

/* ---- pinned ---- */
async function loadPins() {
  const r = await fetch('/pins'); const j = await r.json();
  panePins.innerHTML = '';
  if (!j.pins || j.pins.length === 0) {
    panePins.innerHTML = '<div class=empty id=pins-empty>No pinned replies yet. Hover any Jarvis reply and click the bookmark to save it here.</div>';
  } else {
    j.pins.forEach((p, i) => {
      const c = document.createElement('div');
      c.className = 'pin-card';
      c.innerHTML =
        '<div class=q>' + escapeHtml(p.question || '') + '</div>' +
        '<div class=r>' + renderMarkdown(p.reply || '') + '</div>' +
        '<button class=unpin data-i=' + i + '>Remove pin</button>';
      panePins.appendChild(c);
    });
  }
}
panePins.addEventListener('click', async (e) => {
  const b = e.target.closest('button.unpin');
  if (!b) return;
  await fetch('/pins/' + b.dataset.i, {method: 'DELETE'});
  loadPins();
});
loadPins();

/* ---- clear & export ---- */
async function clearConv() {
  await fetch('/history', {method: 'DELETE'});
  $$('#pane-conv .msg').forEach(n => n.remove());
  emptystate.style.display = 'block';
  lastJarvisMsg = null;
  toast('Conversation cleared', 'ok');
}
$('#clearbtn').onclick = clearConv;

async function exportConv() {
  const r = await fetch('/export/markdown');
  const text = await r.text();
  const blob = new Blob([text], {type: 'text/markdown'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'jarvis-' + new Date().toISOString().slice(0,19).replace(/[:T]/g,'-') + '.md';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast('Exported', 'ok');
}
$('#exportbtn').onclick = exportConv;

/* ---- settings drawer ---- */
const drawer = $('#drawer'), drawerMask = $('#drawermask');
function openDrawer() { drawer.classList.add('open'); drawerMask.classList.add('open'); loadSettings(); }
function closeDrawer() { drawer.classList.remove('open'); drawerMask.classList.remove('open'); }
$('#settingsbtn').onclick = openDrawer;
$('#closesettings').onclick = closeDrawer;
drawerMask.onclick = closeDrawer;

let SETTINGS_LOCAL = null;
let PRESETS_LOCAL = null;
async function loadSettings() {
  const s = await (await fetch('/settings')).json();
  SETTINGS_LOCAL = s;
  $('#sysprompt').value = s.system_prompt;
  $('#maxtokens').value = s.max_tokens; $('#maxtokensv').textContent = s.max_tokens;
  $('#temperature').value = Math.round(s.temperature * 10); $('#temperaturev').textContent = s.temperature.toFixed(1);
  $('#recsec').value = s.record_seconds; $('#recsecv').textContent = s.record_seconds + 's';
  $('#liveint').value = s.live_interval_s; $('#liveintv').textContent = s.live_interval_s + 's';
  // presets
  const p = await (await fetch('/presets')).json();
  PRESETS_LOCAL = p.presets;
  const row = $('#presetrow');
  row.innerHTML = Object.keys(PRESETS_LOCAL).map(name =>
    '<button class="preset-chip ' + (name === s.preset ? 'active' : '') + '" data-name="' + name + '">' + name + '</button>'
  ).join('') + '<span class="preset-chip" style="color:var(--dim);border-style:dashed">custom</span>';
}
$('#presetrow').addEventListener('click', (e) => {
  const b = e.target.closest('.preset-chip[data-name]'); if (!b) return;
  const name = b.dataset.name;
  $('#sysprompt').value = PRESETS_LOCAL[name];
  $$('.preset-chip').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
});
$('#maxtokens').oninput = () => $('#maxtokensv').textContent = $('#maxtokens').value;
$('#temperature').oninput = () => $('#temperaturev').textContent = (Number($('#temperature').value) / 10).toFixed(1);
$('#recsec').oninput = () => $('#recsecv').textContent = $('#recsec').value + 's';
$('#liveint').oninput = () => $('#liveintv').textContent = $('#liveint').value + 's';

$('#savesettings').onclick = async () => {
  const active = $('.preset-chip.active');
  const presetName = active && active.dataset.name ? active.dataset.name : 'custom';
  const body = {
    system_prompt: $('#sysprompt').value,
    preset: presetName,
    max_tokens: Number($('#maxtokens').value),
    temperature: Number($('#temperature').value) / 10,
    record_seconds: Number($('#recsec').value),
    live_interval_s: Number($('#liveint').value),
  };
  await fetch('/settings', {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  closeDrawer(); toast('Settings saved', 'ok');
};
$('#resetsettings').onclick = async () => {
  await fetch('/settings/reset', {method:'POST'});
  loadSettings(); toast('Settings reset', 'ok');
};

/* ---- shortcut modal ---- */
const shortcutsModal = $('#shortcuts-modal');
$('#shortcutsbtn').onclick = () => shortcutsModal.classList.add('open');
shortcutsModal.onclick = (e) => { if (e.target === shortcutsModal) shortcutsModal.classList.remove('open'); };

/* ---- global shortcuts ---- */
document.addEventListener('keydown', (e) => {
  // Shortcuts work when no text field is focused OR when the focused
  // text field is empty (so users can fire shortcuts without first
  // clicking outside the autofocused textarea). Modifier keys are
  // ignored to avoid clobbering native shortcuts.
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.target;
  const inField = k && (k.tagName === 'INPUT' || k.tagName === 'TEXTAREA');
  if (inField && k.value.length > 0) return;
  if (e.key === ' ' && !isStreaming) { e.preventDefault(); doTurn({kind: 'snap'}); }
  else if (e.key === '/') { e.preventDefault(); text.focus(); text.value = '/'; openPalette(''); }
  else if (e.key === 'Escape') {
    if (shortcutsModal.classList.contains('open')) shortcutsModal.classList.remove('open');
    else if (isStreaming) stopCurrent();
    else if (drawer.classList.contains('open')) closeDrawer();
  }
  else if (e.key === '?') { e.preventDefault(); shortcutsModal.classList.add('open'); }
  else if (e.key === 'l' || e.key === 'L') { e.preventDefault(); toggleLive(); }
  else if (e.key === 'r' || e.key === 'R') { e.preventDefault(); if (!isStreaming) doTurn({kind: 'regenerate'}); }
  else if (e.key === 'p' || e.key === 'P') {
    e.preventDefault();
    if (lastJarvisMsg) {
      const btn = lastJarvisMsg.querySelector('button[data-act=pin]');
      if (btn) btn.click();
    }
  }
  else if (e.key === '1') { switchPane('conv'); }
  else if (e.key === '2') { switchPane('live'); }
  else if (e.key === '3') { switchPane('pins'); }
});

async function stopCurrent() {
  if (!currentTurn) return;
  try { await fetch('/turn/' + currentTurn.id + '/stop', {method: 'POST'}); } catch {}
  toast('Stopping...');
}
$('#stopbtn').onclick = stopCurrent;

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

    def _send_bytes(self, body: bytes, mime: str, headers: dict | None = None):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
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
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        last_ts = 0.0
        try:
            while True:
                with CAMERA.lock:
                    frame = CAMERA.latest; ts = CAMERA.latest_ts
                if frame is None or ts == last_ts:
                    CAMERA.frame_event.wait(0.5); continue
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
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                try:
                    ev = ctx.q.get(timeout=30)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                    if ctx.done.is_set(): return
                    continue
                self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                self.wfile.flush()
                if ev.get("phase") in ("done", "error"): return
        except (BrokenPipeError, ConnectionResetError):
            return

    def _stream_audio_meter(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = AUDIO_MONITOR.subscribe()
        try:
            while True:
                try:
                    lvl = q.get(timeout=10)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); continue
                self.wfile.write(("data: " + f"{lvl:.3f}" + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            AUDIO_MONITOR.unsubscribe(q)

    def _stream_live(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = LIVE.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); continue
                self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            LIVE.unsubscribe(q)

    def do_GET(self):
        p = self.path
        if p == "/":
            self._send_bytes(HTML.encode(), "text/html; charset=utf-8")
        elif p == "/metrics":
            self._send_json(200, gather_metrics())
        elif p == "/history":
            self._send_json(200, {"messages": history_messages()})
        elif p == "/settings":
            with SETTINGS_LOCK:
                self._send_json(200, dict(SETTINGS))
        elif p == "/presets":
            self._send_json(200, {"presets": PRESETS})
        elif p == "/pins":
            with PINS_LOCK:
                self._send_json(200, {"pins": list(PINS)})
        elif p == "/export/markdown":
            body = export_markdown().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition", "attachment; filename=jarvis-session.md")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/stream.mjpeg":
            self._stream_mjpeg()
        elif p == "/snapshot.jpg":
            try:
                self._send_bytes(CAMERA.get_latest(), "image/jpeg")
            except TimeoutError:
                self.send_response(503); self.end_headers()
        elif p == "/audio_meter":
            self._stream_audio_meter()
        elif p == "/live/stream":
            self._stream_live()
        elif p.startswith("/events/"):
            tid = p.split("/", 2)[2]
            self._stream_events(tid)
        elif p.startswith("/audio/"):
            tid = p.split("/")[-1].replace(".wav", "")
            self._send_file(SESSION_DIR / tid / "reply.wav", "audio/wav")
        elif p.startswith("/frame/"):
            tid = p.split("/")[-1].replace(".jpg", "")
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
                for k in ("system_prompt", "preset", "max_tokens", "temperature",
                          "record_seconds", "live_interval_s",
                          "live_max_observations"):
                    if k in payload:
                        SETTINGS[k] = payload[k]
            self._send_json(200, {"ok": True})
        else:
            self.send_response(404); self.end_headers()

    def do_DELETE(self):
        p = self.path
        if p == "/history":
            history_clear()
            self._send_json(200, {"ok": True})
        elif p.startswith("/pins/"):
            try:
                idx = int(p.split("/")[2])
            except ValueError:
                self._send_json(400, {"error": "bad index"}); return
            with PINS_LOCK:
                if 0 <= idx < len(PINS):
                    PINS.pop(idx); save_pins(PINS)
                    self._send_json(200, {"ok": True})
                else:
                    self._send_json(404, {"error": "no such pin"})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        p = self.path
        if p == "/turn":
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
        elif p.startswith("/turn/") and p.endswith("/stop"):
            tid = p.split("/")[2]
            ctx = get_turn(tid)
            if ctx is None:
                self._send_json(404, {"error": "no such turn"}); return
            ctx.cancel.set()
            self._send_json(200, {"ok": True})
        elif p == "/settings/reset":
            with SETTINGS_LOCK:
                SETTINGS["system_prompt"] = DEFAULT_SYSTEM_PROMPT
                SETTINGS["preset"] = "focused"
                SETTINGS["max_tokens"] = 240
                SETTINGS["temperature"] = 0.2
                SETTINGS["record_seconds"] = 6
                SETTINGS["live_interval_s"] = 8
            self._send_json(200, {"ok": True})
        elif p == "/live/start":
            ok = LIVE.start()
            self._send_json(200, {"ok": ok, "running": LIVE.is_running()})
        elif p == "/live/stop":
            ok = LIVE.stop()
            self._send_json(200, {"ok": ok, "running": LIVE.is_running()})
        elif p == "/pins":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            with PINS_LOCK:
                PINS.append({
                    "question": payload.get("question", ""),
                    "reply":    payload.get("reply", ""),
                    "frame_url":payload.get("frame_url", ""),
                    "ts":       time.time(),
                })
                save_pins(PINS)
            self._send_json(200, {"ok": True, "count": len(PINS)})
        else:
            self.send_response(404); self.end_headers()


class JarvisServer(ThreadingHTTPServer):
    # daemon handler threads so abandoned SSE streams do not block shutdown
    daemon_threads = True
    # bigger accept backlog so MJPEG + SSE + metrics polling do not starve
    request_queue_size = 128
    allow_reuse_address = True


def main():
    CAMERA.start()
    threading.Thread(target=_vlm_health_poller, daemon=True).start()
    print(f"jarvis listening on http://{LISTEN_HOST}:{LISTEN_PORT}/")
    JarvisServer((LISTEN_HOST, LISTEN_PORT), H).serve_forever()


if __name__ == "__main__":
    main()
