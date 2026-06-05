#!/usr/bin/env python3
"""Jarvis voice loop on Jetson Orin Nano Super.

Pipeline per turn:
    1. record N seconds from C615 mic via ffmpeg
    2. whisper.cpp tiny.en -> transcription
    3. capture frame from C615 video
    4. POST to llama-server /v1/chat/completions with frame + question
    5. piper TTS the response
    6. serve audio + transcript to the browser

UI: tiny single-page web app at http://<jetson>:8085/
    Click TALK -> 6s recording -> processing -> auto-play response.
"""
from __future__ import annotations
import base64, json, os, subprocess, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request

LAB = Path("/home/zip/jarvis-lab")
WHISPER_BIN = LAB / "build/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = LAB / "build/whisper.cpp/models/ggml-tiny.en.bin"
PIPER_BIN = LAB / "piper/piper/piper"
PIPER_VOICE = LAB / "piper/voices/en_US-amy-medium.onnx"
VLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8085

SESSION_DIR = LAB / "logs/sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are Jarvis, the user's personal AI agent. You see what the camera "
    "sees and you hear what the user says. Answer briefly and helpfully. If "
    "asked to read a sign or transcribe text, transcribe it exactly. If unsure, "
    "say so. Keep responses under 40 words unless explicitly asked for more."
)


# ----- pipeline steps ---------------------------------------------------------
def record_audio(out_wav: Path, seconds: int = RECORD_SECONDS) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
         "-f", "alsa", "-ar", str(SAMPLE_RATE), "-ac", "1",
         "-i", "plughw:CARD=C615,DEV=0",
         "-t", str(seconds), str(out_wav)],
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
    return txt.read_text().strip() if txt.exists() else ""


def capture_frame(out_jpg: Path) -> None:
    subprocess.run(
        [str(LAB / "scripts/capture_frame.sh"), str(out_jpg)],
        check=True, capture_output=True, timeout=10,
    )


def ask_vlm(question: str, frame_jpg: Path) -> str:
    b64 = base64.b64encode(frame_jpg.read_bytes()).decode()
    body = {
        "model": "qwen2.5-vl-3b",
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": question},
            ]},
        ],
        "max_tokens": 200,
    }
    req = urllib.request.Request(
        VLM_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        obj = json.loads(r.read())
    return obj["choices"][0]["message"]["content"].strip()


def synthesize(text: str, out_wav: Path) -> None:
    subprocess.run(
        [str(PIPER_BIN), "--model", str(PIPER_VOICE),
         "--output_file", str(out_wav)],
        input=text, capture_output=True, text=True, check=True, timeout=30,
    )


# ----- per-turn orchestration --------------------------------------------------
def run_turn(turn_id: str, status_cb) -> dict:
    work = SESSION_DIR / turn_id
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    timings = {}

    status_cb({"phase": "recording"})
    record_audio(work / "user.wav")
    timings["record_s"] = round(time.monotonic() - t0, 2); t0 = time.monotonic()

    status_cb({"phase": "transcribing"})
    transcription = transcribe(work / "user.wav")
    timings["transcribe_s"] = round(time.monotonic() - t0, 2); t0 = time.monotonic()

    status_cb({"phase": "capturing"})
    capture_frame(work / "frame.jpg")
    timings["capture_s"] = round(time.monotonic() - t0, 2); t0 = time.monotonic()

    status_cb({"phase": "thinking", "transcription": transcription})
    if not transcription:
        reply = "I did not catch that. Please try again."
    else:
        reply = ask_vlm(transcription, work / "frame.jpg")
    timings["vlm_s"] = round(time.monotonic() - t0, 2); t0 = time.monotonic()

    status_cb({"phase": "speaking", "reply": reply})
    synthesize(reply, work / "reply.wav")
    timings["tts_s"] = round(time.monotonic() - t0, 2)

    return {
        "turn_id": turn_id,
        "transcription": transcription,
        "reply": reply,
        "timings": timings,
        "audio_url": f"/audio/{turn_id}.wav",
        "frame_url": f"/frame/{turn_id}.jpg",
    }


# ----- HTTP server -------------------------------------------------------------
STATE_LOCK = threading.Lock()
CURRENT = {"phase": "idle"}


def set_state(s):
    with STATE_LOCK:
        CURRENT.clear()
        CURRENT.update(s)


HTML = """<!doctype html>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Jarvis</title>
<style>
:root { color-scheme: dark; }
body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2em auto;
       padding: 0 1em; background:#0b0c10; color:#e6e6e6; }
h1 { font-weight: 200; letter-spacing: 0.1em; }
button { font-size: 1.5em; padding: 1em 2em; border: 1px solid #4af;
         background:#0e1a30; color:#cfe; border-radius:1em; cursor:pointer; }
button:disabled { opacity:0.5; cursor:wait; }
#status { margin: 1em 0; font-family: monospace; color:#9cf; min-height:1.5em; }
.bubble { background:#14182a; border-radius:1em; padding:1em; margin:0.5em 0; }
.user { border-left:3px solid #4af; }
.jarvis { border-left:3px solid #fa6; }
.timings { font-size:0.85em; color:#888; font-family:monospace; }
img { max-width:100%; border-radius:0.5em; display:block; }
audio { width:100%; }
hr { border:none; border-top:1px solid #333; margin:1.5em 0; }
</style>
<h1>JARVIS</h1>
<button id=talk>TALK (6s)</button>
<div id=status>idle</div>
<div id=feed></div>
<script>
const btn = document.getElementById("talk");
const statusEl = document.getElementById("status");
const feed = document.getElementById("feed");

async function tick() {
  try {
    const s = await (await fetch("/state")).json();
    let line = "phase: " + s.phase;
    if (s.transcription) line += "  you: " + s.transcription;
    if (s.reply) line += "  jarvis: " + s.reply;
    statusEl.textContent = line;
  } catch(e) {}
}
setInterval(tick, 500);

btn.onclick = async () => {
  btn.disabled = true;
  statusEl.textContent = "starting turn...";
  try {
    const r = await fetch("/turn", {method:"POST"});
    const j = await r.json();
    if (j.error) { statusEl.textContent = "ERROR: " + j.error; return; }
    const div = document.createElement("div");
    div.innerHTML =
      '<div class=bubble><b>frame</b><br><img src=' + j.frame_url + '></div>' +
      '<div class="bubble user">you: ' + (j.transcription || '<i>(silent)</i>') + '</div>' +
      '<div class="bubble jarvis">jarvis: ' + j.reply + '</div>' +
      '<div class=timings>' + JSON.stringify(j.timings) + '</div>' +
      '<audio controls autoplay src=' + j.audio_url + '></audio>' +
      '<hr>';
    feed.prepend(div);
  } finally { btn.disabled = false; }
};
</script>
"""


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        return  # quiet

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, mime: str):
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            with STATE_LOCK:
                self._send_json(200, dict(CURRENT))
        elif self.path.startswith("/audio/"):
            tid = self.path.split("/")[-1].replace(".wav", "")
            self._send_file(SESSION_DIR / tid / "reply.wav", "audio/wav")
        elif self.path.startswith("/frame/"):
            tid = self.path.split("/")[-1].replace(".jpg", "")
            self._send_file(SESSION_DIR / tid / "frame.jpg", "image/jpeg")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/turn":
            self.send_response(404)
            self.end_headers()
            return
        tid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        try:
            result = run_turn(tid, set_state)
            set_state({"phase": "idle"})
            self._send_json(200, result)
        except subprocess.CalledProcessError as e:
            set_state({"phase": "error"})
            self._send_json(500, {
                "error": f"subprocess failed: {e.cmd[0]} -> {e.returncode}",
                "stderr": (e.stderr or "")[:500],
            })
        except Exception as e:
            set_state({"phase": "error"})
            self._send_json(500, {"error": str(e)})


def main():
    print(f"jarvis_voice listening on http://{LISTEN_HOST}:{LISTEN_PORT}/")
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), H).serve_forever()


if __name__ == "__main__":
    main()
