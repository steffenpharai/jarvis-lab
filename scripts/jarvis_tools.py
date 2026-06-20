"""
jarvis_tools.py — tool catalog for Jarvis Lab

Self-contained tool registry. Imported by jarvis_voice.py at startup. Provides:

    TOOLS.set_context(ctx)            # bind the orchestrator handles
    TOOLS.migrate(memory)             # create new SQLite tables
    TOOLS.catalog()                   # list of tool schemas
    TOOLS.call(name, args)            # dispatch with safety gating
    TOOLS.reminder_tick(memory)       # background fire loop

Tools are grouped: vision / audio / memory / self / reason / productivity / web.
Each tool declares a JSON Schema for args, a category, and a safety level
(safe | gated | dangerous). Gated tools require ?confirm=1 on the HTTP call.

Zero new pip deps. Uses httpx, requests, numpy, stdlib only.
"""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import json
import math
import operator as op
import os
import random
import re
import secrets
import shutil
import socket
import sqlite3
import string
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid as uuid_mod
import xml.etree.ElementTree as ET
import zoneinfo
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx


# ============================================================================
# REGISTRY
# ============================================================================

@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    category: str
    safety: str
    handler: Callable[..., Any]
    needs_ctx: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._ctx: ToolContext | None = None
        self._reminder_thread: threading.Thread | None = None
        self._reminder_stop = threading.Event()
        # notifications: in-memory ring buffer + SSE subscribers
        import collections as _coll
        import queue as _q
        self._notifications: "_coll.deque[dict]" = _coll.deque(maxlen=200)
        self._note_subscribers: list = []
        self._note_lock = threading.Lock()
        self._note_seq = 0

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name: {t.name}")
        self._tools[t.name] = t

    def set_context(self, ctx: "ToolContext") -> None:
        self._ctx = ctx

    def catalog(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "schema": t.schema,
                "category": t.category,
                "safety": t.safety,
            }
            for t in sorted(self._tools.values(), key=lambda x: (x.category, x.name))
        ]

    def call(self, name: str, args: dict, *, confirmed: bool = False) -> dict:
        t = self._tools.get(name)
        if t is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        if t.safety in ("gated", "dangerous") and not confirmed:
            return {
                "ok": False,
                "error": f"tool '{name}' requires confirmation (POST ?confirm=1)",
                "safety": t.safety,
            }
        try:
            kwargs = self._coerce(t, args or {})
            if t.needs_ctx:
                if self._ctx is None:
                    return {"ok": False, "error": "tool context not initialised"}
                result = t.handler(self._ctx, **kwargs)
            else:
                result = t.handler(**kwargs)
            return {"ok": True, "result": result, "tool": name}
        except ToolError as e:
            return {"ok": False, "error": str(e), "tool": name}
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "tool": name,
            }

    def _coerce(self, t: Tool, args: dict) -> dict:
        # very light type coercion based on schema
        props = (t.schema or {}).get("properties", {}) or {}
        out = {}
        for k, v in args.items():
            spec = props.get(k, {})
            typ = spec.get("type")
            if typ == "integer" and isinstance(v, str):
                try:
                    v = int(v)
                except ValueError:
                    pass
            elif typ == "number" and isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    pass
            elif typ == "boolean" and isinstance(v, str):
                v = v.lower() in ("1", "true", "yes", "on")
            out[k] = v
        return out

    # --- DB migration -----------------------------------------------------
    SCHEMA_EXT = """
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        body  TEXT NOT NULL,
        tags  TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at DESC);

    CREATE TABLE IF NOT EXISTS todos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        due_at REAL,
        done INTEGER DEFAULT 0,
        done_at REAL,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_todos_open ON todos(done, due_at);

    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        fire_at REAL NOT NULL,
        fired INTEGER DEFAULT 0,
        fired_at REAL,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_reminders_fire ON reminders(fired, fire_at);

    CREATE TABLE IF NOT EXISTS bookmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        title TEXT,
        tags TEXT,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_bookmarks_created ON bookmarks(created_at DESC);

    CREATE TABLE IF NOT EXISTS journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        body TEXT NOT NULL,
        day TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_journal_day ON journal(day);

    CREATE TABLE IF NOT EXISTS tool_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        args_json TEXT,
        result_json TEXT,
        ok INTEGER,
        ms REAL,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_tool_calls_created ON tool_calls(created_at DESC);
    """

    def migrate(self, memory) -> None:
        with memory.lock:
            memory._con.executescript(self.SCHEMA_EXT)

    def log_call(self, memory, name: str, args: dict, result: dict, ms: float) -> None:
        try:
            with memory.lock:
                memory._con.execute(
                    """INSERT INTO tool_calls (name, args_json, result_json, ok, ms, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        name,
                        json.dumps(args)[:8000],
                        json.dumps(result)[:8000],
                        1 if result.get("ok") else 0,
                        ms,
                        time.time(),
                    ),
                )
        except Exception:
            pass

    # --- notifications ---------------------------------------------------
    def _next_note_id(self) -> int:
        with self._note_lock:
            self._note_seq += 1
            return self._note_seq

    def emit_notification(self, note: dict) -> dict:
        """Add a notification to the ring buffer and fan it out to all
        active SSE subscribers. Returns the stored record."""
        if "id" not in note:
            note["id"] = self._next_note_id()
        note.setdefault("ts", time.time())
        with self._note_lock:
            self._notifications.append(note)
            subs = list(self._note_subscribers)
        for q in subs:
            try:
                q.put_nowait(note)
            except Exception:
                pass
        return note

    def notifications(self, *, since_id: int = 0) -> list[dict]:
        with self._note_lock:
            return [n for n in self._notifications if n.get("id", 0) > since_id]

    def subscribe_notifications(self):
        import queue as _q
        q = _q.Queue(maxsize=256)
        with self._note_lock:
            self._note_subscribers.append(q)
        return q

    def unsubscribe_notifications(self, q) -> None:
        with self._note_lock:
            if q in self._note_subscribers:
                self._note_subscribers.remove(q)

    # --- reminder background tick ----------------------------------------
    def start_reminder_loop(self, memory) -> None:
        if self._reminder_thread is not None:
            return

        def _fire(rid: int, text: str, fired_at: float) -> None:
            """Synthesize TTS for a fired reminder and emit a notification."""
            note: dict = {
                "kind": "reminder",
                "reminder_id": rid,
                "text": text,
                "audio_url": None,
            }
            try:
                ctx = self._ctx
                if ctx is not None and ctx.session_dir:
                    note_dir = ctx.session_dir / "notifications"
                    note_dir.mkdir(parents=True, exist_ok=True)
                    wav = note_dir / f"reminder-{rid}.wav"
                    spoken = f"Reminder: {text}"
                    try:
                        ctx.synthesize(spoken, wav)
                        if wav.exists():
                            note["audio_url"] = f"/audio_note/reminder-{rid}.wav"
                    except Exception as e:  # noqa: BLE001
                        note["tts_error"] = str(e)
            except Exception as e:  # noqa: BLE001
                note["tts_error"] = str(e)
            self.emit_notification(note)
            print(
                f"[reminder] fired #{rid}: {text!r} "
                f"(emitted notification {note['id']})",
                flush=True,
            )

        def _loop():
            while not self._reminder_stop.is_set():
                try:
                    now = time.time()
                    with memory.lock:
                        rows = memory._con.execute(
                            "SELECT id, text, fire_at FROM reminders "
                            "WHERE fired=0 AND fire_at<=? ORDER BY fire_at",
                            (now,),
                        ).fetchall()
                        ids = [r[0] for r in rows]
                        if ids:
                            placeholders = ",".join("?" * len(ids))
                            memory._con.execute(
                                f"UPDATE reminders SET fired=1, fired_at=? "
                                f"WHERE id IN ({placeholders})",
                                (now, *ids),
                            )
                    for rid, text, _fa in rows:
                        _fire(rid, text, now)
                except Exception as e:  # noqa: BLE001
                    print(f"[reminder] tick error: {e}", flush=True)
                # tick more frequently so a "in 1 minute" reminder fires near
                # its due time
                self._reminder_stop.wait(10.0)

        self._reminder_thread = threading.Thread(
            target=_loop, name="reminder-tick", daemon=True
        )
        self._reminder_thread.start()


TOOLS = ToolRegistry()


class ToolError(Exception):
    """Raised by tool handlers to signal a clean error back to the caller."""


def tool(
    name: str,
    *,
    description: str,
    category: str,
    schema: dict | None = None,
    safety: str = "safe",
    needs_ctx: bool = False,
):
    """Decorator that registers a tool."""
    def deco(fn):
        # local-only by default; set JARVIS_ALLOW_CLOUD=1 to enable cloud escalation tools
        if category == "cloud" and os.environ.get("JARVIS_ALLOW_CLOUD", "0") != "1":
            return fn
        TOOLS.register(
            Tool(
                name=name,
                description=description,
                schema=schema or {"type": "object", "properties": {}},
                category=category,
                safety=safety,
                handler=fn,
                needs_ctx=needs_ctx,
            )
        )
        return fn
    return deco


# ============================================================================
# CONTEXT (handles injected by jarvis_voice.py)
# ============================================================================

@dataclass
class ToolContext:
    memory: Any
    camera: Any           # CameraStreamer (.get_latest() -> bytes)
    audio: Any            # AudioBus
    live: Any             # LiveMode
    wake: Any             # WakeWordListener
    settings: dict        # SETTINGS
    settings_lock: threading.Lock
    presets: dict         # PRESETS
    set_wake_enabled: Callable[[bool], bool]
    capture_frame: Callable[..., dict]      # capture_frame_for_vlm
    stream_vlm: Callable[..., tuple]        # stream_vlm
    transcribe: Callable[[Path], str]
    synthesize: Callable[[str, Path], None]
    record_voice: Callable[..., dict]       # record_voice_via_bus
    lab_root: Path
    session_dir: Path
    vlm_w: int = 512
    vlm_h: int = 384
    cam_w: int = 1280
    cam_h: int = 720
    mic_device: str = ""
    visual_memory: Any = None   # VisualMemory (ambient captioner)


# ============================================================================
# HELPERS
# ============================================================================

HTTP = httpx.Client(
    timeout=httpx.Timeout(10.0, connect=5.0),
    headers={"User-Agent": "jarvis-lab/1.0 (+https://github.com/steffenpharai/jarvis-lab)"},
    follow_redirects=True,
)

# Serializes all VLM image inference. On the 8GB Orin, concurrent mmproj image
# calls (e.g. an interactive investigate while the ambient captioner fires) spike
# memory and SIGSEGV the server. Interactive callers block on this; background
# callers (captioner/watcher) try-acquire and skip when busy.
VLM_BUSY = threading.Lock()


def _tmp_jpg(ctx: ToolContext, tag: str) -> Path:
    base = ctx.session_dir / "tools"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{tag}-{uuid_mod.uuid4().hex[:8]}.jpg"


def _hd_capture(ctx: ToolContext, out_jpg: Path) -> None:
    """Capture a 1280x720 frame (native resolution) for OCR-quality work."""
    raw = ctx.camera.get_latest()
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "mjpeg", "-i", "-",
            "-vf", f"scale={ctx.cam_w}:{ctx.cam_h}:flags=lanczos",
            "-frames:v", "1", "-q:v", "3", str(out_jpg),
        ],
        input=raw, check=True, capture_output=True, timeout=10,
    )


def _vlm_oneshot(
    ctx: ToolContext, prompt: str, frame: Path,
    *, max_seconds: float = 25.0, include_history: bool = False,
) -> str:
    cancel = threading.Event()
    chunks: list[str] = []

    def _collect(delta: str) -> None:
        chunks.append(delta)

    deadline = time.monotonic() + max_seconds
    threading.Timer(max_seconds, cancel.set).start()
    text, _usage = ctx.stream_vlm(
        prompt, frame, cancel, _collect,
        include_history=include_history,
        read_timeout_s=max_seconds,
    )
    if not text and chunks:
        text = "".join(chunks)
    return text.strip()


def _now() -> float:
    return time.time()


def _parse_when(when: str) -> float:
    """Parse "in 5 minutes" / "at 14:30" / "tomorrow 9am" — minimal grammar."""
    s = (when or "").strip().lower()
    now = time.localtime()
    # "in <n> <unit>"
    m = re.match(r"in\s+(\d+)\s*(s|sec|second|m|min|minute|h|hr|hour|d|day)s?", s)
    if m:
        n = int(m.group(1)); u = m.group(2)
        mult = {"s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60,
                "h": 3600, "hr": 3600, "hour": 3600, "d": 86400, "day": 86400}[u]
        return time.time() + n * mult
    # absolute "HH:MM" (today, or tomorrow if past)
    m = re.match(r"(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?", s)
    if m:
        h = int(m.group(1)); mm = int(m.group(2)); ampm = m.group(3)
        if ampm == "pm" and h < 12:
            h += 12
        if ampm == "am" and h == 12:
            h = 0
        t = list(now)
        t[3] = h; t[4] = mm; t[5] = 0
        target = time.mktime(time.struct_time(tuple(t)))
        if target < time.time():
            target += 86400
        return target
    # bare epoch float?
    try:
        return float(s)
    except ValueError:
        raise ToolError(
            "could not parse time. try 'in 10 minutes', '14:30', '9am'."
        )


# ============================================================================
# VISION TOOLS
# ============================================================================

@tool(
    "zoom_into",
    description="Re-capture the current camera frame at a tighter crop and "
                "ask the VLM about it. Use for distant signs, small print, "
                "or specific regions. Coordinates are normalised 0-1.",
    category="vision",
    schema={
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "left edge, 0-1 (default 0.25)"},
            "y": {"type": "number", "description": "top edge, 0-1 (default 0.25)"},
            "w": {"type": "number", "description": "width, 0-1 (default 0.5)"},
            "h": {"type": "number", "description": "height, 0-1 (default 0.5)"},
            "question": {"type": "string",
                         "description": "what to ask about the cropped region"},
        },
        "required": ["question"],
    },
    needs_ctx=True,
)
def zoom_into(ctx, x: float = 0.25, y: float = 0.25, w: float = 0.5, h: float = 0.5,
              question: str = "Describe what you see.") -> dict:
    out = _tmp_jpg(ctx, "zoom")
    ctx.capture_frame(out, crop=(x, y, w, h), allow_reuse=False)
    answer = _vlm_oneshot(ctx, question, out, max_seconds=20.0)
    return {"answer": answer, "crop": [x, y, w, h], "frame": str(out)}


@tool(
    "read_all_text",
    description="Capture a high-resolution (1280x720) frame and read all "
                "visible text. Use for menus, signs, receipts, screens.",
    category="vision",
    needs_ctx=True,
)
def read_all_text(ctx) -> dict:
    out = _tmp_jpg(ctx, "ocr")
    _hd_capture(ctx, out)
    prompt = (
        "Transcribe every piece of readable text in this image, verbatim. "
        "Preserve line breaks. If there is no text, reply 'NO_TEXT'. "
        "Do not describe the scene, just output the text."
    )
    text = _vlm_oneshot(ctx, prompt, out, max_seconds=30.0)
    return {"text": text, "frame": str(out)}


@tool(
    "export_dataset",
    description="Export the on-device visual memory + grounded Q&A into a "
                "portable, standards-aligned training dataset (Open-X / LeRobot "
                "JSONL + frames) with a consent/provenance card. Returns counts "
                "and a download URL. Observational vision-language data only "
                "(no robot action labels — that needs the Zip robot drive loop).",
    category="memory",
    schema={
        "type": "object",
        "properties": {
            "since_days": {"type": "number",
                           "description": "only include data newer than N days"},
            "limit": {"type": "integer",
                      "description": "max keyframes (default 5000, cap 50000)"},
            "include_frames": {"type": "boolean",
                               "description": "copy the JPEG pixels (default true)"},
        },
    },
)
def export_dataset(since_days: float | None = None, limit: int = 5000,
                   include_frames: bool = True) -> dict:
    body = {"limit": int(limit), "include_frames": bool(include_frames)}
    if since_days is not None:
        body["since_days"] = float(since_days)
    r = HTTP.post("http://127.0.0.1:8085/dataset/export", json=body,
                  timeout=httpx.Timeout(180.0, connect=5.0))
    r.raise_for_status()
    return r.json()


@tool(
    "multi_frame_compare",
    description="Capture N frames spaced over the next few seconds and "
                "compare them. Useful for change detection and motion.",
    category="vision",
    schema={
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "frames to capture (2-4)"},
            "spacing_s": {"type": "number", "description": "seconds between frames"},
            "question": {"type": "string"},
        },
    },
    needs_ctx=True,
)
def multi_frame_compare(ctx, n: int = 3, spacing_s: float = 1.0,
                        question: str = "What changed across these frames?") -> dict:
    n = max(2, min(4, int(n)))
    frames: list[Path] = []
    for i in range(n):
        p = _tmp_jpg(ctx, f"mfc-{i}")
        ctx.capture_frame(p, allow_reuse=False)
        frames.append(p)
        if i < n - 1:
            time.sleep(max(0.0, float(spacing_s)))
    # Encode all frames into one VLM call as separate image_url entries.
    answer = _multi_image_vlm(ctx, question, frames, max_seconds=30.0)
    return {"answer": answer, "frames": [str(p) for p in frames]}


def _multi_image_vlm(ctx: ToolContext, prompt: str, frames: list[Path],
                     *, max_seconds: float = 30.0) -> str:
    # Build the chat completion body manually — stream_vlm only supports
    # one frame, so we hit the endpoint directly here.
    parts = []
    for p in frames:
        b64 = base64.b64encode(p.read_bytes()).decode()
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    parts.append({"type": "text", "text": prompt})
    with ctx.settings_lock:
        sysp = ctx.settings.get("system_prompt", "")
        max_t = ctx.settings.get("max_tokens", 240)
        temp = ctx.settings.get("temperature", 0.2)
    body = {
        "model": "qwen2.5-vl-3b",
        "stream": False,
        "messages": [
            {"role": "system", "content": sysp},
            {"role": "user", "content": parts},
        ],
        "max_tokens": max_t,
        "temperature": temp,
    }
    with VLM_BUSY:
        r = httpx.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            json=body, timeout=httpx.Timeout(max_seconds, connect=5.0),
        )
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()


@tool(
    "timelapse",
    description="Capture a burst of frames at fixed intervals.",
    category="vision",
    schema={
        "type": "object",
        "properties": {
            "every_s": {"type": "number"},
            "count": {"type": "integer"},
        },
    },
    needs_ctx=True,
)
def timelapse(ctx, every_s: float = 2.0, count: int = 6) -> dict:
    count = max(2, min(30, int(count)))
    out_dir = ctx.session_dir / "tools" / f"timelapse-{uuid_mod.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = out_dir / f"f-{i:03d}.jpg"
        ctx.capture_frame(p, allow_reuse=False)
        paths.append(str(p))
        if i < count - 1:
            time.sleep(max(0.1, float(every_s)))
    return {"frames": paths, "count": count, "every_s": every_s, "dir": str(out_dir)}


@tool(
    "track_object",
    description="Use YOLO11 (perception-lab) to detect/track a class in the "
                "current frame. Returns bounding boxes for the detected class.",
    category="vision",
    schema={
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    },
    needs_ctx=True,
)
def track_object(ctx, label: str) -> dict:
    detect_py = ctx.lab_root.parent / "perception-lab" / "detect.py"
    if not detect_py.exists():
        raise ToolError(f"perception-lab/detect.py not found at {detect_py}")
    snap = _tmp_jpg(ctx, "track")
    ctx.camera.get_latest()  # warm
    _hd_capture(ctx, snap)
    try:
        r = subprocess.run(
            [sys.executable, str(detect_py),
             "--image", str(snap), "--class", label, "--json"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        raise ToolError("track_object: detect.py timed out")
    if r.returncode != 0:
        # detect.py may not accept these flags — surface stdout/stderr
        return {
            "label": label, "frame": str(snap),
            "raw_stdout": r.stdout[-2000:], "raw_stderr": r.stderr[-2000:],
            "hint": "detect.py may need a --json flag; falling back to raw output",
        }
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"label": label, "frame": str(snap), "raw": r.stdout[-2000:]}


@tool(
    "depth_of",
    description="Estimate depth at a point in the current frame. "
                "Coordinates are normalised 0-1.",
    category="vision",
    schema={
        "type": "object",
        "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
        },
    },
    needs_ctx=True,
)
def depth_of(ctx, x: float = 0.5, y: float = 0.5) -> dict:
    depth_py = ctx.lab_root.parent / "depth-lab" / "depth.py"
    if not depth_py.exists():
        raise ToolError(f"depth-lab/depth.py not found at {depth_py}")
    snap = _tmp_jpg(ctx, "depth")
    _hd_capture(ctx, snap)
    try:
        r = subprocess.run(
            [sys.executable, str(depth_py), "--image", str(snap),
             "--at", f"{x},{y}", "--json"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise ToolError("depth_of: depth.py timed out")
    out = {"x": x, "y": y, "frame": str(snap)}
    if r.returncode == 0:
        try:
            out.update(json.loads(r.stdout))
        except json.JSONDecodeError:
            out["raw"] = r.stdout[-2000:]
    else:
        out["raw_stdout"] = r.stdout[-2000:]
        out["raw_stderr"] = r.stderr[-2000:]
        out["hint"] = "depth.py may not accept these flags; consult its --help"
    return out


@tool(
    "scan_room",
    description="Kick off a 3D Gaussian Splat capture of the current scene "
                "(via splat-lab/capture.py + bake.py). Long-running; returns "
                "a job id.",
    category="vision",
    safety="gated",
    needs_ctx=True,
)
def scan_room(ctx) -> dict:
    cap = ctx.lab_root.parent / "splat-lab" / "scripts" / "capture.py"
    bake = ctx.lab_root.parent / "splat-lab" / "scripts" / "bake.py"
    if not cap.exists() or not bake.exists():
        raise ToolError(f"splat-lab scripts not found ({cap}, {bake})")
    job_id = "scan-" + uuid_mod.uuid4().hex[:8]
    log_path = ctx.session_dir / "tools" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"({sys.executable} {cap} && {sys.executable} {bake}) "
        f">>{log_path} 2>&1"
    )
    subprocess.Popen(cmd, shell=True, start_new_session=True)
    return {
        "job_id": job_id,
        "log": str(log_path),
        "hint": "tail -f the log; final scene appears under splat-lab/scenes/",
    }


# ============================================================================
# AUDIO TOOLS
# ============================================================================

@tool(
    "record_ambient",
    description="Record N seconds from the mic and return what was said.",
    category="audio",
    schema={
        "type": "object",
        "properties": {"seconds": {"type": "integer"}},
    },
    needs_ctx=True,
)
def record_ambient(ctx, seconds: int = 5) -> dict:
    seconds = max(1, min(30, int(seconds)))
    out_dir = ctx.session_dir / "tools" / f"rec-{uuid_mod.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio.wav"
    info = ctx.record_voice(wav, max_seconds=seconds)
    text = ""
    try:
        text = ctx.transcribe(wav) if wav.exists() and wav.stat().st_size > 100 else ""
    except Exception as e:  # noqa: BLE001
        text = f"(transcribe error: {e})"
    return {"transcription": text, "audio": str(wav), "info": info}


@tool(
    "monitor_for",
    description="Listen for up to timeout_s for any speech. Returns the first "
                "transcription that contains the label phrase, or empty if "
                "none.",
    category="audio",
    schema={
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "timeout_s": {"type": "integer"},
        },
        "required": ["label"],
    },
    needs_ctx=True,
)
def monitor_for(ctx, label: str, timeout_s: int = 30) -> dict:
    timeout_s = max(5, min(120, int(timeout_s)))
    deadline = time.time() + timeout_s
    needle = (label or "").lower().strip()
    if not needle:
        raise ToolError("label required")
    out_dir = ctx.session_dir / "tools" / f"mon-{uuid_mod.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    heard = []
    chunk_i = 0
    while time.time() < deadline:
        wav = out_dir / f"chunk-{chunk_i:03d}.wav"
        try:
            ctx.record_voice(wav, max_seconds=5)
        except Exception:
            break
        chunk_i += 1
        if not wav.exists() or wav.stat().st_size < 200:
            continue
        try:
            text = ctx.transcribe(wav)
        except Exception:
            text = ""
        heard.append(text)
        if needle in text.lower():
            return {"matched": True, "text": text, "elapsed_s": round(time.time() - (deadline - timeout_s), 2)}
    return {"matched": False, "text": " | ".join(heard)[-2000:],
            "elapsed_s": timeout_s}


@tool(
    "play_sound",
    description="Speak text through the current audio output via Piper TTS. "
                "Returns the URL to the synthesised WAV.",
    category="audio",
    schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    needs_ctx=True,
)
def play_sound(ctx, text: str) -> dict:
    out_dir = ctx.session_dir / "tools" / f"tts-{uuid_mod.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "say.wav"
    ctx.synthesize(text, wav)
    return {"audio": str(wav), "url": f"/audio_tool/{wav.parent.name}/{wav.name}"}


@tool(
    "transcribe_file",
    description="Transcribe an audio file already on disk.",
    category="audio",
    schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    needs_ctx=True,
)
def transcribe_file(ctx, path: str) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        raise ToolError(f"file not found: {p}")
    text = ctx.transcribe(p)
    return {"text": text, "file": str(p)}


# ============================================================================
# MEMORY TOOLS
# ============================================================================

@tool(
    "recall",
    description="Search the conversation memory (FTS5 today; semantic in "
                "Sprint C). Returns up to k matching past turns.",
    category="memory",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer"},
        },
        "required": ["query"],
    },
    needs_ctx=True,
)
def recall(ctx, query: str, k: int = 5) -> dict:
    k = max(1, min(50, int(k)))
    items = ctx.memory.search(query, k)
    return {"query": query, "count": len(items), "items": items}


@tool(
    "recent",
    description="The last N turns in time order.",
    category="memory",
    schema={"type": "object", "properties": {"n": {"type": "integer"}}},
    needs_ctx=True,
)
def recent(ctx, n: int = 10) -> dict:
    n = max(1, min(100, int(n)))
    return {"items": ctx.memory.recent(n)}


@tool(
    "pinned",
    description="All pinned turns.",
    category="memory",
    needs_ctx=True,
)
def pinned(ctx) -> dict:
    return {"items": ctx.memory.pinned()}


@tool(
    "pin_this",
    description="Pin a turn by its turn_id.",
    category="memory",
    schema={
        "type": "object",
        "properties": {"turn_id": {"type": "string"}},
        "required": ["turn_id"],
    },
    needs_ctx=True,
)
def pin_this(ctx, turn_id: str) -> dict:
    ctx.memory.set_pin(turn_id, True)
    return {"pinned": turn_id}


@tool(
    "unpin",
    description="Unpin a turn by its turn_id.",
    category="memory",
    schema={
        "type": "object",
        "properties": {"turn_id": {"type": "string"}},
        "required": ["turn_id"],
    },
    needs_ctx=True,
)
def unpin(ctx, turn_id: str) -> dict:
    ctx.memory.set_pin(turn_id, False)
    return {"unpinned": turn_id}


@tool(
    "forget",
    description="Soft-delete turns matching a query. Gated.",
    category="memory",
    safety="gated",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    needs_ctx=True,
)
def forget(ctx, query: str) -> dict:
    matches = ctx.memory.search(query, 50)
    ids = [m["turn_id"] for m in matches]
    if not ids:
        return {"deleted": 0}
    with ctx.memory.lock:
        for tid in ids:
            ctx.memory._con.execute(
                "UPDATE turns SET reply='[forgotten]', question='[forgotten]' "
                "WHERE turn_id=?",
                (tid,),
            )
    return {"deleted": len(ids), "turn_ids": ids}


@tool(
    "summarize_today",
    description="Use the VLM to produce a short summary of today's "
                "conversation turns.",
    category="memory",
    needs_ctx=True,
)
def summarize_today(ctx) -> dict:
    day_start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    with ctx.memory.lock:
        rows = ctx.memory._con.execute(
            "SELECT question, reply, created_at FROM turns "
            "WHERE created_at >= ? ORDER BY created_at",
            (day_start,),
        ).fetchall()
    if not rows:
        return {"summary": "No turns yet today.", "count": 0}
    transcript = "\n".join(
        f"- Q: {q or '(no question)'} -> A: {(r or '')[:200]}"
        for q, r, _ in rows[:50]
    )
    prompt = (
        "Summarise this conversation log into 3-5 short bullet points. "
        "Focus on the topics covered and any decisions or facts mentioned.\n\n"
        + transcript
    )
    # Use most recent frame so the VLM endpoint accepts the call.
    snap = _tmp_jpg(ctx, "sum")
    ctx.capture_frame(snap, allow_reuse=True)
    summary = _vlm_oneshot(ctx, prompt, snap, max_seconds=40.0)
    return {"summary": summary, "count": len(rows)}


@tool(
    "list_entities",
    description="List entities extracted from past turns. Stub until Sprint C.",
    category="memory",
    schema={
        "type": "object",
        "properties": {"type": {"type": "string"}},
    },
    needs_ctx=True,
)
def list_entities(ctx, type: str = "") -> dict:
    return {
        "entities": [],
        "note": "entity extraction lands in Sprint C; this is a placeholder",
    }


@tool(
    "whats_new",
    description="Turns since a given offset (seconds ago).",
    category="memory",
    schema={
        "type": "object",
        "properties": {"since_s": {"type": "integer"}},
    },
    needs_ctx=True,
)
def whats_new(ctx, since_s: int = 3600) -> dict:
    cutoff = time.time() - max(1, int(since_s))
    with ctx.memory.lock:
        rows = ctx.memory._con.execute(
            "SELECT turn_id, question, reply, created_at FROM turns "
            "WHERE created_at >= ? ORDER BY created_at DESC",
            (cutoff,),
        ).fetchall()
    return {
        "since_s": since_s,
        "count": len(rows),
        "items": [
            {"turn_id": r[0], "question": r[1], "reply": r[2], "ts": r[3]}
            for r in rows
        ],
    }


# ============================================================================
# SELF / STATE TOOLS
# ============================================================================

@tool(
    "system_status",
    description="CPU, GPU, RAM, disk, thermal snapshot of the Jetson.",
    category="self",
)
def system_status() -> dict:
    info = {"ts": _now()}
    # uptime
    try:
        with open("/proc/uptime") as f:
            info["uptime_s"] = float(f.read().split()[0])
    except Exception:
        pass
    # meminfo
    try:
        m = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                m[k.strip()] = v.strip()
        info["mem"] = {
            "total_kb": int(m["MemTotal"].split()[0]),
            "available_kb": int(m["MemAvailable"].split()[0]),
            "free_kb": int(m["MemFree"].split()[0]),
            "cma_free_kb": int(m.get("CmaFree", "0 kB").split()[0]),
        }
    except Exception:
        pass
    # loadavg
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            info["loadavg"] = [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        pass
    # thermal zones
    temps = []
    for p in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*/temp")):
        try:
            temps.append(int(Path(p).read_text().strip()) / 1000.0)
        except Exception:
            pass
    info["thermal_c"] = temps
    # disk
    try:
        u = shutil.disk_usage("/")
        info["disk"] = {
            "total_gb": round(u.total / 1e9, 1),
            "used_gb": round(u.used / 1e9, 1),
            "free_gb": round(u.free / 1e9, 1),
        }
    except Exception:
        pass
    return info


@tool(
    "get_logs",
    description="Tail journalctl logs for a service.",
    category="self",
    schema={
        "type": "object",
        "properties": {
            "service": {"type": "string"},
            "lines": {"type": "integer"},
        },
    },
)
def get_logs(service: str = "jarvis-vlm", lines: int = 50) -> dict:
    lines = max(1, min(500, int(lines)))
    try:
        r = subprocess.run(
            ["journalctl", "-u", service, "-n", str(lines), "--no-pager"],
            capture_output=True, text=True, timeout=10,
        )
        return {"service": service, "lines": lines, "log": r.stdout[-50000:]}
    except FileNotFoundError:
        raise ToolError("journalctl not available")


@tool(
    "restart_self",
    description="Restart the jarvis-vlm systemd unit. Gated; destructive.",
    category="self",
    safety="dangerous",
    schema={
        "type": "object",
        "properties": {"service": {"type": "string"}},
    },
)
def restart_self(service: str = "jarvis-vlm") -> dict:
    try:
        r = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", service],
            capture_output=True, text=True, timeout=20,
        )
        return {
            "service": service, "rc": r.returncode,
            "stderr": r.stderr[-500:],
        }
    except FileNotFoundError:
        raise ToolError("systemctl not available")


@tool(
    "enable_live_mode",
    description="Turn on continuous live narration mode.",
    category="self",
    needs_ctx=True,
)
def enable_live_mode(ctx) -> dict:
    ok = ctx.live.start()
    return {"ok": ok, "running": ctx.live.is_running()}


@tool(
    "disable_live_mode",
    description="Turn off continuous live narration mode.",
    category="self",
    needs_ctx=True,
)
def disable_live_mode(ctx) -> dict:
    ok = ctx.live.stop()
    return {"ok": ok, "running": ctx.live.is_running()}


@tool(
    "set_persona",
    description="Switch the active persona (focused / inspector / "
                "companion / curator).",
    category="self",
    schema={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    needs_ctx=True,
)
def set_persona(ctx, name: str) -> dict:
    name = (name or "").strip().lower()
    if name not in ctx.presets:
        raise ToolError(f"unknown persona: {name}; "
                        f"available: {list(ctx.presets)}")
    with ctx.settings_lock:
        ctx.settings["preset"] = name
        ctx.settings["system_prompt"] = ctx.presets[name]
    return {"persona": name}


@tool(
    "wake_word_off",
    description="Disable 'Hey Jarvis' wake word for N minutes.",
    category="self",
    schema={
        "type": "object",
        "properties": {"minutes": {"type": "integer"}},
    },
    needs_ctx=True,
)
def wake_word_off(ctx, minutes: int = 10) -> dict:
    minutes = max(1, min(120, int(minutes)))
    ctx.set_wake_enabled(False)
    # auto-re-enable after N minutes
    def _reenable():
        try:
            ctx.set_wake_enabled(True)
        except Exception:
            pass
    threading.Timer(minutes * 60.0, _reenable).start()
    return {"disabled_for_minutes": minutes}


@tool(
    "health_check",
    description="Round-trip a tiny VLM ping and report timing.",
    category="self",
    needs_ctx=True,
)
def health_check(ctx) -> dict:
    snap = _tmp_jpg(ctx, "hc")
    t0 = time.monotonic()
    ctx.capture_frame(snap, allow_reuse=True)
    cap_ms = (time.monotonic() - t0) * 1000.0
    t1 = time.monotonic()
    txt = _vlm_oneshot(ctx, "Reply with exactly: ok", snap, max_seconds=10.0)
    vlm_ms = (time.monotonic() - t1) * 1000.0
    return {
        "capture_ms": round(cap_ms, 1),
        "vlm_ms": round(vlm_ms, 1),
        "reply": txt[:200],
    }


# ============================================================================
# REASONING UTILITIES
# ============================================================================

_SAFE_AST_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge,
    ast.BitAnd: op.and_, ast.BitOr: op.or_, ast.BitXor: op.xor,
    ast.LShift: op.lshift, ast.RShift: op.rshift,
}

_SAFE_FUNCS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "len": len, "pow": pow, "sqrt": math.sqrt, "log": math.log,
    "log2": math.log2, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "factorial": math.factorial, "gcd": math.gcd, "lcm": math.lcm,
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        l = _safe_eval(node.left); r = _safe_eval(node.right)
        return _SAFE_AST_OPS[type(node.op)](l, r)
    if isinstance(node, ast.UnaryOp):
        return _SAFE_AST_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left)
        for op_node, c in zip(node.ops, node.comparators):
            right = _safe_eval(c)
            if not _SAFE_AST_OPS[type(op_node)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        raise ValueError(f"name not allowed: {node.id}")
    if isinstance(node, ast.Call):
        fn = _safe_eval(node.func)
        if not callable(fn) or fn not in _SAFE_FUNCS.values():
            raise ValueError("call to non-whitelisted function")
        args = [_safe_eval(a) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_safe_eval(e) for e in node.elts]
    raise ValueError(f"node not allowed: {type(node).__name__}")


@tool(
    "do_math",
    description="Evaluate an arithmetic expression. Supports + - * / // % **, "
                "abs/min/max/sum/round/sqrt/log/sin/cos/tan, pi/e/tau.",
    category="reason",
    schema={
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"],
    },
)
def do_math(expr: str) -> dict:
    try:
        tree = ast.parse(expr, mode="eval")
        val = _safe_eval(tree)
        return {"expr": expr, "value": val}
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"math error: {e}")


_UNIT_FACTORS = {
    # length -> meters
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
    # mass -> kg
    "g": 0.001, "kg": 1.0, "lb": 0.45359237, "oz": 0.028349523125,
    # time -> s
    "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0,
    # volume -> liters
    "ml": 0.001, "l": 1.0, "gal": 3.785411784, "qt": 0.946352946,
    "cup": 0.2365882365, "tbsp": 0.0147867647825,
    "tsp": 0.00492892159375,
    # data -> bytes
    "b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4,
}
_UNIT_FAMILY = {
    **{u: "len" for u in ("mm", "cm", "m", "km", "in", "ft", "yd", "mi")},
    **{u: "mass" for u in ("g", "kg", "lb", "oz")},
    **{u: "time" for u in ("s", "min", "h", "d")},
    **{u: "vol" for u in ("ml", "l", "gal", "qt", "cup", "tbsp", "tsp")},
    **{u: "data" for u in ("b", "kb", "mb", "gb", "tb")},
}


@tool(
    "convert",
    description="Convert a value between units. Supports length, mass, "
                "time, volume, data sizes, plus C/F temperature.",
    category="reason",
    schema={
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from": {"type": "string"},
            "to": {"type": "string"},
        },
        "required": ["value", "from", "to"],
    },
)
def convert(value: float, **kwargs) -> dict:
    f = (kwargs.get("from") or "").lower().strip()
    t = (kwargs.get("to") or "").lower().strip()
    # temperature special-case
    if f in ("c", "f") and t in ("c", "f"):
        if f == t:
            out = value
        elif f == "c":
            out = value * 9.0 / 5.0 + 32.0
        else:
            out = (value - 32.0) * 5.0 / 9.0
        return {"value": value, "from": f, "to": t, "result": out}
    if f not in _UNIT_FACTORS or t not in _UNIT_FACTORS:
        raise ToolError(f"unknown unit: {f} or {t}")
    if _UNIT_FAMILY[f] != _UNIT_FAMILY[t]:
        raise ToolError(f"incompatible units: {f} ({_UNIT_FAMILY[f]}) "
                        f"vs {t} ({_UNIT_FAMILY[t]})")
    base = value * _UNIT_FACTORS[f]
    out = base / _UNIT_FACTORS[t]
    return {"value": value, "from": f, "to": t, "result": out}


@tool(
    "regex_test",
    description="Test a regex pattern against a string. Returns all matches.",
    category="reason",
    schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "text": {"type": "string"},
            "flags": {"type": "string", "description": "subset of imsx"},
        },
        "required": ["pattern", "text"],
    },
)
def regex_test(pattern: str, text: str, flags: str = "") -> dict:
    f = 0
    for c in (flags or "").lower():
        f |= {"i": re.I, "m": re.M, "s": re.S, "x": re.X}.get(c, 0)
    try:
        rx = re.compile(pattern, f)
    except re.error as e:
        raise ToolError(f"bad regex: {e}")
    matches = [
        {"match": m.group(0), "groups": m.groups(), "span": [m.start(), m.end()]}
        for m in rx.finditer(text)
    ]
    return {"matches": matches, "count": len(matches)}


@tool(
    "json_query",
    description="Read a value from a JSON document using a dotted path "
                "(e.g. 'data.items.0.name'). No jq syntax — just dotted keys "
                "and array indexes.",
    category="reason",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "doc":  {"description": "the JSON document"},
        },
        "required": ["path", "doc"],
    },
)
def json_query(path: str, doc) -> dict:
    cur = doc
    for part in (path or "").split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                raise ToolError(f"bad list index at '{part}'")
        elif isinstance(cur, dict):
            if part not in cur:
                raise ToolError(f"key not found: {part}")
            cur = cur[part]
        else:
            raise ToolError(f"cannot descend into {type(cur).__name__} at '{part}'")
    return {"path": path, "value": cur}


@tool(
    "decode",
    description="Decode a string. Encoding ∈ {base64, url, hex}.",
    category="reason",
    schema={
        "type": "object",
        "properties": {
            "encoding": {"type": "string"},
            "data": {"type": "string"},
        },
        "required": ["encoding", "data"],
    },
)
def decode(encoding: str, data: str) -> dict:
    e = (encoding or "").lower().strip()
    try:
        if e == "base64":
            out = base64.b64decode(data).decode("utf-8", errors="replace")
        elif e == "url":
            out = urllib.parse.unquote(data)
        elif e == "hex":
            out = binascii.unhexlify(data).decode("utf-8", errors="replace")
        else:
            raise ToolError(f"unknown encoding: {encoding}")
    except (binascii.Error, ValueError) as ex:
        raise ToolError(f"decode error: {ex}")
    return {"encoding": e, "result": out}


@tool(
    "random_choice",
    description="Pick one item from a list at random.",
    category="reason",
    schema={
        "type": "object",
        "properties": {"options": {"type": "array"}},
        "required": ["options"],
    },
)
def random_choice(options: list) -> dict:
    if not options:
        raise ToolError("options must be non-empty")
    return {"choice": random.choice(options), "from": options}


@tool(
    "coin_flip",
    description="Heads or tails.",
    category="reason",
)
def coin_flip() -> dict:
    return {"result": random.choice(["heads", "tails"])}


@tool(
    "roll",
    description="Roll dice in NdM notation (e.g. '3d6', '1d20+5').",
    category="reason",
    schema={
        "type": "object",
        "properties": {"spec": {"type": "string"}},
        "required": ["spec"],
    },
)
def roll(spec: str) -> dict:
    m = re.match(r"\s*(\d*)d(\d+)\s*([+-]\s*\d+)?\s*$", spec or "", re.I)
    if not m:
        raise ToolError("spec must look like '3d6' or '1d20+5'")
    n = int(m.group(1) or "1"); sides = int(m.group(2))
    mod = int((m.group(3) or "0").replace(" ", ""))
    if not (1 <= n <= 100 and 2 <= sides <= 1000):
        raise ToolError("ridiculous dice; keep it sane")
    rolls = [random.randint(1, sides) for _ in range(n)]
    return {"spec": spec, "rolls": rolls, "total": sum(rolls) + mod, "mod": mod}


@tool(
    "password",
    description="Generate a random password.",
    category="reason",
    schema={
        "type": "object",
        "properties": {
            "length": {"type": "integer"},
            "style":  {"type": "string",
                       "description": "alpha | alphanum | symbols"},
        },
    },
)
def password(length: int = 20, style: str = "symbols") -> dict:
    length = max(6, min(128, int(length)))
    if style == "alpha":
        alphabet = string.ascii_letters
    elif style == "alphanum":
        alphabet = string.ascii_letters + string.digits
    else:
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}"
    pw = "".join(secrets.choice(alphabet) for _ in range(length))
    return {"password": pw, "length": length, "style": style}


@tool(
    "uuid",
    description="Generate a UUID4.",
    category="reason",
)
def uuid_tool() -> dict:
    return {"uuid": str(uuid_mod.uuid4())}


@tool(
    "now",
    description="Current time, optionally in a named timezone (IANA, e.g. "
                "'America/New_York').",
    category="reason",
    schema={
        "type": "object",
        "properties": {"tz": {"type": "string"}},
    },
)
def now(tz: str = "") -> dict:
    import datetime as dt
    if tz:
        try:
            z = zoneinfo.ZoneInfo(tz)
        except Exception:
            raise ToolError(f"unknown tz: {tz}")
        n = dt.datetime.now(z)
    else:
        n = dt.datetime.now().astimezone()
    return {
        "iso": n.isoformat(),
        "epoch": n.timestamp(),
        "tz": str(n.tzinfo),
        "human": n.strftime("%A %Y-%m-%d %H:%M:%S %Z"),
    }


@tool(
    "cron_next",
    description="Compute the next firing of a cron expression. Stub — "
                "supports only '@hourly', '@daily', '@weekly'.",
    category="reason",
    schema={
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"],
    },
)
def cron_next(expr: str) -> dict:
    e = (expr or "").strip().lower()
    now = time.time()
    if e == "@hourly":
        nxt = now - (now % 3600) + 3600
    elif e == "@daily":
        nxt = now - (now % 86400) + 86400
    elif e == "@weekly":
        nxt = now - (now % (86400 * 7)) + 86400 * 7
    else:
        raise ToolError("only @hourly/@daily/@weekly supported "
                        "until croniter ships")
    return {"expr": expr, "next_epoch": nxt,
            "next_human": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(nxt))}


# ============================================================================
# PRODUCTIVITY TOOLS (notes / todos / reminders / bookmarks / journal)
# ============================================================================

def _mem(ctx): return ctx.memory


@tool(
    "note_create",
    description="Create a note.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "body":  {"type": "string"},
            "tags":  {"type": "string"},
        },
        "required": ["title", "body"],
    },
    needs_ctx=True,
)
def note_create(ctx, title: str, body: str, tags: str = "") -> dict:
    now = _now()
    with _mem(ctx).lock:
        cur = _mem(ctx)._con.execute(
            "INSERT INTO notes (title, body, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?)", (title, body, tags, now, now),
        )
        nid = cur.lastrowid
    return {"id": nid, "title": title}


@tool(
    "note_append",
    description="Append to an existing note (by id).",
    category="productivity",
    schema={
        "type": "object",
        "properties": {
            "id":   {"type": "integer"},
            "body": {"type": "string"},
        },
        "required": ["id", "body"],
    },
    needs_ctx=True,
)
def note_append(ctx, id: int, body: str) -> dict:
    with _mem(ctx).lock:
        r = _mem(ctx)._con.execute(
            "SELECT body FROM notes WHERE id=?", (int(id),)
        ).fetchone()
        if r is None:
            raise ToolError(f"note id {id} not found")
        new = (r[0] or "") + "\n" + body
        _mem(ctx)._con.execute(
            "UPDATE notes SET body=?, updated_at=? WHERE id=?",
            (new, _now(), int(id)),
        )
    return {"id": int(id), "appended_chars": len(body)}


@tool(
    "note_search",
    description="Search notes by substring in title or body.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    needs_ctx=True,
)
def note_search(ctx, query: str) -> dict:
    pat = f"%{query}%"
    with _mem(ctx).lock:
        rows = _mem(ctx)._con.execute(
            "SELECT id, title, body, tags, created_at FROM notes "
            "WHERE title LIKE ? OR body LIKE ? ORDER BY updated_at DESC LIMIT 50",
            (pat, pat),
        ).fetchall()
    return {"count": len(rows),
            "items": [{"id": r[0], "title": r[1], "body": r[2],
                       "tags": r[3], "ts": r[4]} for r in rows]}


@tool(
    "note_list",
    description="List recent notes.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {"limit": {"type": "integer"}},
    },
    needs_ctx=True,
)
def note_list(ctx, limit: int = 20) -> dict:
    limit = max(1, min(200, int(limit)))
    with _mem(ctx).lock:
        rows = _mem(ctx)._con.execute(
            "SELECT id, title, tags, created_at FROM notes "
            "ORDER BY updated_at DESC LIMIT ?", (limit,),
        ).fetchall()
    return {"count": len(rows),
            "items": [{"id": r[0], "title": r[1], "tags": r[2], "ts": r[3]}
                      for r in rows]}


@tool(
    "todo_add",
    description="Add a todo. Optional `due` is a natural-language time.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "due":  {"type": "string"},
        },
        "required": ["text"],
    },
    needs_ctx=True,
)
def todo_add(ctx, text: str, due: str = "") -> dict:
    due_at = _parse_when(due) if due else None
    with _mem(ctx).lock:
        cur = _mem(ctx)._con.execute(
            "INSERT INTO todos (text, due_at, created_at) VALUES (?,?,?)",
            (text, due_at, _now()),
        )
        tid = cur.lastrowid
    return {"id": tid, "text": text, "due_at": due_at}


@tool(
    "todo_done",
    description="Mark a todo done by id.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    },
    needs_ctx=True,
)
def todo_done(ctx, id: int) -> dict:
    now = _now()
    with _mem(ctx).lock:
        r = _mem(ctx)._con.execute(
            "UPDATE todos SET done=1, done_at=? WHERE id=?",
            (now, int(id)),
        )
    return {"id": int(id), "ok": True}


@tool(
    "todo_list",
    description="List todos. filter ∈ open | done | all.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {"filter": {"type": "string"}},
    },
    needs_ctx=True,
)
def todo_list(ctx, **kwargs) -> dict:
    f = (kwargs.get("filter") or "open").lower()
    sql = "SELECT id, text, due_at, done, created_at FROM todos"
    if f == "open":
        sql += " WHERE done=0"
    elif f == "done":
        sql += " WHERE done=1"
    sql += " ORDER BY COALESCE(due_at, created_at) LIMIT 200"
    with _mem(ctx).lock:
        rows = _mem(ctx)._con.execute(sql).fetchall()
    return {"filter": f,
            "items": [{"id": r[0], "text": r[1], "due_at": r[2],
                       "done": bool(r[3]), "ts": r[4]} for r in rows]}


@tool(
    "todo_due_today",
    description="Open todos with a due time today.",
    category="productivity",
    needs_ctx=True,
)
def todo_due_today(ctx) -> dict:
    day_start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    day_end = day_start + 86400
    with _mem(ctx).lock:
        rows = _mem(ctx)._con.execute(
            "SELECT id, text, due_at FROM todos "
            "WHERE done=0 AND due_at >= ? AND due_at < ? ORDER BY due_at",
            (day_start, day_end),
        ).fetchall()
    return {"items": [{"id": r[0], "text": r[1], "due_at": r[2]} for r in rows]}


@tool(
    "reminder_set",
    description="Set a reminder. `when` is natural language "
                "('in 10 minutes', '14:30', '9am').",
    category="productivity",
    schema={
        "type": "object",
        "properties": {
            "when": {"type": "string"},
            "what": {"type": "string"},
        },
        "required": ["when", "what"],
    },
    needs_ctx=True,
)
def reminder_set(ctx, when: str, what: str) -> dict:
    fire_at = _parse_when(when)
    with _mem(ctx).lock:
        cur = _mem(ctx)._con.execute(
            "INSERT INTO reminders (text, fire_at, created_at) VALUES (?,?,?)",
            (what, fire_at, _now()),
        )
        rid = cur.lastrowid
    return {
        "id": rid, "what": what, "fire_at": fire_at,
        "fire_human": time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(fire_at)),
        "in_s": round(fire_at - _now(), 1),
    }


@tool(
    "reminder_list",
    description="List pending and recently-fired reminders.",
    category="productivity",
    needs_ctx=True,
)
def reminder_list(ctx) -> dict:
    with _mem(ctx).lock:
        rows = _mem(ctx)._con.execute(
            "SELECT id, text, fire_at, fired, fired_at FROM reminders "
            "ORDER BY fire_at DESC LIMIT 100",
        ).fetchall()
    return {"items": [{"id": r[0], "text": r[1], "fire_at": r[2],
                       "fired": bool(r[3]), "fired_at": r[4]} for r in rows]}


@tool(
    "bookmark",
    description="Save a URL bookmark with optional tags.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {
            "url":   {"type": "string"},
            "title": {"type": "string"},
            "tags":  {"type": "string"},
        },
        "required": ["url"],
    },
    needs_ctx=True,
)
def bookmark(ctx, url: str, title: str = "", tags: str = "") -> dict:
    with _mem(ctx).lock:
        cur = _mem(ctx)._con.execute(
            "INSERT INTO bookmarks (url, title, tags, created_at) "
            "VALUES (?,?,?,?)",
            (url, title, tags, _now()),
        )
        bid = cur.lastrowid
    return {"id": bid, "url": url}


@tool(
    "bookmarks",
    description="List bookmarks, optionally filtered by tag substring.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {"tag": {"type": "string"}},
    },
    needs_ctx=True,
)
def bookmarks(ctx, tag: str = "") -> dict:
    with _mem(ctx).lock:
        if tag:
            rows = _mem(ctx)._con.execute(
                "SELECT id, url, title, tags, created_at FROM bookmarks "
                "WHERE tags LIKE ? ORDER BY created_at DESC LIMIT 200",
                (f"%{tag}%",),
            ).fetchall()
        else:
            rows = _mem(ctx)._con.execute(
                "SELECT id, url, title, tags, created_at FROM bookmarks "
                "ORDER BY created_at DESC LIMIT 200",
            ).fetchall()
    return {"items": [{"id": r[0], "url": r[1], "title": r[2],
                       "tags": r[3], "ts": r[4]} for r in rows]}


@tool(
    "journal_entry",
    description="Add a journal entry for today.",
    category="productivity",
    schema={
        "type": "object",
        "properties": {"body": {"type": "string"}},
        "required": ["body"],
    },
    needs_ctx=True,
)
def journal_entry(ctx, body: str) -> dict:
    day = time.strftime("%Y-%m-%d", time.localtime())
    with _mem(ctx).lock:
        cur = _mem(ctx)._con.execute(
            "INSERT INTO journal (body, day, created_at) VALUES (?,?,?)",
            (body, day, _now()),
        )
        jid = cur.lastrowid
    return {"id": jid, "day": day}


# ============================================================================
# WEB TOOLS (no API keys)
# ============================================================================

@tool(
    "wikipedia",
    description="Search Wikipedia, return summary of the top result.",
    category="web",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
def wikipedia(query: str) -> dict:
    # 1) search to resolve title
    sr = HTTP.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search",
                "srsearch": query, "format": "json", "srlimit": 1},
    )
    sr.raise_for_status()
    hits = sr.json().get("query", {}).get("search", [])
    if not hits:
        return {"query": query, "found": False}
    title = hits[0]["title"]
    # 2) summary endpoint
    sm = HTTP.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title.replace(" ", "_"))
    )
    if sm.status_code != 200:
        return {"query": query, "title": title, "found": True}
    j = sm.json()
    return {
        "query": query, "title": j.get("title"),
        "extract": j.get("extract"),
        "url": j.get("content_urls", {}).get("desktop", {}).get("page"),
    }


@tool(
    "duckduckgo",
    description="DuckDuckGo Instant Answer API.",
    category="web",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
def duckduckgo(query: str) -> dict:
    r = HTTP.get("https://api.duckduckgo.com/",
                 params={"q": query, "format": "json", "no_html": "1"})
    r.raise_for_status()
    j = r.json()
    return {
        "abstract": j.get("AbstractText"),
        "abstract_source": j.get("AbstractSource"),
        "abstract_url": j.get("AbstractURL"),
        "answer": j.get("Answer"),
        "answer_type": j.get("AnswerType"),
        "related": [
            {"text": t.get("Text"), "url": t.get("FirstURL")}
            for t in j.get("RelatedTopics", []) if t.get("Text")
        ][:5],
    }


# ----- device location (the Orin has no GPS — derive from the public IP) -------
_DEVICE_LOC: dict = {"data": None, "ts": 0.0}
_DEVICE_LOC_TTL = 6 * 3600


def get_device_location(force: bool = False) -> dict:
    """City/region/country/lat/lon/timezone from the public IP, cached ~6h."""
    now = time.time()
    if (not force and _DEVICE_LOC["data"]
            and now - _DEVICE_LOC["ts"] < _DEVICE_LOC_TTL):
        return _DEVICE_LOC["data"]
    try:
        r = HTTP.get("https://ipinfo.io/json")
        r.raise_for_status()
        d = r.json()
        loc = (d.get("loc") or "").split(",")
        data = {
            "city": d.get("city"), "region": d.get("region"),
            "country": d.get("country"),
            "lat": float(loc[0]) if len(loc) == 2 else None,
            "lon": float(loc[1]) if len(loc) == 2 else None,
            "timezone": d.get("timezone"),
        }
        _DEVICE_LOC["data"] = data
        _DEVICE_LOC["ts"] = now
        return data
    except Exception:  # noqa: BLE001
        return _DEVICE_LOC["data"] or {}


def device_location_str() -> str:
    d = get_device_location()
    return ", ".join(p for p in (d.get("city"), d.get("region"),
                                 d.get("country")) if p) if d else ""


@tool(
    "where_am_i",
    description="The device's own location (city, region, country, lat/lon, "
                "timezone) from its public IP. Use for 'where am I' and as the "
                "default place for weather/time/local questions.",
    category="web",
    schema={"type": "object", "properties": {}},
)
def where_am_i() -> dict:
    return get_device_location() or {"error": "location unavailable"}


@tool(
    "weather",
    description="Current weather. Omit location to use the device's own location.",
    category="web",
    schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
    },
)
def weather(location: str = "") -> dict:
    location = (location or "").strip() or device_location_str()
    r = HTTP.get(f"https://wttr.in/{urllib.parse.quote(location)}",
                 params={"format": "j1"})
    r.raise_for_status()
    j = r.json()
    cur = (j.get("current_condition") or [{}])[0]
    return {
        "location": location,
        "temp_c": cur.get("temp_C"),
        "temp_f": cur.get("temp_F"),
        "feels_c": cur.get("FeelsLikeC"),
        "humidity": cur.get("humidity"),
        "desc": (cur.get("weatherDesc") or [{}])[0].get("value"),
        "wind_kph": cur.get("windspeedKmph"),
        "wind_dir": cur.get("winddir16Point"),
        "observed_at": cur.get("observation_time"),
    }


@tool(
    "forecast",
    description="Multi-day forecast via wttr.in.",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "days":     {"type": "integer"},
        },
        "required": ["location"],
    },
)
def forecast(location: str, days: int = 3) -> dict:
    days = max(1, min(3, int(days)))
    r = HTTP.get(f"https://wttr.in/{urllib.parse.quote(location)}",
                 params={"format": "j1"})
    r.raise_for_status()
    j = r.json()
    out = []
    for d in (j.get("weather") or [])[:days]:
        out.append({
            "date": d.get("date"),
            "max_c": d.get("maxtempC"),
            "min_c": d.get("mintempC"),
            "max_f": d.get("maxtempF"),
            "min_f": d.get("mintempF"),
            "sun_hours": d.get("sunHour"),
            "desc": (d.get("hourly", [{}])[len(d.get("hourly", []))//2]
                     .get("weatherDesc", [{}])[0].get("value")),
        })
    return {"location": location, "days": out}


# ============================================================================
# JARVIS CAPABILITIES — information, utility, briefing, diagnostics
# (all free / no API keys; location comes from get_device_location())
# ============================================================================
_LANG_CODES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "chinese": "zh", "mandarin": "zh", "japanese": "ja", "korean": "ko",
    "arabic": "ar", "hindi": "hi", "polish": "pl", "swedish": "sv",
    "norwegian": "no", "danish": "da", "finnish": "fi", "greek": "el",
    "turkish": "tr", "hebrew": "he", "thai": "th", "vietnamese": "vi",
    "ukrainian": "uk", "czech": "cs", "romanian": "ro", "hungarian": "hu",
}
_CRYPTO_IDS = {
    "btc": "bitcoin", "bitcoin": "bitcoin", "eth": "ethereum", "ethereum": "ethereum",
    "sol": "solana", "solana": "solana", "doge": "dogecoin", "dogecoin": "dogecoin",
    "ada": "cardano", "cardano": "cardano", "xrp": "ripple", "ripple": "ripple",
    "ltc": "litecoin", "litecoin": "litecoin", "bnb": "binancecoin",
    "matic": "matic-network", "dot": "polkadot", "polkadot": "polkadot",
    "avax": "avalanche-2", "link": "chainlink", "chainlink": "chainlink",
}
_OSM_AMENITY = {
    "coffee": "cafe", "cafe": "cafe", "coffee shop": "cafe",
    "gas": "fuel", "gas station": "fuel", "petrol": "fuel", "fuel": "fuel",
    "pharmacy": "pharmacy", "drugstore": "pharmacy",
    "restaurant": "restaurant", "food": "restaurant", "bar": "bar", "pub": "pub",
    "bank": "bank", "atm": "atm", "hospital": "hospital", "clinic": "clinic",
    "supermarket": "supermarket", "grocery": "supermarket", "school": "school",
    "parking": "parking", "hotel": "hotel", "library": "library",
    "fast food": "fast_food", "fast_food": "fast_food",
}


@tool(
    "sun_times",
    description="Sunrise, sunset, and daylight hours for a place (default: here).",
    category="web",
    schema={"type": "object", "properties": {"location": {"type": "string"}}},
)
def sun_times(location: str = "") -> dict:
    lat = lon = None
    place = ""
    if location.strip():
        g = HTTP.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": location, "count": 1})
        res = (g.json().get("results") or [])
        if res:
            lat, lon, place = res[0]["latitude"], res[0]["longitude"], res[0]["name"]
    else:
        d = get_device_location()
        lat, lon, place = d.get("lat"), d.get("lon"), device_location_str()
    if lat is None:
        return {"error": "location unavailable"}
    r = HTTP.get("https://api.open-meteo.com/v1/forecast",
                 params={"latitude": lat, "longitude": lon,
                         "daily": "sunrise,sunset,daylight_duration",
                         "timezone": "auto", "forecast_days": 1})
    dd = r.json().get("daily", {})
    sr = (dd.get("sunrise") or [None])[0]
    ss = (dd.get("sunset") or [None])[0]
    dl = (dd.get("daylight_duration") or [None])[0]
    return {"location": place, "sunrise": sr, "sunset": ss,
            "daylight_hours": round(dl / 3600, 1) if dl else None}


@tool(
    "news",
    description="Top news headlines. Optional topic (e.g. 'technology', 'world').",
    category="web",
    schema={"type": "object", "properties": {
        "topic": {"type": "string"}, "n": {"type": "integer"}}},
)
def news(topic: str = "", n: int = 5) -> dict:
    import xml.etree.ElementTree as ET
    n = max(1, min(10, int(n)))
    if topic.strip():
        url = ("https://news.google.com/rss/search?q="
               + urllib.parse.quote(topic) + "&hl=en-US&gl=US&ceid=US:en")
    else:
        url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
    r = HTTP.get(url, headers={"User-Agent": "Mozilla/5.0"})
    root = ET.fromstring(r.text)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if title:
            items.append(title)
        if len(items) >= n:
            break
    return {"topic": topic or "top stories", "headlines": items}


@tool(
    "define",
    description="Dictionary definition of an English word.",
    category="web",
    schema={"type": "object", "properties": {"word": {"type": "string"}},
            "required": ["word"]},
)
def define(word: str) -> dict:
    r = HTTP.get("https://api.dictionaryapi.dev/api/v2/entries/en/"
                 + urllib.parse.quote(word.strip()))
    if r.status_code != 200:
        return {"word": word, "error": "no definition found"}
    j = r.json()[0]
    meanings = []
    for m in j.get("meanings", [])[:3]:
        defs = [d.get("definition") for d in m.get("definitions", [])[:2]]
        meanings.append({"part_of_speech": m.get("partOfSpeech"), "definitions": defs})
    return {"word": j.get("word"), "phonetic": j.get("phonetic"), "meanings": meanings}


@tool(
    "translate",
    description="Translate text into a target language (e.g. to='Spanish' or 'fr').",
    category="web",
    schema={"type": "object", "properties": {
        "text": {"type": "string"}, "to": {"type": "string"}},
        "required": ["text", "to"]},
)
def translate(text: str, to: str) -> dict:
    tl = _LANG_CODES.get(to.strip().lower(), to.strip().lower()[:2])
    r = HTTP.get("https://translate.googleapis.com/translate_a/single",
                 params={"client": "gtx", "sl": "auto", "tl": tl, "dt": "t", "q": text})
    j = r.json()
    translated = "".join(seg[0] for seg in j[0] if seg and seg[0])
    src = j[2] if len(j) > 2 else "auto"
    return {"original": text, "translated": translated, "to": to,
            "detected_source": src}


@tool(
    "stock_price",
    description="Current stock/ETF price by ticker (e.g. AAPL, TSLA, SPY).",
    category="web",
    schema={"type": "object", "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]},
)
def stock_price(symbol: str) -> dict:
    sym = symbol.strip().upper()
    r = HTTP.get("https://query1.finance.yahoo.com/v8/finance/chart/"
                 + urllib.parse.quote(sym), headers={"User-Agent": "Mozilla/5.0"})
    res = (r.json().get("chart", {}).get("result") or [{}])[0]
    meta = res.get("meta", {}) if res else {}
    price = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    chg = (price - prev) if (price is not None and prev) else None
    return {"symbol": sym, "price": price, "currency": meta.get("currency"),
            "change": round(chg, 2) if chg is not None else None,
            "change_pct": round(100 * chg / prev, 2) if chg is not None and prev else None}


@tool(
    "crypto_price",
    description="Current cryptocurrency price in USD (e.g. bitcoin, ethereum, BTC).",
    category="web",
    schema={"type": "object", "properties": {"coin": {"type": "string"}},
            "required": ["coin"]},
)
def crypto_price(coin: str) -> dict:
    c = coin.strip().lower()
    cid = _CRYPTO_IDS.get(c, c)
    r = HTTP.get("https://api.coingecko.com/api/v3/simple/price",
                 params={"ids": cid, "vs_currencies": "usd",
                         "include_24hr_change": "true"})
    j = r.json()
    if cid not in j:
        s = HTTP.get("https://api.coingecko.com/api/v3/search", params={"query": c})
        coins = s.json().get("coins", [])
        if coins:
            cid = coins[0]["id"]
            r = HTTP.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": cid, "vs_currencies": "usd",
                                 "include_24hr_change": "true"})
            j = r.json()
    d = j.get(cid, {})
    return {"coin": cid, "usd": d.get("usd"),
            "change_24h_pct": round(d.get("usd_24h_change", 0), 2) if d else None}


@tool(
    "nearby_places",
    description="Find places near you by category (cafe, pharmacy, gas, restaurant…).",
    category="web",
    schema={"type": "object", "properties": {
        "category": {"type": "string"}, "radius_m": {"type": "integer"}},
        "required": ["category"]},
)
def nearby_places(category: str, radius_m: int = 1500) -> dict:
    loc = get_device_location()
    lat, lon = loc.get("lat"), loc.get("lon")
    if lat is None:
        return {"error": "location unavailable"}
    radius_m = max(200, min(5000, int(radius_m)))
    amenity = _OSM_AMENITY.get(category.strip().lower(), category.strip().lower())
    q = (f"[out:json][timeout:20];"
         f"(node(around:{radius_m},{lat},{lon})[amenity={amenity}];);out 8;")
    body = ("data=" + urllib.parse.quote(q)).encode()
    hdrs = {"Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "jarvis-lab/1.0"}
    els = None
    for url in ("https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter"):
        try:
            r = HTTP.post(url, content=body, headers=hdrs, timeout=25)
            els = r.json().get("elements", [])
            break
        except Exception:  # noqa: BLE001
            continue
    if els is None:
        return {"error": "places lookup unavailable (Overpass)"}
    places = []
    for e in els:
        t = e.get("tags", {})
        if t.get("name"):
            addr = " ".join(filter(None, [t.get("addr:housenumber"),
                                          t.get("addr:street")]))
            places.append({"name": t.get("name"), "type": t.get("amenity"),
                           "address": addr or None})
    return {"category": category, "near": device_location_str(),
            "places": places[:8]}


@tool(
    "network_speed",
    description="Quick internet speed test (download Mbps + latency) from the device.",
    category="self",
    schema={"type": "object", "properties": {}},
)
def network_speed() -> dict:
    t0 = time.monotonic()
    HTTP.get("https://speed.cloudflare.com/__down", params={"bytes": 1}, timeout=10)
    ping_ms = round((time.monotonic() - t0) * 1000, 1)
    nbytes = 8_000_000
    t0 = time.monotonic()
    r = HTTP.get("https://speed.cloudflare.com/__down",
                 params={"bytes": nbytes}, timeout=30)
    got = len(r.content)
    dt = time.monotonic() - t0
    mbps = round((got * 8) / dt / 1e6, 1) if dt > 0 else None
    return {"download_mbps": mbps, "latency_ms": ping_ms,
            "mb_downloaded": round(got / 1e6, 1), "seconds": round(dt, 2)}


@tool(
    "research",
    description="Deep-dive a topic — compiles an encyclopedia summary plus live web "
                "results into one briefing. Use for 'pull up everything on X'.",
    category="web",
    schema={"type": "object", "properties": {"topic": {"type": "string"}},
            "required": ["topic"]},
)
def research(topic: str) -> dict:
    out = {"topic": topic}
    try:
        out["summary"] = TOOLS.call("wikipedia", {"query": topic}).get("result")
    except Exception:  # noqa: BLE001
        pass
    try:
        out["web"] = TOOLS.call("web_search", {"query": topic, "n": 5}).get("result")
    except Exception:  # noqa: BLE001
        pass
    return out


@tool(
    "briefing",
    description="A spoken daily briefing: date/time, local weather, top headlines, "
                "and today's reminders. Use for 'good morning' / 'brief me'.",
    category="web",
    schema={"type": "object", "properties": {}},
    needs_ctx=True,
)
def briefing(ctx) -> dict:
    out = {}
    try:
        out["now"] = TOOLS.call("now", {}).get("result")
    except Exception:  # noqa: BLE001
        pass
    out["location"] = device_location_str()
    try:
        out["weather"] = TOOLS.call("weather", {}).get("result")
    except Exception:  # noqa: BLE001
        pass
    try:
        out["headlines"] = (TOOLS.call("news", {"n": 3}).get("result") or {}).get("headlines")
    except Exception:  # noqa: BLE001
        pass
    try:
        out["reminders"] = TOOLS.call("reminder_list", {}).get("result")
    except Exception:  # noqa: BLE001
        pass
    return out


@tool(
    "show_panel",
    description="Open a panel in the Intel sidebar so the user can SEE it. "
                "panel ∈ talk | seen (what the camera has seen) | memory | "
                "tools | activity | entities | live | pinned | timeline | "
                "cloud (3D point cloud) | graph (entity link chart). Call this "
                "whenever the user asks to see / show / pull up / open something.",
    category="self",
    schema={"type": "object", "properties": {"panel": {"type": "string"}},
            "required": ["panel"]},
)
def show_panel(panel: str) -> dict:
    # The effect is client-side: the browser reads this off the agent step stream
    # and opens the panel. The server just acknowledges the request.
    return {"ok": True, "panel": (panel or "").strip().lower()}


@tool(
    "status_report",
    description="A systems status report — power, temperatures, memory, disk. "
                "JARVIS-style diagnostics ('how are you doing', 'status report').",
    category="self",
    schema={"type": "object", "properties": {}},
)
def status_report() -> dict:
    return TOOLS.call("system_status", {}).get("result") or {"error": "unavailable"}


@tool(
    "timer",
    description="Set a countdown timer; it chimes and announces when done. "
                "Give minutes and/or seconds.",
    category="productivity",
    schema={"type": "object", "properties": {
        "minutes": {"type": "number"}, "seconds": {"type": "number"},
        "label": {"type": "string"}}},
    needs_ctx=True,
)
def timer(ctx, minutes: float = 0, seconds: float = 0, label: str = "") -> dict:
    total = int((minutes or 0) * 60 + (seconds or 0))
    if total <= 0:
        return {"error": "specify minutes and/or seconds"}
    mins, secs = total // 60, total % 60
    dur = " ".join(p for p in (
        f"{mins} minute{'s' if mins != 1 else ''}" if mins else "",
        f"{secs} second{'s' if secs != 1 else ''}" if secs else "") if p)
    spoken = (label.strip() or f"Your {dur} timer") + " is up"   # said when it FIRES
    fire_at = _now() + total
    with _mem(ctx).lock:
        cur = _mem(ctx)._con.execute(
            "INSERT INTO reminders (text, fire_at, created_at) VALUES (?,?,?)",
            (spoken, fire_at, _now()),
        )
        rid = cur.lastrowid
    # result the agent confirms with — make clear the timer is SET (not done)
    return {"timer_set_for": dur or f"{total} seconds", "in_seconds": total,
            "fires_at": time.strftime("%H:%M:%S", time.localtime(fire_at)), "id": rid}


@tool(
    "ocr_translate",
    description="Read text in the camera view and translate it (e.g. to='English'). "
                "For foreign signs, menus, labels.",
    category="vision",
    schema={"type": "object", "properties": {"to": {"type": "string"}},
            "required": ["to"]},
    needs_ctx=True,
)
def ocr_translate(ctx, to: str = "English") -> dict:
    r = TOOLS.call("read_all_text", {}).get("result") or {}
    text = (r.get("text") or "").strip()
    if not text or text.upper() == "NO_TEXT":
        return {"error": "no readable text in view"}
    tr = TOOLS.call("translate", {"text": text, "to": to}).get("result") or {}
    return {"original": text, "translated": tr.get("translated"), "to": to}


@tool(
    "geocode",
    description="Forward geocode an address via Nominatim.",
    category="web",
    schema={
        "type": "object",
        "properties": {"address": {"type": "string"}},
        "required": ["address"],
    },
)
def geocode(address: str) -> dict:
    r = HTTP.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": "jarvis-lab/1.0"},
    )
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return {"address": address, "found": False}
    o = arr[0]
    return {
        "address": address, "found": True,
        "lat": float(o["lat"]), "lon": float(o["lon"]),
        "display": o.get("display_name"), "type": o.get("type"),
    }


@tool(
    "reverse_geocode",
    description="Reverse geocode lat/lon via Nominatim.",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "lat": {"type": "number"}, "lon": {"type": "number"},
        },
        "required": ["lat", "lon"],
    },
)
def reverse_geocode(lat: float, lon: float) -> dict:
    r = HTTP.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "json"},
        headers={"User-Agent": "jarvis-lab/1.0"},
    )
    r.raise_for_status()
    j = r.json()
    return {
        "lat": lat, "lon": lon,
        "display": j.get("display_name"),
        "address": j.get("address", {}),
    }


@tool(
    "arxiv",
    description="Search arXiv.",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
)
def arxiv(query: str, limit: int = 5) -> dict:
    limit = max(1, min(20, int(limit)))
    r = HTTP.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": "all:" + query,
                "start": 0, "max_results": limit},
    )
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    items = []
    for e in root.findall("a:entry", ns):
        title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
        link = ""
        for ln in e.findall("a:link", ns):
            if ln.get("type") == "application/pdf":
                link = ln.get("href"); break
        if not link:
            link = e.findtext("a:id", default="", namespaces=ns) or ""
        items.append({"title": title, "summary": summary[:500], "url": link})
    return {"query": query, "items": items}


@tool(
    "hn_top",
    description="Top stories from Hacker News.",
    category="web",
    schema={
        "type": "object",
        "properties": {"limit": {"type": "integer"}},
    },
)
def hn_top(limit: int = 10) -> dict:
    limit = max(1, min(30, int(limit)))
    r = HTTP.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    r.raise_for_status()
    ids = r.json()[:limit]
    items = []
    for i in ids:
        ir = HTTP.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json")
        if ir.status_code != 200:
            continue
        d = ir.json() or {}
        items.append({"id": i, "title": d.get("title"),
                      "url": d.get("url"), "score": d.get("score"),
                      "by": d.get("by")})
    return {"items": items}


@tool(
    "hn_search",
    description="Search Hacker News via Algolia.",
    category="web",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
def hn_search(query: str) -> dict:
    r = HTTP.get("https://hn.algolia.com/api/v1/search",
                 params={"query": query, "hitsPerPage": 10})
    r.raise_for_status()
    j = r.json()
    return {
        "query": query,
        "hits": [
            {"title": h.get("title") or h.get("story_title"),
             "url": h.get("url") or h.get("story_url"),
             "points": h.get("points"),
             "by": h.get("author")}
            for h in j.get("hits", [])
        ],
    }


@tool(
    "rss_fetch",
    description="Fetch an RSS or Atom feed; return latest items.",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["url"],
    },
)
def rss_fetch(url: str, limit: int = 10) -> dict:
    limit = max(1, min(50, int(limit)))
    r = HTTP.get(url)
    r.raise_for_status()
    items = []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        raise ToolError(f"feed parse error: {e}")
    # RSS 2.0
    for it in root.findall(".//item")[:limit]:
        items.append({
            "title": (it.findtext("title") or "").strip(),
            "link":  (it.findtext("link") or "").strip(),
            "date":  (it.findtext("pubDate") or "").strip(),
            "summary": (it.findtext("description") or "").strip()[:500],
        })
    # Atom fallback
    if not items:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for it in root.findall(".//a:entry", ns)[:limit]:
            link = ""
            ln = it.find("a:link", ns)
            if ln is not None:
                link = ln.get("href") or ""
            items.append({
                "title": (it.findtext("a:title", default="", namespaces=ns) or "").strip(),
                "link": link,
                "date": (it.findtext("a:updated", default="", namespaces=ns) or "").strip(),
                "summary": (it.findtext("a:summary", default="", namespaces=ns) or "").strip()[:500],
            })
    return {"url": url, "items": items}


@tool(
    "github_repo",
    description="Fetch metadata about a GitHub repo (anonymous).",
    category="web",
    schema={
        "type": "object",
        "properties": {"owner_repo": {"type": "string"}},
        "required": ["owner_repo"],
    },
)
def github_repo(owner_repo: str) -> dict:
    r = HTTP.get(f"https://api.github.com/repos/{owner_repo}")
    if r.status_code == 404:
        return {"owner_repo": owner_repo, "found": False}
    r.raise_for_status()
    j = r.json()
    return {
        "owner_repo": owner_repo, "found": True,
        "description": j.get("description"),
        "stars": j.get("stargazers_count"),
        "forks": j.get("forks_count"),
        "open_issues": j.get("open_issues_count"),
        "language": j.get("language"),
        "default_branch": j.get("default_branch"),
        "url": j.get("html_url"),
    }


@tool(
    "time_in",
    description="Current time in a named IANA timezone.",
    category="web",
    schema={
        "type": "object",
        "properties": {"tz": {"type": "string"}},
        "required": ["tz"],
    },
)
def time_in(tz: str) -> dict:
    return now(tz=tz)


@tool(
    "currency_rate",
    description="Spot exchange rate via open.er-api.com (no key).",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "from_ccy": {"type": "string"},
            "to_ccy":   {"type": "string"},
        },
        "required": ["from_ccy", "to_ccy"],
    },
)
def currency_rate(from_ccy: str, to_ccy: str) -> dict:
    f = from_ccy.upper(); t = to_ccy.upper()
    r = HTTP.get(f"https://open.er-api.com/v6/latest/{f}")
    r.raise_for_status()
    j = r.json()
    if j.get("result") != "success":
        raise ToolError(j.get("error-type", "rate fetch failed"))
    rate = j.get("rates", {}).get(t)
    if rate is None:
        raise ToolError(f"unknown currency: {t}")
    return {"from": f, "to": t, "rate": rate,
            "updated_unix": j.get("time_last_update_unix")}


@tool(
    "is_online",
    description="TCP-connect probe to a host:port. Defaults to port 80.",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "host": {"type": "string"},
            "port": {"type": "integer"},
        },
        "required": ["host"],
    },
)
def is_online(host: str, port: int = 80) -> dict:
    t0 = time.monotonic()
    try:
        with socket.create_connection((host, int(port)), timeout=3.0):
            dt_ms = (time.monotonic() - t0) * 1000.0
        return {"host": host, "port": port, "online": True,
                "rtt_ms": round(dt_ms, 1)}
    except Exception as e:  # noqa: BLE001
        return {"host": host, "port": port, "online": False, "error": str(e)}


@tool(
    "whois",
    description="Basic WHOIS lookup over port 43.",
    category="web",
    schema={
        "type": "object",
        "properties": {"domain": {"type": "string"}},
        "required": ["domain"],
    },
)
def whois(domain: str) -> dict:
    server = "whois.iana.org"
    try:
        with socket.create_connection((server, 43), timeout=5) as s:
            s.sendall((domain + "\r\n").encode())
            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
        text = data.decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"whois error: {e}")
    return {"domain": domain, "text": text[:8000]}


@tool(
    "dns",
    description="Resolve a hostname. record_type ∈ A | AAAA (PTR for IPs).",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "record_type": {"type": "string"},
        },
        "required": ["name"],
    },
)
def dns(name: str, record_type: str = "A") -> dict:
    t = record_type.upper()
    try:
        if t == "PTR":
            host, _, _ = socket.gethostbyaddr(name)
            return {"name": name, "type": t, "answer": host}
        family = socket.AF_INET if t == "A" else socket.AF_INET6
        infos = socket.getaddrinfo(name, None, family=family)
        addrs = sorted({i[4][0] for i in infos})
        return {"name": name, "type": t, "answers": addrs}
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"dns error: {e}")


@tool(
    "screenshot_url",
    description="Render a URL to PNG using headless Chromium. Returns a "
                "file path. NOTE: chromium is not installed on this Jetson "
                "yet; this tool will error until it is.",
    category="web",
    schema={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    needs_ctx=True,
)
def screenshot_url(ctx, url: str) -> dict:
    chrome = shutil.which("chromium-browser") or shutil.which("chromium") \
        or shutil.which("google-chrome")
    if chrome is None:
        raise ToolError(
            "chromium not installed. install with: "
            "sudo apt install -y chromium-browser"
        )
    out = ctx.session_dir / "tools" / f"shot-{uuid_mod.uuid4().hex[:8]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-sandbox",
         "--window-size=1280,800", f"--screenshot={out}", url],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0 or not out.exists():
        raise ToolError(f"screenshot failed: {r.stderr[-500:]}")
    return {"url": url, "path": str(out), "size": out.stat().st_size}


@tool(
    "pdf_to_text",
    description="Fetch a PDF and extract its text.",
    category="web",
    schema={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    needs_ctx=True,
)
def pdf_to_text(ctx, url: str) -> dict:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise ToolError("pdftotext not installed (install poppler-utils)")
    out_dir = ctx.session_dir / "tools" / f"pdf-{uuid_mod.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "doc.pdf"
    r = HTTP.get(url)
    r.raise_for_status()
    pdf.write_bytes(r.content)
    txt = out_dir / "doc.txt"
    cr = subprocess.run([pdftotext, str(pdf), str(txt)],
                        capture_output=True, text=True, timeout=30)
    if cr.returncode != 0:
        raise ToolError(f"pdftotext failed: {cr.stderr[-500:]}")
    text = txt.read_text(errors="replace")
    return {"url": url, "pages_chars": len(text), "text": text[:20000],
            "saved_to": str(txt)}


@tool(
    "read_barcode",
    description="Scan the current camera view for barcodes or QR codes. "
                "Returns a list of decoded codes with their format (EAN13, "
                "UPC, QR, CODE128, etc.) and data. Use barcode_lookup to "
                "resolve a UPC/EAN to a product.",
    category="vision",
    schema={
        "type": "object",
        "properties": {
            "lookup": {
                "type": "boolean",
                "description": "If true, also call barcode_lookup on each "
                               "decoded UPC/EAN code (default true).",
            },
        },
    },
    needs_ctx=True,
)
def read_barcode(ctx, lookup: bool = True) -> dict:
    try:
        from pyzbar.pyzbar import decode as _zbar_decode
        from PIL import Image as _Image
    except ImportError as e:
        raise ToolError(f"pyzbar/Pillow not installed: {e}")
    snap = _tmp_jpg(ctx, "barcode")
    _hd_capture(ctx, snap)
    img = _Image.open(snap)
    codes = _zbar_decode(img)
    items: list[dict] = []
    for c in codes:
        data = c.data.decode("utf-8", errors="replace") if c.data else ""
        fmt = c.type
        rect = c.rect
        item = {
            "data": data,
            "format": fmt,
            "rect": {"x": rect.left, "y": rect.top,
                     "w": rect.width, "h": rect.height},
        }
        if lookup and fmt in ("EAN13", "UPCA", "EAN8", "UPCE"):
            try:
                item["product"] = _lookup_upc(data)
            except Exception as e:  # noqa: BLE001
                item["lookup_error"] = str(e)
        items.append(item)
    return {"frame": str(snap), "count": len(items), "items": items}


def _lookup_upc(code: str) -> dict:
    """OpenFoodFacts UPC/EAN lookup (no API key)."""
    r = HTTP.get(
        f"https://world.openfoodfacts.org/api/v0/product/{code}.json"
    )
    if r.status_code != 200:
        return {"found": False, "code": code, "status": r.status_code}
    j = r.json()
    if j.get("status") != 1:
        return {"found": False, "code": code}
    p = j.get("product") or {}
    return {
        "found": True,
        "code": code,
        "name": p.get("product_name") or p.get("product_name_en"),
        "brand": p.get("brands"),
        "categories": p.get("categories"),
        "quantity": p.get("quantity"),
        "nutriscore": p.get("nutriscore_grade"),
        "ecoscore": p.get("ecoscore_grade"),
        "image_url": p.get("image_front_url"),
        "ingredients": (p.get("ingredients_text") or "")[:500],
        "off_url": f"https://world.openfoodfacts.org/product/{code}",
    }


@tool(
    "barcode_lookup",
    description="Look up a UPC/EAN barcode digit string against the "
                "OpenFoodFacts database. Returns product name, brand, "
                "ingredients, nutri-score, image URL.",
    category="web",
    schema={
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
)
def barcode_lookup(code: str) -> dict:
    code = (code or "").strip().replace(" ", "")
    if not code or not code.isdigit():
        raise ToolError(f"code must be digits, got: {code!r}")
    return _lookup_upc(code)


@tool(
    "research_visual",
    description="Look at what's in front of the camera, extract identifying "
                "text/brand/sign, and search the web for more information about "
                "it. Use this for: products on a shelf, storefronts, articles "
                "of clothing, posters, book covers, barcodes, packaging. "
                "Returns OCR + identifying phrase + web hits in one call.",
    category="vision",
    schema={
        "type": "object",
        "properties": {
            "hint": {
                "type": "string",
                "description": "Optional hint about what to focus on, "
                               "e.g. 'the can on the left' or 'the storefront'.",
            },
        },
    },
    needs_ctx=True,
)
def research_visual(ctx, hint: str = "") -> dict:
    # 1. HD capture for OCR-grade resolution
    snap = _tmp_jpg(ctx, "research")
    _hd_capture(ctx, snap)

    # 2. Identify: ask the VLM for the most search-worthy phrase
    focus = f" Focus on {hint}." if hint else ""
    identify_prompt = (
        f"Look at this image.{focus} Identify the single most distinctive "
        f"text, brand name, product, sign, title, or label visible. Return "
        f"ONLY that phrase — no preamble, no commentary. If there is a "
        f"barcode, transcribe the digits exactly. If nothing identifying "
        f"is visible, return the word: NONE"
    )
    phrase = _vlm_oneshot(ctx, identify_prompt, snap, max_seconds=30.0)
    phrase = (phrase or "").strip().strip("\"'`")
    out: dict = {"frame": str(snap), "identified": phrase, "hint": hint}

    if not phrase or phrase.upper() == "NONE" or len(phrase) < 2:
        # Fallback: ask for a general description and search that
        desc = _vlm_oneshot(
            ctx,
            "In one short sentence, what is the most prominent object in this "
            "image? Just the noun phrase, e.g. 'red guitar', 'office chair'.",
            snap, max_seconds=20.0,
        )
        out["identified"] = desc[:80]
        if not desc:
            out["error"] = "could not identify anything in the frame"
            return out

    query = (out["identified"])[:80]

    # 3. Web search (no API key)
    try:
        out["duckduckgo"] = duckduckgo(query)
    except Exception as e:  # noqa: BLE001
        out["duckduckgo_error"] = str(e)

    # 4. Wikipedia summary (often more authoritative for products/brands)
    try:
        out["wikipedia"] = wikipedia(query)
    except Exception as e:  # noqa: BLE001
        out["wikipedia_error"] = str(e)

    return out


@tool(
    "summarize_page",
    description="Fetch a URL, strip HTML, and ask the VLM to summarise.",
    category="web",
    schema={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    needs_ctx=True,
)
def summarize_page(ctx, url: str) -> dict:
    r = HTTP.get(url)
    r.raise_for_status()
    html = r.text
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html,
                  flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text,
                  flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()[:8000]
    snap = _tmp_jpg(ctx, "sum")
    ctx.capture_frame(snap, allow_reuse=True)
    prompt = (
        "Summarise this web page text into 3-5 bullet points. "
        "Then give a one-line takeaway.\n\nPAGE TEXT:\n" + text
    )
    summary = _vlm_oneshot(ctx, prompt, snap, max_seconds=45.0)
    return {"url": url, "chars_used": len(text), "summary": summary}


# ============================================================================
# IRON-MAN VISION  —  enhance -> locate -> zoom -> identify -> web
# ============================================================================
#
# The "Jarvis vision" loop: point the camera at something, ask what it is,
# then drill in ("what kind of bird?") and Jarvis auto-locates the subject,
# digitally ZOOMS (crop + low-light enhance + upscale), re-identifies at high
# detail, and looks it up on the web. Works in dim rooms (auto-brightness) and
# on arbitrary subjects (open-vocabulary grounding via the VLM, plus a robust
# grid-cell fallback). Drives the dashboard's Vision HUD via run_investigate().


def _mean_luma(jpg_bytes: bytes) -> float:
    """Mean luminance 0-255 of a JPEG, via a 1x1 gray downscale."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "mjpeg", "-i", "-",
             "-vf", "format=gray,scale=1:1", "-frames:v", "1",
             "-f", "rawvideo", "-"],
            input=jpg_bytes, capture_output=True, timeout=4,
        )
        if proc.stdout:
            return float(proc.stdout[0])
    except Exception:
        pass
    return 128.0


def _enhance_vf(luma: float) -> str:
    """ffmpeg filter chain tuned to scene brightness. Dark scenes get a big
    brighten+gamma lift; bright scenes get only mild contrast + sharpen.
    Always finishes with a subtle saturation + unsharp pass."""
    if luma < 45:
        b, c, g = 0.32, 1.7, 1.55
    elif luma < 80:
        b, c, g = 0.20, 1.45, 1.35
    elif luma < 120:
        b, c, g = 0.10, 1.25, 1.18
    elif luma < 160:
        b, c, g = 0.03, 1.12, 1.06
    else:
        b, c, g = 0.0, 1.08, 1.0
    return (f"eq=brightness={b}:contrast={c}:gamma={g}:saturation=1.15,"
            f"unsharp=5:5:0.9:5:5:0.0")


def _render_region(ctx: ToolContext, raw_jpg: bytes, out_jpg: Path,
                   bbox: tuple | None, *, enhance: bool = True,
                   luma: float | None = None, target_long: int = 768) -> dict:
    """Crop a normalised (x, y, w, h) region from a raw camera JPEG, optionally
    low-light enhance, and upscale so the long edge ~= target_long (digital
    zoom). bbox=None renders the full frame. Returns {bbox, px, zoom}."""
    cw, ch = ctx.cam_w, ctx.cam_h
    if bbox is None:
        x, y, w, h = 0.0, 0.0, 1.0, 1.0
    else:
        x, y, w, h = bbox
    cx = max(0, min(cw - 16, int(x * cw)))
    cy = max(0, min(ch - 16, int(y * ch)))
    pw = max(32, min(cw - cx, int(w * cw)))
    ph = max(32, min(ch - cy, int(h * ch)))
    if luma is None:
        luma = _mean_luma(raw_jpg)
    long_edge = max(pw, ph)
    zoom = max(1.0, min(4.0, target_long / float(long_edge)))
    ow, oh = int(pw * zoom), int(ph * zoom)
    vf = [f"crop={pw}:{ph}:{cx}:{cy}"]
    if enhance:
        vf.append(_enhance_vf(luma))
    vf.append(f"scale={ow}:{oh}:flags=lanczos")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "mjpeg", "-i", "-", "-vf", ",".join(vf),
         "-frames:v", "1", "-q:v", "3", str(out_jpg)],
        input=raw_jpg, check=True, capture_output=True, timeout=12,
    )
    return {
        "bbox": [round(cx / cw, 4), round(cy / ch, 4),
                 round(pw / cw, 4), round(ph / ch, 4)],
        "px": [cx, cy, pw, ph],
        "zoom": round(zoom, 2),
        "luma": round(luma, 1),
    }


OWL_URL = os.environ.get("OWL_URL", "http://127.0.0.1:8086")


def _owl_available() -> bool:
    try:
        r = HTTP.get(OWL_URL + "/health",
                     timeout=httpx.Timeout(2.0, connect=1.0))
        return r.status_code == 200
    except Exception:
        return False


def _owl_detect(jpg_path: Path, query: str, threshold: float = 0.1) -> dict | None:
    """Open-vocab detect via the NanoOWL sidecar. Returns the best detection
    {label, score, bbox:[x,y,w,h]} or None (also None if sidecar is down)."""
    try:
        b64 = base64.b64encode(Path(jpg_path).read_bytes()).decode()
        r = HTTP.post(OWL_URL + "/detect",
                      json={"image_b64": b64, "query": query,
                            "threshold": threshold},
                      timeout=httpx.Timeout(20.0, connect=2.0))
        if r.status_code == 200:
            return r.json().get("best")
    except Exception:
        return None
    return None


_GRID_COLS, _GRID_ROWS = 4, 3


def _grid_locate(ctx: ToolContext, subject: str, enhanced_full: Path) -> tuple | None:
    """Robust open-vocabulary localisation for a 3B VLM: instead of regressing
    pixel coords (unreliable), ask which grid cell contains the subject, then
    return a padded normalised bbox around that cell. None if not found."""
    prompt = (
        f"This image is divided into a grid of {_GRID_COLS} columns "
        f"(1=left .. {_GRID_COLS}=right) and {_GRID_ROWS} rows "
        f"(1=top .. {_GRID_ROWS}=bottom). In which single cell is the CENTER "
        f"of the {subject or 'main subject'}? Reply with ONLY two numbers "
        f"'column,row' (e.g. '3,2'). If the {subject or 'subject'} is not "
        f"visible, reply '0,0'."
    )
    raw = _vlm_oneshot(ctx, prompt, enhanced_full, max_seconds=18.0)
    nums = re.findall(r"\d+", raw or "")
    if len(nums) < 2:
        return None
    col, row = int(nums[0]), int(nums[1])
    if col < 1 or row < 1 or col > _GRID_COLS or row > _GRID_ROWS:
        return None
    cw_n, ch_n = 1.0 / _GRID_COLS, 1.0 / _GRID_ROWS
    # centre of the chosen cell, then a padded box ~2 cells wide/tall
    cxc = (col - 0.5) * cw_n
    cyc = (row - 0.5) * ch_n
    half_w, half_h = cw_n, ch_n  # => box spans 2 cells each way
    x = max(0.0, cxc - half_w)
    y = max(0.0, cyc - half_h)
    w = min(1.0 - x, half_w * 2)
    h = min(1.0 - y, half_h * 2)
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))


def _wiki_summary(query: str) -> dict:
    """Wikipedia summary incl. thumbnail (for the HUD fact card)."""
    try:
        sr = HTTP.get("https://en.wikipedia.org/w/api.php",
                      params={"action": "query", "list": "search",
                              "srsearch": query, "format": "json", "srlimit": 1})
        sr.raise_for_status()
        hits = sr.json().get("query", {}).get("search", [])
        if not hits:
            return {"found": False}
        title = hits[0]["title"]
        sm = HTTP.get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                      + urllib.parse.quote(title.replace(" ", "_")))
        if sm.status_code != 200:
            return {"found": True, "title": title}
        j = sm.json()
        return {
            "found": True,
            "title": j.get("title"),
            "extract": j.get("extract"),
            "url": (j.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
            "thumbnail": (j.get("thumbnail") or {}).get("source"),
        }
    except Exception as e:  # noqa: BLE001
        return {"found": False, "error": str(e)}


_DDG_ANCHOR_RE = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.S)
_DDG_HREF_RE = re.compile(r'href="([^"]+)"')
_DDG_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>', re.S)


def _strip_tags(s: str) -> str:
    import html as _html
    return _html.unescape(
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or ""))).strip()


def _ddg_unwrap(href: str) -> str:
    """DDG wraps result links as //duckduckgo.com/l/?uddg=<urlencoded>."""
    href = href.replace("&amp;", "&")
    if "uddg=" in href:
        q = urllib.parse.urlparse(
            href if "://" in href else "https:" + href).query
        uddg = urllib.parse.parse_qs(q).get("uddg")
        if uddg:
            return urllib.parse.unquote(uddg[0])
    if href.startswith("//"):
        return "https:" + href
    return href


_DDG_UA = ("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


@tool(
    "web_search",
    description="General web search with real result links (title, url, "
                "snippet). Use this for facts, identifications, prices, "
                "species/model/brand lookups — anything where you need live "
                "results from the open web, not just an encyclopedia entry.",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "n": {"type": "integer", "description": "max results (1-8)"},
        },
        "required": ["query"],
    },
)
def web_search(query: str, n: int = 5) -> dict:
    n = max(1, min(8, int(n)))
    results: list[dict] = []
    seen: set[str] = set()

    def _add(title: str, url: str, snippet: str = "") -> None:
        url = (url or "").strip()
        if not url or url in seen or "duckduckgo.com" in url:
            return
        seen.add(url)
        results.append({"title": _strip_tags(title), "url": url,
                        "snippet": _strip_tags(snippet)})

    try:
        r = HTTP.get("https://html.duckduckgo.com/html/",
                     params={"q": query}, headers={"User-Agent": _DDG_UA})
        if r.status_code == 200:
            snips = _DDG_SNIPPET_RE.findall(r.text)
            si = 0
            for attrs, inner in _DDG_ANCHOR_RE.findall(r.text):
                if "result__a" not in attrs:
                    continue
                m = _DDG_HREF_RE.search(attrs)
                if not m:
                    continue
                snip = snips[si] if si < len(snips) else ""
                si += 1
                _add(inner, _ddg_unwrap(m.group(1)), snip)
                if len(results) >= n:
                    break
    except Exception as e:  # noqa: BLE001
        return {"query": query, "results": [], "error": str(e)}

    # Fallback to the lite endpoint if the html layout yielded nothing.
    if not results:
        try:
            r = HTTP.post("https://lite.duckduckgo.com/lite/",
                          data={"q": query}, headers={"User-Agent": _DDG_UA})
            if r.status_code == 200:
                for attrs, inner in _DDG_ANCHOR_RE.findall(r.text):
                    m = _DDG_HREF_RE.search(attrs)
                    if not m:
                        continue
                    url = _ddg_unwrap(m.group(1))
                    if url.startswith("http"):
                        _add(inner, url)
                    if len(results) >= n:
                        break
        except Exception as e:  # noqa: BLE001
            return {"query": query, "results": results, "error": str(e)}

    return {"query": query, "count": len(results), "results": results}


def _clean_query(s: str) -> str:
    """Turn a model-authored identification/search line into a clean web query:
    strip quotes/brackets/trailing punctuation, drop hedge words, cap length."""
    s = (s or "").strip().strip("\"'`*").strip()
    s = re.sub(r"^(a|an|the|some|likely|possibly|probably)\s+", "", s, flags=re.I)
    s = re.sub(r"[\"'`\[\](){}]", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,:;-")
    words = s.split()
    return " ".join(words[:8])


def _parse_identify(text: str, subject: str) -> dict:
    """Parse the structured identify reply (ID/CONFIDENCE/DETAILS/SEARCH)."""
    out = {"name": "", "confidence": "", "details": "", "search": ""}
    for line in (text or "").splitlines():
        m = re.match(r"\s*(ID|CONFIDENCE|DETAILS|SEARCH)\s*[:\-]\s*(.+)",
                     line, re.I)
        if m:
            out[m.group(1).lower().replace("id", "name")
                if m.group(1).upper() == "ID" else m.group(1).lower()] = \
                m.group(2).strip()
    # fallbacks if the model ignored the format
    if not out["name"]:
        first = _strip_tags(text).strip().split(". ")[0][:80]
        out["name"] = first or (subject or "unknown")
    if not out["search"]:
        out["search"] = out["name"] if out["name"] else subject
    return out


def run_investigate(ctx: ToolContext, *, subject: str = "", point=None,
                    region=None, web: bool = True,
                    on_event=None) -> dict:
    """The full Iron-Man vision pipeline. Emits progress via on_event(dict) for
    the SSE HUD; also returns the final structured result. Reused by both the
    /investigate endpoint and the `investigate` tool."""
    def emit(ev: dict) -> None:
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:
                pass

    tid = ev_dir = None
    # artifacts live under session_dir/inv/<id>/ ; served via /inv/<id>/<f>
    inv_id = "inv-" + uuid_mod.uuid4().hex[:8]
    ev_dir = ctx.session_dir / "inv" / inv_id
    ev_dir.mkdir(parents=True, exist_ok=True)

    emit({"phase": "capturing"})
    raw = ctx.camera.get_latest()
    luma = _mean_luma(raw)
    emit({"phase": "enhancing", "luma": round(luma, 1)})

    # full enhanced frame (HUD context + grid localisation surface)
    full = ev_dir / "full.jpg"
    _render_region(ctx, raw, full, None, enhance=True, luma=luma,
                   target_long=max(ctx.cam_w, ctx.cam_h))
    full_url = f"/inv/{inv_id}/full.jpg"

    # ---- locate -----------------------------------------------------------
    emit({"phase": "locating", "subject": subject})
    bbox = None
    locate_method = "full"
    if region and len(region) == 4:
        bbox = tuple(float(v) for v in region)
        locate_method = "region"
    elif point and len(point) == 2:
        px, py = float(point[0]), float(point[1])
        side = 0.20   # tight crop so the TAPPED object dominates, not surroundings
        bbox = (max(0.0, px - side / 2), max(0.0, py - side / 2), side, side)
        bbox = (bbox[0], bbox[1], min(side, 1.0 - bbox[0]),
                min(side, 1.0 - bbox[1]))
        locate_method = "point"
    elif subject:
        # Prefer precise open-vocab detection (NanoOWL) when the sidecar is up;
        # fall back to VLM grid-cell voting. Detect on the enhanced full frame
        # (better recall in dim scenes).
        best = _owl_detect(full, subject, threshold=0.1)
        if best and best.get("bbox") and best.get("score", 0) >= 0.1:
            bx = best["bbox"]
            # pad the tight detector box ~15% so the zoom keeps context
            px, py, pw, ph = bx
            bbox = (max(0.0, px - pw * 0.15), max(0.0, py - ph * 0.15),
                    min(1.0, pw * 1.3), min(1.0, ph * 1.3))
            locate_method = "owl"
            emit({"phase": "detect", "label": best.get("label"),
                  "score": best.get("score")})
        else:
            found = _grid_locate(ctx, subject, full)
            if found:
                bbox = found
                locate_method = "grid"
    emit({"phase": "located", "bbox": list(bbox) if bbox else None,
          "method": locate_method})

    # ---- zoom -------------------------------------------------------------
    emit({"phase": "zooming"})
    zoom = ev_dir / "zoom.jpg"
    zmeta = _render_region(ctx, raw, zoom, bbox, enhance=True, luma=luma,
                           target_long=768)
    zoom_url = f"/inv/{inv_id}/zoom.jpg"
    emit({"phase": "zoomed", "zoom_url": zoom_url, "bbox": zmeta["bbox"],
          "zoom": zmeta["zoom"]})

    # ---- identify (fine-grained) -----------------------------------------
    emit({"phase": "identifying"})
    if subject:
        subj_clause = f"Focus on the {subject}. "
    elif locate_method == "point":
        subj_clause = ("The user tapped a precise spot in the scene — identify "
                       "the single object at the CENTER of this crop, ignoring "
                       "the background, the chair, and anything at the edges. ")
    else:
        subj_clause = "Focus on the most prominent / central subject. "
    id_prompt = (
        "You are a visual identification expert looking at an enhanced, "
        "zoomed-in crop. " + subj_clause +
        "Identify it as SPECIFICALLY as the image actually supports — exact "
        "species, make/model, brand, or type. Base it ONLY on what is clearly "
        "visible; do NOT guess a specific identity the image does not support. "
        "If the crop is too dark, blurry, or ambiguous to tell, give your best "
        "GENERAL guess and set CONFIDENCE to low. Use high confidence only when "
        "the identification is unmistakable. Reply in EXACTLY this format, one "
        "per line, nothing else:\n"
        "ID: <a SHORT specific name, 2-5 words, no full sentence — "
        "e.g. 'Northern Cardinal', 'Toyota Tacoma', 'office chair'>\n"
        "CONFIDENCE: <high|medium|low>\n"
        "DETAILS: <one sentence of distinguishing visual features>\n"
        "SEARCH: <2-6 word web search query to learn more>"
    )
    id_raw = _vlm_oneshot(ctx, id_prompt, zoom, max_seconds=30.0)
    ident = _parse_identify(id_raw, subject)
    emit({"phase": "identified", **ident, "raw": id_raw})

    out: dict = {
        "inv_id": inv_id,
        "subject": subject,
        "full_url": full_url,
        "zoom_url": zoom_url,
        "bbox": zmeta["bbox"],
        "zoom": zmeta["zoom"],
        "luma": zmeta["luma"],
        "locate_method": locate_method,
        "identification": ident,
        "web": {},
    }

    # ---- web --------------------------------------------------------------
    # The model's free-form SEARCH line is often over-specific/quoted; the
    # cleaned identification name is a far better query for wiki + web.
    query = (_clean_query(ident.get("name", ""))
             or _clean_query(ident.get("search", ""))
             or _clean_query(subject))
    out["query"] = query
    # Don't web-search a non-identification (e.g. VLM hiccup -> "unknown"),
    # which would return junk results for the literal word.
    _JUNK_Q = {"", "unknown", "none", "n/a", "nothing", "object", "thing",
               "unclear", "unidentified", "it"}
    if web and query and query.lower() not in _JUNK_Q:
        emit({"phase": "researching", "query": query})
        wiki = _wiki_summary(query)
        try:
            srch = web_search(query, n=4)
        except Exception as e:  # noqa: BLE001
            srch = {"results": [], "error": str(e)}
        out["web"] = {"wikipedia": wiki, "search": srch}
        emit({"phase": "researched", "query": query,
              "wikipedia": wiki, "search": srch})

    emit({"phase": "done", "result": out})
    return out


@tool(
    "investigate",
    description="THE vision drill-down tool. Point the camera at something and "
                "identify it in detail: auto-locates the subject, digitally "
                "zooms in (low-light enhance + upscale), identifies it as "
                "specifically as possible (species, make/model, brand), and "
                "looks it up on the web. Use for 'what kind of bird/car/plant "
                "is that', storefronts, products, anything needing a closer "
                "look. Pass `subject` (e.g. 'bird', 'car', 'the sign').",
    category="vision",
    schema={
        "type": "object",
        "properties": {
            "subject": {"type": "string",
                        "description": "what to zoom in on and identify"},
            "web": {"type": "boolean",
                    "description": "also look it up on the web (default true)"},
        },
    },
    needs_ctx=True,
)
def investigate(ctx, subject: str = "", web: bool = True) -> dict:
    res = run_investigate(ctx, subject=subject, web=web)
    # compact view for the agent transcript (full artifacts via the HUD)
    ident = res.get("identification", {})
    wiki = (res.get("web", {}) or {}).get("wikipedia", {}) or {}
    hits = (((res.get("web", {}) or {}).get("search", {}) or {})
            .get("results", []) or [])
    return {
        "identified": ident.get("name"),
        "confidence": ident.get("confidence"),
        "details": ident.get("details"),
        "zoom_url": res.get("zoom_url"),
        "wikipedia": {"title": wiki.get("title"),
                      "extract": (wiki.get("extract") or "")[:600]},
        "web_results": [{"title": h.get("title"), "url": h.get("url")}
                        for h in hits[:3]],
    }


@tool(
    "detect_objects",
    description="Open-vocabulary object detection: find ANY objects you name in "
                "the current camera view and return their locations (bounding "
                "boxes). Real-time NanoOWL detector. Use to locate/count/track "
                "specific things, e.g. 'person, dog, cup' or 'red car'.",
    category="vision",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "comma-separated things to find, "
                                     "e.g. 'person, laptop, coffee cup'"},
            "threshold": {"type": "number",
                          "description": "confidence 0-1 (default 0.1)"},
        },
        "required": ["query"],
    },
    needs_ctx=True,
)
def detect_objects(ctx, query: str, threshold: float = 0.1) -> dict:
    if not _owl_available():
        raise ToolError("open-vocab detector (NanoOWL sidecar) is not running")
    snap = _tmp_jpg(ctx, "owl")
    ctx.capture_frame(snap, allow_reuse=False)
    try:
        b64 = base64.b64encode(snap.read_bytes()).decode()
        r = HTTP.post(OWL_URL + "/detect",
                      json={"image_b64": b64, "query": query,
                            "threshold": float(threshold)},
                      timeout=httpx.Timeout(20.0, connect=2.0))
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"detect failed: {e}")
    dets = j.get("detections", [])
    return {
        "query": query, "count": len(dets), "frame": str(snap),
        "detections": [{"label": d["label"], "score": d["score"],
                        "bbox": d["bbox"]} for d in dets[:15]],
    }


# ============================================================================
# VISUAL MEMORY  —  recall what the camera has seen over time ("world model")
# ============================================================================

def _ago(ts: float) -> str:
    d = max(0, int(time.time() - ts))
    if d < 60:
        return f"{d}s ago"
    if d < 3600:
        return f"{d // 60}m ago"
    if d < 86400:
        return f"{d // 3600}h ago"
    return f"{d // 86400}d ago"


@tool(
    "recall_visual",
    description="Search Jarvis's visual memory of everything the camera has "
                "seen over time. Use for 'when/where did you last see X', "
                "'have you seen my keys', 'what did you see earlier'. Returns "
                "matching past observations with how long ago they were seen.",
    category="memory",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "what to look for in past observations"},
            "k": {"type": "integer", "description": "max results (default 6)"},
        },
        "required": ["query"],
    },
    needs_ctx=True,
)
def recall_visual(ctx, query: str, k: int = 6) -> dict:
    k = max(1, min(20, int(k)))
    hits = ctx.memory.vmem_search(query, k)
    return {
        "query": query,
        "found": len(hits),
        "observations": [
            {"when": _ago(h["ts"]), "caption": h["caption"],
             "objects": h["objects"], "frame_url": h["frame_url"]}
            for h in hits
        ],
    }


@tool(
    "visual_timeline",
    description="The most recent things the camera has observed, newest first. "
                "Use for 'what have you seen recently' or 'what's been "
                "happening'.",
    category="memory",
    schema={"type": "object",
            "properties": {"n": {"type": "integer"}}},
    needs_ctx=True,
)
def visual_timeline(ctx, n: int = 8) -> dict:
    n = max(1, min(30, int(n)))
    items = ctx.memory.vmem_recent(n)
    return {"count": len(items),
            "timeline": [{"when": _ago(h["ts"]), "caption": h["caption"],
                          "objects": h["objects"]} for h in items]}


@tool(
    "remember_now",
    description="Capture and commit the current camera view to visual memory "
                "right now (caption + objects). Use when the user says "
                "'remember this' / 'note what you see'.",
    category="memory",
    needs_ctx=True,
)
def remember_now(ctx) -> dict:
    if ctx.visual_memory is None:
        raise ToolError("visual memory not available")
    rec = ctx.visual_memory.capture_now(source="manual")
    return {"saved": True, "caption": rec.get("caption"),
            "objects": rec.get("objects")}


@tool(
    "watch_add",
    description="Set up a proactive alert. Give a natural-language condition "
                "and Jarvis will watch the camera and notify when it becomes "
                "true. Use for 'alert me if/when X', 'tell me if someone X', "
                "'let me know when X'. e.g. 'a person is at the door', "
                "'the dog is on the couch', 'someone is near the desk'.",
    category="self",
    schema={
        "type": "object",
        "properties": {
            "condition": {"type": "string",
                          "description": "what to watch for, as a statement "
                                         "that is true when it should alert"},
        },
        "required": ["condition"],
    },
    needs_ctx=True,
)
def watch_add(ctx, condition: str) -> dict:
    condition = (condition or "").strip()
    if not condition:
        raise ToolError("condition required")
    rid = ctx.memory.watch_add(condition)
    return {"added": True, "id": rid, "condition": condition,
            "note": "Jarvis will check this whenever the scene changes."}


@tool(
    "watch_list",
    description="List the active proactive watch/alert rules.",
    category="self",
    needs_ctx=True,
)
def watch_list(ctx) -> dict:
    rules = ctx.memory.watch_list()
    return {"count": len(rules),
            "rules": [{"id": r["id"], "condition": r["text"],
                       "active": r["active"], "times_fired": r["fire_count"]}
                      for r in rules]}


@tool(
    "watch_remove",
    description="Remove/stop a proactive watch rule by its id (from watch_list).",
    category="self",
    schema={"type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"]},
    needs_ctx=True,
)
def watch_remove(ctx, id: int) -> dict:
    ctx.memory.watch_remove(int(id))
    return {"removed": True, "id": int(id)}


# ============================================================================
# AGENTIC LOOP  (plan -> act -> observe -> re-plan, until natural answer)
# ============================================================================

# Tools the VLM is allowed to call autonomously. Excludes destructive ones
# (restart_self, forget, scan_room) and any that need explicit human intent.
# Curated for a voice-first, location-aware desk Jarvis (trimmed from 89 → leaner
# prompt = faster, more stable, better tool selection). Cut tools still EXIST and
# are callable via the UI/API — they're just not offered to the agent. Re-add a
# name here to bring it back.
DEFAULT_AGENT_ALLOW: set[str] = {
    # reasoning / utility
    "do_math", "convert", "now", "coin_flip", "roll",
    # web / knowledge (location-aware)
    "where_am_i", "weather", "forecast", "time_in", "currency_rate",
    "web_search", "wikipedia", "research",
    "sun_times", "news", "define", "translate", "stock_price", "crypto_price",
    "nearby_places", "briefing",
    # vision (the eyes)
    "investigate", "read_all_text", "ocr_translate", "detect_objects",
    "zoom_into", "research_visual", "multi_frame_compare",
    "read_barcode", "barcode_lookup",
    # memory
    "recall", "recent", "recall_visual", "visual_timeline", "remember_now",
    "summarize_today", "pin_this", "pinned", "unpin",
    "profile_get", "profile_set",
    # productivity
    "reminder_set", "reminder_list", "timer",
    # smart home
    "hue_lights", "lights_state", "lights_on", "lights_off",
    "lights_set", "scene_activate", "lights_pulse",
    # self / diagnostics + UI control
    "show_panel",
    "system_status", "status_report", "network_speed", "health_check",
    "enable_live_mode", "disable_live_mode", "set_persona", "wake_word_off",
    "watch_add", "watch_list", "watch_remove", "restart_self",
    # audio
    "play_sound",
    # cloud escalation — opt-in only (skipped at registration unless keys set)
    "ask_claude", "ask_gpt", "ask_gemini", "escalate",
}


AGENT_SYSTEM = """\
You are Jarvis, a personal AI agent that takes action by calling tools.

You MUST use a tool whenever the user asks about anything you cannot know
from training data alone, including:
- Current time, date, day of week
- Current weather, forecast
- Live news, Hacker News, web search, Wikipedia
- The user's notes, todos, reminders, bookmarks
- What the camera sees right now — including any object, sign, product,
  storefront, barcode, or clothing item visible in front of the user.
  For these, CALL research_visual or read_all_text or zoom_into. Do NOT
  refuse — the tools fetch the camera frame for you.
- Math that benefits from a calculator
- Anything happening "right now", "today", "currently", or "the latest"
- Controlling lights via Hue (lights_on, lights_off, lights_set, scene_activate)
- Showing the user something on screen — whenever they ask to see / show /
  pull up / open / bring up a panel (what you've SEEN, the timeline, your
  memory, tools, entities, the link chart, the 3D point cloud), CALL show_panel
  with the panel name. Open the panel first, THEN say a brief sentence.

Tool-call format. When you need a tool, output EXACTLY this and STOP — no
preamble, no explanation, just the tag:

<tool_call>{"name": "TOOL_NAME", "args": {"PARAM": "VALUE"}}</tool_call>

The system will run the tool and reply:

<tool_result name="TOOL_NAME">...JSON result...</tool_result>

You then either call another tool, or write your final answer in plain
prose (no tags).

Examples:

User: What time is it in Sydney?
You: <tool_call>{"name": "time_in", "args": {"tz": "Australia/Sydney"}}</tool_call>
(then, after <tool_result>): It's currently 3:14 PM in Sydney.

User: What's the weather in Tokyo?
You: <tool_call>{"name": "weather", "args": {"location": "Tokyo"}}</tool_call>
(then): Tokyo is 19°C and partly cloudy, with 77% humidity.

User: Show me what you've seen recently.
You: <tool_call>{"name": "show_panel", "args": {"panel": "seen"}}</tool_call>
(then): Here you are, sir — that's what I've observed.

User: Pull up the timeline.
You: <tool_call>{"name": "show_panel", "args": {"panel": "timeline"}}</tool_call>
(then): Timeline's up.

User: What's 1.23 times 456 plus 78?
You: <tool_call>{"name": "do_math", "args": {"expr": "1.23*456 + 78"}}</tool_call>
(then): The result is 638.88.

User: Remind me to call Alice in 20 minutes.
You: <tool_call>{"name": "reminder_set", "args": {"when": "in 20 minutes", "what": "call Alice"}}</tool_call>
(then): Reminder set for 20 minutes from now.

Vision + web example — when the user asks ABOUT something the camera sees
(a product on a shelf, a storefront, a barcode, an article of clothing):

User: What is this can?
You: <tool_call>{"name": "research_visual", "args": {"hint": "the can"}}</tool_call>
(then): That's a 12 oz Coca-Cola Zero Sugar — zero calories, sweetened with aspartame.

Alternatively, if you want fine control, chain manually:

User: What does that storefront sign say and what is the place?
You: <tool_call>{"name": "read_all_text", "args": {}}</tool_call>
(then): <tool_call>{"name": "duckduckgo", "args": {"query": "<the storefront text>"}}</tool_call>
(then): It says "BLUE BOTTLE COFFEE". They're a roaster from Oakland — most outlets are in CA, NY, and Tokyo.

Drill-down example — when the user asks to identify something SPECIFICALLY
("what kind of bird/car/plant is that", a storefront, a product), use
investigate. It auto-locates the subject, zooms in with low-light enhance,
identifies it precisely, and looks it up on the web in one call:

User: What kind of bird is that?
You: <tool_call>{"name": "investigate", "args": {"subject": "bird"}}</tool_call>
(then): That's a Northern Cardinal — male, by the crest and black face mask. They're year-round residents across the eastern US.

Use investigate for "what kind / what model / tell me more about that <thing>".
Use research_visual for one-shot text/brand identification + lookup. Use the
manual chain when you need to disambiguate or pick a specific search angle.

Compound requests. If the user asks for multiple distinct actions in one
message (e.g. "get the weather AND set a reminder"), emit one <tool_call>
tag per action, back-to-back, in a single response. The system will run
all of them and return all results together. Example:

User: Get the weather in Tokyo and remind me to drink water in 30 minutes.
You:
<tool_call>{"name": "weather", "args": {"location": "Tokyo"}}</tool_call>
<tool_call>{"name": "reminder_set", "args": {"when": "in 30 minutes", "what": "drink water"}}</tool_call>

Rules:
- ONE tool call per ACTION. For compound requests, emit MULTIPLE
  <tool_call> tags back-to-back (one per action).
- Output ONLY <tool_call> tags — no preamble or narration between them.
- Use exact tool names from the catalog. Do not invent tools.
- Once you have all the results from <tool_result>, write a concise reply.
- Be terse. The user wants the answer, not the process.

Available tools:"""


def _summarize_tool_for_prompt(t: Tool) -> str:
    """One-line description of a tool for the agent prompt."""
    props = (t.schema or {}).get("properties", {}) or {}
    required = set((t.schema or {}).get("required") or [])
    parts = []
    for k, v in props.items():
        typ = v.get("type", "any")
        marker = "" if k in required else "?"
        parts.append(f"{k}{marker}:{typ}")
    sig = ", ".join(parts)
    desc = (t.description or "").split(". ")[0]  # first sentence only
    return f"- {t.name}({sig}) — {desc}"


def agent_prompt(allowlist: set | None = None,
                 memory=None) -> str:
    """Build the FULL system prompt for agent mode (replaces persona)."""
    allowed = allowlist or DEFAULT_AGENT_ALLOW
    lines = []
    if memory is not None:
        prof = profile_prompt_block(memory)
        if prof:
            lines.append(prof)
    lines.append(AGENT_SYSTEM)
    lines.append("")
    loc = device_location_str()
    if loc:
        lines.append(f"Current location: {loc}. Use it as the default place for "
                     f"weather, time, and local questions unless the user names "
                     f"another place.")
        lines.append("")
    by_cat: dict[str, list[str]] = {}
    for name in sorted(allowed):
        t = TOOLS._tools.get(name)
        if t is None:
            continue
        by_cat.setdefault(t.category, []).append(_summarize_tool_for_prompt(t))
    for cat in sorted(by_cat):
        lines.append(f"## {cat}")
        lines.extend(by_cat[cat])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_TOOL_CALL_RE_STRICT = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S,
)
_TOOL_CALL_RE_OPEN = re.compile(r"<tool_call>\s*(\{.*)", re.S)


def _balanced_json(s: str) -> str | None:
    """Return the first balanced {...} substring from s, or None."""
    s = s.lstrip()
    if not s.startswith("{"):
        return None
    depth = 0; in_str = False; esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False; continue
        if ch == "\\":
            esc = True; continue
        if ch == '"' and not esc:
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[:i + 1]
    return None


def _decode_call_blob(raw: str) -> dict | None:
    for attempt in (raw, raw.replace("'", '"'),
                    re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            obj = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            return {
                "name": str(obj["name"]),
                "args": obj.get("args") or obj.get("arguments") or {},
            }
    return None


def parse_tool_call(text: str) -> dict | None:
    """Extract the FIRST tool call from model output. Tolerates:
       1) Properly closed   <tool_call>{...}</tool_call>
       2) Unclosed tag      <tool_call>{...}
       3) Bare JSON         {"name": "...", "args": {...}}
    """
    calls = parse_all_tool_calls(text, max_calls=1)
    return calls[0] if calls else None


def parse_all_tool_calls(text: str, *, max_calls: int = 5) -> list[dict]:
    """Find up to max_calls tool calls in the text, in order of appearance.

    Handles mixed forms (closed tags, open tags, bare JSON) in the same
    message. Designed for compound requests where the model emits multiple
    <tool_call> tags in one response.
    """
    if not text:
        return []
    found: list[dict] = []
    seen_spans: list[tuple[int, int]] = []

    def _take(blob: str, start: int, end: int) -> None:
        for s, e in seen_spans:
            if not (end <= s or start >= e):
                return
        obj = _decode_call_blob(blob)
        if obj:
            found.append(obj)
            seen_spans.append((start, end))

    for m in _TOOL_CALL_RE_STRICT.finditer(text):
        if len(found) >= max_calls:
            break
        _take(m.group(1).strip(), m.start(), m.end())

    for m in re.finditer(r"<tool_call>\s*", text):
        if len(found) >= max_calls:
            break
        rest = text[m.end():]
        blob = _balanced_json(rest)
        if blob:
            _take(blob, m.start(), m.end() + len(blob))

    if not found:
        # bare JSON fallback: scan for objects containing "name"
        i = 0
        while i < len(text) and len(found) < max_calls:
            if text[i] == "{":
                blob = _balanced_json(text[i:])
                if blob and '"name"' in blob:
                    _take(blob, i, i + len(blob))
                    i += len(blob)
                    continue
            i += 1

    # de-dup identical sequential calls (some models stutter)
    out: list[dict] = []
    for c in found:
        key = (c["name"], json.dumps(c.get("args") or {}, sort_keys=True))
        if out and (out[-1]["name"], json.dumps(out[-1].get("args") or {}, sort_keys=True)) == key:
            continue
        out.append(c)
    return out[:max_calls]


def _agent_chat(messages: list[dict], *, max_tokens: int, temperature: float,
                timeout_s: float = 45.0) -> tuple[str, dict]:
    """Single non-streaming call to llama-server."""
    body = {
        "model": "qwen2.5-vl-3b",
        "stream": False,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    with VLM_BUSY:
        r = httpx.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            json=body, timeout=httpx.Timeout(timeout_s, connect=5.0),
        )
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"].strip()
    usage = j.get("usage") or {}
    return text, usage


def agentic_loop(
    ctx: ToolContext,
    question: str,
    *,
    frame_path: Path | None = None,
    max_steps: int = 3,
    allowlist: set | None = None,
    use_frame: bool = False,
    on_step: Callable[[dict], None] | None = None,
) -> dict:
    """
    Plan -> act -> observe -> re-plan until the model emits a natural answer
    or max_steps is reached. Non-streaming; one VLM call per step.

    Returns: {question, final, steps, stopped, frame, elapsed_s, usage_total}
    """
    allowed = allowlist or DEFAULT_AGENT_ALLOW
    t_start = time.monotonic()

    user_content: list = []
    if use_frame:
        if frame_path is None:
            frame_path = _tmp_jpg(ctx, "agent")
            ctx.capture_frame(frame_path, allow_reuse=True)
        b64 = base64.b64encode(frame_path.read_bytes()).decode()
        user_content.append(
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    user_content.append({"type": "text", "text": question})

    with ctx.settings_lock:
        max_t = max(ctx.settings.get("max_tokens", 240), 400)
        temp = ctx.settings.get("temperature", 0.2)

    # Agent mode REPLACES the persona system prompt; the persona's vision
    # constraints (e.g. "describe what you see") would otherwise crowd out
    # tool reasoning. Profile facts are prepended if the user has set any.
    sys_prompt = agent_prompt(allowed, memory=ctx.memory)
    messages: list = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    steps: list[dict] = []
    final_text = ""
    stopped = "natural"
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    last_text = ""
    step_counter = 0
    for round_i in range(max_steps):
        try:
            text, usage = _agent_chat(
                messages, max_tokens=max_t, temperature=temp,
            )
        except Exception as e:  # noqa: BLE001
            stopped = f"error: {type(e).__name__}: {e}"
            break
        last_text = text
        for k in usage_total:
            usage_total[k] += int(usage.get(k) or 0)

        # Keep "eyes" on the FIRST call of the turn, then drop the image so the
        # follow-up tool-result rounds run text-only. Re-encoding the ~800MB mmproj
        # frame on every round is what crashes the 8GB box; one encode per turn
        # (same as a normal vision turn) keeps vision always-on but stable.
        if round_i == 0 and use_frame and isinstance(messages[1].get("content"), list):
            messages[1]["content"] = question

        # Multi-tool: parse ALL <tool_call> tags emitted in this response.
        calls = parse_all_tool_calls(text, max_calls=5)
        if not calls:
            final_text = text
            stopped = "natural"
            break

        # Dispatch each call; collect per-step records and result tags.
        result_tags: list[str] = []
        for call in calls:
            step_counter += 1
            name = call["name"]
            args = call["args"] if isinstance(call["args"], dict) else {}
            if name not in allowed:
                result = {
                    "ok": False,
                    "error": f"tool '{name}' is not allowed in agent mode",
                }
            else:
                result = TOOLS.call(name, args, confirmed=False)
                try:
                    if hasattr(ctx.memory, "_con"):
                        TOOLS.log_call(ctx.memory, name, args, result, 0.0)
                except Exception:
                    pass

            payload = result.get("result") if result.get("ok") else result
            try:
                result_compact = json.dumps(payload, default=str)[:2000]
            except (TypeError, ValueError):
                result_compact = str(payload)[:2000]

            step_record = {
                "step": step_counter,
                "round": round_i + 1,
                "model_text": text if step_counter == round_i * 5 + 1 else "",
                "tool": name,
                "args": args,
                "result": result,
                "usage": usage if step_counter == round_i * 5 + 1 else {},
            }
            steps.append(step_record)
            if on_step is not None:
                try:
                    on_step(step_record)
                except Exception:
                    pass

            result_tags.append(
                f'<tool_result name="{name}">{result_compact}</tool_result>'
            )

        # Feed all results back in a single user message so the model can
        # see them together and emit a final natural reply (or another round).
        messages.append({"role": "assistant", "content": text})
        followup = "\n".join(result_tags) + (
            "\n\nAll results above. If you need more tools, emit another "
            "<tool_call>. Otherwise write the final answer in plain text."
        )
        messages.append({"role": "user", "content": followup})
    else:
        # max_steps exhausted without a natural answer
        final_text = last_text
        stopped = "max_steps"

    return {
        "question": question,
        "final": final_text,
        "steps": steps,
        "stopped": stopped,
        "frame": str(frame_path) if frame_path else None,
        "elapsed_s": round(time.monotonic() - t_start, 3),
        "usage_total": usage_total,
    }


# ============================================================================
# FRONTIER ESCALATION (Sprint B) — Claude / GPT / Gemini
# ============================================================================

# Defaults. User can override per-call via the `model` arg.
CLAUDE_DEFAULT_MODEL = "claude-opus-4-8"
OPENAI_DEFAULT_MODEL = "gpt-4o"
GEMINI_DEFAULT_MODEL = "gemini-2.5-pro"


def _frame_b64(ctx: ToolContext) -> str | None:
    """Capture the current frame and return its base64 jpg (or None)."""
    if ctx is None:
        return None
    try:
        snap = _tmp_jpg(ctx, "escalate")
        _hd_capture(ctx, snap)
        return base64.b64encode(snap.read_bytes()).decode()
    except Exception:
        return None


def _get_key(provider: str) -> str | None:
    s = _load_secrets()
    key = (s.get(provider) or {}).get("api_key")
    if key:
        return key
    # also try env var fallback
    env_key = {"anthropic": "ANTHROPIC_API_KEY",
               "openai":    "OPENAI_API_KEY",
               "google":    "GOOGLE_API_KEY"}.get(provider)
    if env_key:
        return os.environ.get(env_key)
    return None


def _call_claude(question: str, model: str, max_tokens: int,
                 frame_b64: str | None) -> dict:
    key = _get_key("anthropic")
    if not key:
        raise ToolError("Anthropic API key missing. Save via: "
                        '{"anthropic": {"api_key": "sk-ant-..."}} in '
                        "~/.config/jarvis/keys.json")
    content: list = []
    if frame_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_b64,
            },
        })
    content.append({"type": "text", "text": question})
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    if r.status_code >= 400:
        raise ToolError(f"Anthropic {r.status_code}: {r.text[:500]}")
    j = r.json()
    blocks = j.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    usage = j.get("usage") or {}
    return {
        "provider": "anthropic", "model": model, "text": text.strip(),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "stop_reason": j.get("stop_reason"),
    }


def _call_openai(question: str, model: str, max_tokens: int,
                 frame_b64: str | None) -> dict:
    key = _get_key("openai")
    if not key:
        raise ToolError("OpenAI API key missing. Save via: "
                        '{"openai": {"api_key": "sk-..."}} in '
                        "~/.config/jarvis/keys.json")
    if frame_b64:
        user_content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
            {"type": "text", "text": question},
        ]
    else:
        user_content = question
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_content}],
        },
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    if r.status_code >= 400:
        raise ToolError(f"OpenAI {r.status_code}: {r.text[:500]}")
    j = r.json()
    text = (j.get("choices", [{}])[0].get("message", {}).get("content") or "")
    usage = j.get("usage") or {}
    return {
        "provider": "openai", "model": model, "text": text.strip(),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "stop_reason": j.get("choices", [{}])[0].get("finish_reason"),
    }


def _call_gemini(question: str, model: str, max_tokens: int,
                 frame_b64: str | None) -> dict:
    key = _get_key("google")
    if not key:
        raise ToolError("Google API key missing. Save via: "
                        '{"google": {"api_key": "..."}} in '
                        "~/.config/jarvis/keys.json")
    parts: list = []
    if frame_b64:
        parts.append({"inline_data":
                      {"mime_type": "image/jpeg", "data": frame_b64}})
    parts.append({"text": question})
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": parts}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        },
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    if r.status_code >= 400:
        raise ToolError(f"Gemini {r.status_code}: {r.text[:500]}")
    j = r.json()
    candidates = j.get("candidates") or [{}]
    parts_out = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts_out)
    usage = j.get("usageMetadata") or {}
    return {
        "provider": "google", "model": model, "text": text.strip(),
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "stop_reason": candidates[0].get("finishReason"),
    }


@tool(
    "ask_claude",
    description="Ask Anthropic Claude (default Opus 4.8) a question. Use "
                "when the local 3B VLM is not enough — long context, "
                "complex reasoning, vision detail, code review. "
                "Optionally include the camera frame.",
    category="cloud",
    schema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "model": {"type": "string"},
            "max_tokens": {"type": "integer"},
            "include_frame": {"type": "boolean"},
        },
        "required": ["question"],
    },
    needs_ctx=True,
)
def ask_claude(ctx, question: str, model: str = CLAUDE_DEFAULT_MODEL,
               max_tokens: int = 1024, include_frame: bool = False) -> dict:
    frame = _frame_b64(ctx) if include_frame else None
    return _call_claude(question, model, max(64, min(8192, int(max_tokens))),
                        frame)


@tool(
    "ask_gpt",
    description="Ask OpenAI GPT a question. Best for code-heavy queries. "
                "Optionally include the camera frame.",
    category="cloud",
    schema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "model": {"type": "string"},
            "max_tokens": {"type": "integer"},
            "include_frame": {"type": "boolean"},
        },
        "required": ["question"],
    },
    needs_ctx=True,
)
def ask_gpt(ctx, question: str, model: str = OPENAI_DEFAULT_MODEL,
            max_tokens: int = 1024, include_frame: bool = False) -> dict:
    frame = _frame_b64(ctx) if include_frame else None
    return _call_openai(question, model, max(64, min(8192, int(max_tokens))),
                        frame)


@tool(
    "ask_gemini",
    description="Ask Google Gemini a question. Best for grounded web answers. "
                "Optionally include the camera frame.",
    category="cloud",
    schema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "model": {"type": "string"},
            "max_tokens": {"type": "integer"},
            "include_frame": {"type": "boolean"},
        },
        "required": ["question"],
    },
    needs_ctx=True,
)
def ask_gemini(ctx, question: str, model: str = GEMINI_DEFAULT_MODEL,
               max_tokens: int = 1024, include_frame: bool = False) -> dict:
    frame = _frame_b64(ctx) if include_frame else None
    return _call_gemini(question, model, max(64, min(8192, int(max_tokens))),
                        frame)


@tool(
    "escalate",
    description="Route a hard question to a frontier model. Default order: "
                "Claude (best general) -> GPT (code) -> Gemini (web). Pass "
                "`prefer` to force a provider.",
    category="cloud",
    schema={
        "type": "object",
        "properties": {
            "question":   {"type": "string"},
            "prefer":     {"type": "string",
                           "description": "claude | gpt | gemini"},
            "max_tokens": {"type": "integer"},
            "include_frame": {"type": "boolean"},
        },
        "required": ["question"],
    },
    needs_ctx=True,
)
def escalate(ctx, question: str, prefer: str = "claude",
             max_tokens: int = 1024, include_frame: bool = False) -> dict:
    order = ["claude", "gpt", "gemini"]
    p = (prefer or "").strip().lower()
    if p in order:
        order.remove(p); order.insert(0, p)
    frame = _frame_b64(ctx) if include_frame else None
    max_tokens = max(64, min(8192, int(max_tokens)))
    tried: list = []
    for which in order:
        try:
            if which == "claude":
                return _call_claude(question, CLAUDE_DEFAULT_MODEL,
                                    max_tokens, frame)
            if which == "gpt":
                return _call_openai(question, OPENAI_DEFAULT_MODEL,
                                    max_tokens, frame)
            return _call_gemini(question, GEMINI_DEFAULT_MODEL,
                                max_tokens, frame)
        except ToolError as e:
            tried.append({"provider": which, "error": str(e)})
            continue
    raise ToolError(
        "no frontier provider available. tried: " + json.dumps(tried)
    )


_VISION_HINTS = re.compile(
    r"\b(this|that|these|those|here|see|look(ing)?|show|"
    r"in (front|the )?(of|frame|photo|image|view|scene|shot)|"
    r"camera|object|sign|label|can|bottle|barcode|shirt|jacket|"
    r"storefront|poster|book cover|product|what'?s on the|"
    r"i'?m looking at|in my hand|on the (shelf|table|wall|floor))\b",
    re.I,
)


def question_implies_vision(q: str) -> bool:
    """True if the question seems to reference what the camera sees."""
    if not q:
        return False
    return bool(_VISION_HINTS.search(q))


def run_agent(question: str, *, max_steps: int = 3,
              allowlist: set | None = None,
              use_frame=None) -> dict:
    """Sugar for callers that already populated the registry context.

    use_frame defaults to False. For vision-y questions the agent is
    expected to CALL a vision tool (research_visual / read_all_text /
    zoom_into) rather than receive the image directly — at the 3B model
    scale, bundling an image into the agent prompt crowds out tool
    selection and the model emits whitespace.
    """
    if TOOLS._ctx is None:
        return {"error": "tool context not initialised"}
    if use_frame is None or use_frame == "auto":
        use_frame = False
    return agentic_loop(
        TOOLS._ctx, question,
        max_steps=max_steps, allowlist=allowlist, use_frame=bool(use_frame),
    )


# ============================================================================
# SECRETS VAULT  (~/.config/jarvis/keys.json, 0600)
# ============================================================================

SECRETS_PATH = Path.home() / ".config" / "jarvis" / "keys.json"


def _load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        return {}
    try:
        return json.loads(SECRETS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_secrets(data: dict) -> None:
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SECRETS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(SECRETS_PATH)


# ============================================================================
# PERSISTENT IDENTITY / PROFILE (Sprint D)
# ============================================================================

PROFILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


def _profile_dict(memory) -> dict:
    with memory.lock:
        memory._con.executescript(PROFILE_SCHEMA)
        rows = memory._con.execute(
            "SELECT key, value FROM profile",
        ).fetchall()
    return {k: v for k, v in rows}


def profile_prompt_block(memory) -> str:
    """Render the profile as a system prompt section."""
    p = _profile_dict(memory)
    if not p:
        return ""
    lines = ["USER PROFILE (what you know about the user):"]
    for k, v in sorted(p.items()):
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n\n"


@tool(
    "profile_get",
    description="Show what Jarvis has learned about the user (name, "
                "preferences, recurring contexts).",
    category="memory",
    needs_ctx=True,
)
def profile_get(ctx) -> dict:
    return {"profile": _profile_dict(ctx.memory)}


@tool(
    "profile_set",
    description="Set a single profile fact (e.g. key='name' value='Steffen', "
                "key='location' value='Pomona NY', key='commute' value='by bike'). "
                "Gated.",
    category="memory",
    safety="gated",
    schema={
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["key", "value"],
    },
    needs_ctx=True,
)
def profile_set(ctx, key: str, value: str) -> dict:
    key = (key or "").strip().lower().replace(" ", "_")
    value = (value or "").strip()
    if not key or not value:
        raise ToolError("key and value required")
    with ctx.memory.lock:
        ctx.memory._con.executescript(PROFILE_SCHEMA)
        ctx.memory._con.execute(
            "INSERT OR REPLACE INTO profile (key, value, updated_at) "
            "VALUES (?,?,?)",
            (key, value, _now()),
        )
    return {"key": key, "value": value}


@tool(
    "profile_unset",
    description="Remove a profile fact. Gated.",
    category="memory",
    safety="gated",
    schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    needs_ctx=True,
)
def profile_unset(ctx, key: str) -> dict:
    with ctx.memory.lock:
        ctx.memory._con.executescript(PROFILE_SCHEMA)
        ctx.memory._con.execute(
            "DELETE FROM profile WHERE key=?", ((key or "").strip().lower(),),
        )
    return {"deleted": key}


# ============================================================================
# HUE SMART HOME (Tier 3 — Iron Man moment, no pip deps, no cloud)
# ============================================================================

HUE_DEFAULT_BRIDGE = "192.168.86.71"

_NAMED_COLORS_XY = {
    "red":    [0.6750, 0.3220],
    "orange": [0.6000, 0.3800],
    "amber":  [0.5470, 0.4140],
    "yellow": [0.4400, 0.4900],
    "green":  [0.1700, 0.7000],
    "cyan":   [0.1700, 0.3300],
    "blue":   [0.1500, 0.0600],
    "purple": [0.2700, 0.1300],
    "pink":   [0.5000, 0.3000],
    "white":  [0.3127, 0.3290],
}
_NAMED_COLORS_CT = {
    "warm":     500,   # ~2000K
    "warmish":  400,
    "neutral":  300,
    "cool":     200,
    "daylight": 153,   # ~6500K
}


def _hue_get_secrets() -> tuple[str, str | None]:
    s = _load_secrets().get("hue") or {}
    return s.get("bridge_ip") or HUE_DEFAULT_BRIDGE, s.get("app_key")


def _hue_save(bridge_ip: str, app_key: str) -> None:
    s = _load_secrets()
    s["hue"] = {"bridge_ip": bridge_ip, "app_key": app_key}
    _save_secrets(s)


def _hue_url(path: str) -> str:
    bridge, key = _hue_get_secrets()
    if key is None:
        raise ToolError(
            "Hue bridge not paired. Press the link button on the bridge, "
            "then call hue_pair() within 30 seconds."
        )
    return f"http://{bridge}/api/{key}{path}"


def _hue_request(method: str, path: str, body: dict | None = None) -> dict:
    url = _hue_url(path)
    try:
        if method == "GET":
            r = HTTP.get(url)
        elif method == "PUT":
            r = HTTP.put(url, json=body or {})
        elif method == "POST":
            r = HTTP.post(url, json=body or {})
        else:
            raise ToolError(f"bad method: {method}")
    except httpx.RequestError as e:
        raise ToolError(f"Hue bridge unreachable: {e}")
    if r.status_code >= 400:
        raise ToolError(f"Hue HTTP {r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except json.JSONDecodeError:
        return {"raw": r.text[:500]}


def _resolve_group(name_or_id: str) -> str:
    """Return a Hue group id. 'all' = 0. Else lookup by name."""
    n = (name_or_id or "").strip().lower()
    if not n or n in ("all", "everywhere", "0"):
        return "0"
    if n.isdigit():
        return n
    groups = _hue_request("GET", "/groups")
    for gid, info in groups.items():
        if (info.get("name") or "").lower() == n:
            return gid
    raise ToolError(
        f"unknown Hue group: {name_or_id}. "
        f"available: {sorted((g.get('name') or '?') for g in groups.values())}"
    )


def _light_state(brightness: float | None, color: str | None) -> dict:
    state: dict = {"on": True}
    if brightness is not None:
        b = max(0, min(100, int(brightness)))
        state["bri"] = max(1, int(b * 254 / 100))
    if color:
        c = color.strip().lower()
        if c in _NAMED_COLORS_XY:
            state["xy"] = _NAMED_COLORS_XY[c]
        elif c in _NAMED_COLORS_CT:
            state["ct"] = _NAMED_COLORS_CT[c]
        elif c.startswith("#") and len(c) == 7:
            # naive hex -> xy via sRGB midpoint
            r = int(c[1:3], 16) / 255.0
            g = int(c[3:5], 16) / 255.0
            b_ = int(c[5:7], 16) / 255.0
            X = 0.4124 * r + 0.3576 * g + 0.1805 * b_
            Y = 0.2126 * r + 0.7152 * g + 0.0722 * b_
            Z = 0.0193 * r + 0.1192 * g + 0.9505 * b_
            s = X + Y + Z or 1.0
            state["xy"] = [X / s, Y / s]
        else:
            raise ToolError(
                f"unknown color '{color}'. "
                f"try: {sorted(list(_NAMED_COLORS_XY) + list(_NAMED_COLORS_CT))}"
            )
    return state


@tool(
    "hue_pair",
    description="Pair Jarvis with the Hue bridge. FIRST press the link "
                "button on top of the bridge, THEN call this within 30 "
                "seconds. App key is stored in ~/.config/jarvis/keys.json.",
    category="web",
    safety="gated",
    schema={
        "type": "object",
        "properties": {
            "bridge_ip": {"type": "string", "description": "default 192.168.86.71"},
        },
    },
)
def hue_pair(bridge_ip: str = HUE_DEFAULT_BRIDGE) -> dict:
    bridge = (bridge_ip or HUE_DEFAULT_BRIDGE).strip()
    try:
        r = HTTP.post(
            f"http://{bridge}/api",
            json={"devicetype": "jarvis-lab#jetson"},
        )
    except httpx.RequestError as e:
        raise ToolError(f"bridge unreachable: {e}")
    try:
        arr = r.json()
    except json.JSONDecodeError:
        raise ToolError(f"bad bridge response: {r.text[:300]}")
    if not arr or not isinstance(arr, list):
        raise ToolError(f"unexpected response: {arr}")
    first = arr[0]
    if "error" in first:
        msg = first["error"].get("description") or str(first["error"])
        if "link button" in msg.lower():
            raise ToolError(
                "Press the link button on the Hue bridge, then re-run hue_pair."
            )
        raise ToolError(f"Hue error: {msg}")
    key = first.get("success", {}).get("username")
    if not key:
        raise ToolError(f"no username in response: {arr}")
    _hue_save(bridge, key)
    return {"bridge_ip": bridge, "paired": True,
            "key_path": str(SECRETS_PATH)}


@tool(
    "hue_lights",
    description="List all Hue lights and groups with current state.",
    category="web",
)
def hue_lights() -> dict:
    lights = _hue_request("GET", "/lights")
    groups = _hue_request("GET", "/groups")
    return {
        "lights": {lid: {"name": l.get("name"),
                         "on": l.get("state", {}).get("on"),
                         "bri": l.get("state", {}).get("bri"),
                         "reachable": l.get("state", {}).get("reachable")}
                   for lid, l in lights.items()},
        "groups": {gid: {"name": g.get("name"),
                         "lights": g.get("lights", []),
                         "any_on": (g.get("state") or {}).get("any_on")}
                   for gid, g in groups.items()},
    }


@tool(
    "lights_state",
    description="Get the current state of a light group (default: all).",
    category="web",
    schema={
        "type": "object",
        "properties": {"group": {"type": "string"}},
    },
)
def lights_state(group: str = "all") -> dict:
    gid = _resolve_group(group)
    info = _hue_request("GET", f"/groups/{gid}")
    return {
        "group": group, "gid": gid,
        "any_on": (info.get("state") or {}).get("any_on"),
        "all_on": (info.get("state") or {}).get("all_on"),
        "brightness": (info.get("action") or {}).get("bri"),
        "name": info.get("name"),
    }


@tool(
    "lights_on",
    description="Turn on a Hue group. Optionally set brightness (0-100) "
                "and color (named: warm/cool/red/blue/etc, or #RRGGBB).",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "group":      {"type": "string"},
            "brightness": {"type": "number"},
            "color":      {"type": "string"},
        },
    },
)
def lights_on(group: str = "all", brightness: float | None = None,
              color: str | None = None) -> dict:
    gid = _resolve_group(group)
    state = _light_state(brightness, color)
    state["on"] = True
    res = _hue_request("PUT", f"/groups/{gid}/action", state)
    return {"group": group, "gid": gid, "state": state, "response": res}


@tool(
    "lights_off",
    description="Turn off a Hue group.",
    category="web",
    schema={
        "type": "object",
        "properties": {"group": {"type": "string"}},
    },
)
def lights_off(group: str = "all") -> dict:
    gid = _resolve_group(group)
    res = _hue_request("PUT", f"/groups/{gid}/action", {"on": False})
    return {"group": group, "gid": gid, "response": res}


@tool(
    "lights_set",
    description="Adjust brightness and/or color of a Hue group without "
                "changing on/off state.",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "group":      {"type": "string"},
            "brightness": {"type": "number"},
            "color":      {"type": "string"},
        },
    },
)
def lights_set(group: str = "all", brightness: float | None = None,
               color: str | None = None) -> dict:
    if brightness is None and color is None:
        raise ToolError("brightness or color required")
    gid = _resolve_group(group)
    state = _light_state(brightness, color)
    state.pop("on", None)
    res = _hue_request("PUT", f"/groups/{gid}/action", state)
    return {"group": group, "gid": gid, "state": state, "response": res}


@tool(
    "scene_activate",
    description="Activate a Hue scene by name.",
    category="web",
    schema={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
)
def scene_activate(name: str) -> dict:
    scenes = _hue_request("GET", "/scenes")
    needle = (name or "").strip().lower()
    target = None
    for sid, s in scenes.items():
        if (s.get("name") or "").lower() == needle:
            target = sid; break
    if target is None:
        # also try partial match
        for sid, s in scenes.items():
            if needle in (s.get("name") or "").lower():
                target = sid; break
    if target is None:
        raise ToolError(
            f"no scene matching '{name}'. "
            f"have: {sorted({(s.get('name') or '?') for s in scenes.values()})}"
        )
    res = _hue_request("PUT", "/groups/0/action", {"scene": target})
    return {"scene": name, "scene_id": target, "response": res}


@tool(
    "lights_pulse",
    description="Briefly flash a Hue group a color, then restore. Useful "
                "as a silent notification.",
    category="web",
    schema={
        "type": "object",
        "properties": {
            "group": {"type": "string"},
            "color": {"type": "string"},
            "ms":    {"type": "integer", "description": "flash duration"},
        },
    },
)
def lights_pulse(group: str = "all", color: str = "blue",
                 ms: int = 600) -> dict:
    gid = _resolve_group(group)
    ms = max(100, min(5000, int(ms)))
    # snapshot previous state for restore
    prev = _hue_request("GET", f"/groups/{gid}")
    prev_action = prev.get("action") or {}
    flash = _light_state(80, color)
    flash["on"] = True
    _hue_request("PUT", f"/groups/{gid}/action", flash)

    def _restore():
        time.sleep(ms / 1000.0)
        restore: dict = {}
        for k in ("on", "bri", "xy", "ct"):
            if k in prev_action:
                restore[k] = prev_action[k]
        if restore:
            try:
                _hue_request("PUT", f"/groups/{gid}/action", restore)
            except Exception:
                pass

    threading.Thread(target=_restore, daemon=True).start()
    return {"group": group, "color": color, "ms": ms}
