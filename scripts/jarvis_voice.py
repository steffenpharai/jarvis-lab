#!/usr/bin/env python3
"""Jarvis dashboard on Jetson Orin Nano Super.

Round 1+2 roadmap shipped:
  - SQLite persistence (every turn saved, full-text search over history)
  - Sentence-streaming TTS (Piper synth per-sentence as VLM tokens arrive;
    Web Audio API gapless playback with sentence highlighting)
  - openWakeWord "Hey Jarvis" listener with shared ALSA AudioBus so wake
    detection coexists with on-demand recording
  - Perceptual-hash frame cache (Hamming distance gate) for repeat-question
    follow-ups about the same scene
  - Continuous live narration (auto-snap + describe every N seconds,
    deadlock-free)
  - SSE streaming turns with phase + token deltas + cancellation
  - Frontier-grade UX: rust accent + Geist typography + Lucide SVG +
    glass HUD pills + breathing voice orb + wake-listening ring +
    sentence-highlighted playback + scene-change pulse + Memory tab
"""
from __future__ import annotations

import base64
import collections
import contextlib
import json
import os
import queue
import io
import re
import shutil
import sqlite3
import struct
import subprocess
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx              # OpenAI / Anthropic SDK pattern: streaming HTTP
import numpy as np        # audio buffer math + pHash
from PIL import Image     # in-process JPEG decode (fork-free under the 8GB ceiling)

import jarvis_tools       # tool catalog (registry + handlers)

# ----- paths + constants ------------------------------------------------------
LAB = Path("/home/zip/jarvis-lab")
WHISPER_BIN   = LAB / "build/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = LAB / "build/whisper.cpp/models/ggml-tiny.en.bin"
PIPER_BIN     = LAB / "piper/piper/piper"
PIPER_VOICE   = LAB / "piper/voices/en_GB-alan-medium.onnx"  # British male (Jarvis)
VLM_URL       = "http://127.0.0.1:8080/v1/chat/completions"
VLM_HEALTH    = "http://127.0.0.1:8080/health"
DB_PATH       = LAB / "logs/jarvis.db"

MIC_DEVICE = "plughw:CARD=C615,DEV=0"
CAM_DEVICE = "/dev/video0"
# 1280x720 MJPEG (the C615 supports it): sharper investigate/zoom crops and a
# crisper feed. VLM turns stay fast — capture_frame_for_vlm still downscales to
# VLM_W x VLM_H (512x384) before inference; only crop detail improves.
CAM_W, CAM_H = 1280, 720
VLM_W, VLM_H = 512, 384
CAM_FPS = 10
SAMPLE_RATE = 16000
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8085
MAX_HISTORY_PAIRS = 3
METRICS_TTL = 2.0

# wake word
WAKE_MODEL_NAME = "hey_jarvis_v0.1"
WAKE_THRESHOLD = 0.55
WAKE_COOLDOWN_S = 3.0   # don't fire again for N seconds after detection
WAKE_REFRACT_FRAMES = 6  # require sustained detection across N inference frames

# scene-change gate
PHASH_REUSE_DIST = 6    # if Hamming distance <= this, reuse cached frame
PHASH_LIVE_GATE = 4     # live mode skips broadcast unless distance > this

SESSION_DIR = LAB / "logs/sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
VMEM_DIR = SESSION_DIR / "vmem"
VMEM_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = SESSION_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "did", "do", "does", "you",
    "see", "saw", "seen", "what", "where", "when", "last", "my", "me", "i",
    "it", "that", "this", "there", "have", "had", "any", "some", "of", "in",
    "on", "at", "to", "and", "or", "for", "with", "show", "find", "look",
}

# Updated whenever the user actively uses the VLM, so the ambient visual-memory
# captioner can yield the GPU to interactive turns.
LAST_USER_ACTIVITY = {"ts": 0.0}
_BOOT_TS = time.time()


# ----- prompt presets ---------------------------------------------------------
PRESETS = {
    "jarvis": (
        "You are J.A.R.V.I.S., the user's personal AI — in the spirit of Tony "
        "Stark's assistant. You see what the camera sees and hear what the user "
        "says.\n\n"
        "PERSONA:\n"
        "- Refined, composed, and unfailingly loyal. Address the user as 'sir' "
        "naturally — not in every sentence, just where it lands.\n"
        "- Dry, understated British wit: a touch of irony, never goofy, never "
        "over-explaining.\n"
        "- Anticipatory and precise. Get to the point and volunteer the one "
        "genuinely useful detail, briefly.\n"
        "- Unflappable — calm, quiet confidence, even when something is wrong.\n\n"
        "GROUND RULES (accuracy matters):\n"
        "1. Describe only what you can actually see. Even dim scenes have "
        "shapes, colours, and objects — describe them.\n"
        "2. Do NOT invent text, brand names, or model numbers. If a label isn't "
        "clearly legible, say so rather than guessing — a blurry box is not "
        "'Skittles'.\n"
        "3. Confident about general observations; careful about specific "
        "identities. Name uncertainty instead of refusing.\n\n"
        "STYLE: you are spoken aloud — write the way you would speak, no "
        "markdown symbols or bullet characters. Under ~45 words unless asked "
        "for more. Always in character.\n\n"
        "Example of your voice —\n"
        "User: \"Hey Jarvis, what do you see?\"\n"
        "You: \"Quite the command centre, sir — a racing-style chair, a desk "
        "rather well stocked with soft drinks, and a respectable amount of "
        "laundry holding the fort behind you.\"\n"
        "User: \"What's the chair?\"\n"
        "You: \"A GTRacing gaming chair, by the look of the logo. Built for "
        "long campaigns, sir.\""
    ),
    "focused": (
        "You are Jarvis, the user's personal AI agent. You see what the "
        "camera sees and you hear what the user says.\n\n"
        "GROUND RULES:\n"
        "1. Describe what you can actually see. Even dimly lit scenes have "
        "visible shapes, colors, objects, and spatial layouts — describe them. "
        "Only refuse if the image is genuinely fully black, blank, or "
        "unintelligible.\n"
        "2. Do NOT invent specific text, signs, numbers, brand names, or model "
        "numbers. If you can't clearly read text, say 'I don't see any "
        "readable text'.\n"
        "3. For counts, prefer approximate language ('a few', 'several').\n"
        "4. Be confident about general observations; cautious about specific "
        "identities and labels.\n"
        "5. If genuinely uncertain, name the uncertainty rather than refusing.\n\n"
        "Style: brief, conversational, like a person describing what they "
        "see. Use markdown for lists/bold only when it helps. Under 60 words "
        "unless asked for more."
    ),
    "inspector": (
        "You are Jarvis in Inspector mode — a forensic visual analyst. Lead "
        "with a one-line summary, then a bulleted breakdown: subjects, "
        "environment, lighting, notable details. Precise language, no filler. "
        "Never invent text, brands, or counts you can't verify. Note "
        "anything anomalous. Under 100 words."
    ),
    "companion": (
        "You are Jarvis, a warm conversational companion. The user is wearing "
        "you in a backpack and walks through the world with you. Friendly, "
        "natural, like a thoughtful friend. Show curiosity. Don't fabricate "
        "text, brands, or specific identities. Name uncertainty in passing. "
        "Under 50 words."
    ),
    "curator": (
        "You are Jarvis in Curator mode — caption like a museum or magazine "
        "writer. Brief, evocative, observational, present-tense. Sensory and "
        "specific where you can be (color, light, texture, spatial "
        "relationships). Don't fabricate text or identities. Under 60 words."
    ),
}
DEFAULT_SYSTEM_PROMPT = PRESETS["jarvis"]

SETTINGS = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "preset": "jarvis",
    "max_tokens": 240,
    "temperature": 0.2,
    "record_seconds": 6,
    "live_interval_s": 8,
    "wake_word_enabled": False,
    "scene_cache_enabled": True,
    "agent_mode_enabled": False,
    "agent_max_steps": 3,
}
SETTINGS_LOCK = threading.Lock()


# ----- SQLite persistence -----------------------------------------------------
class Memory:
    """Persistent conversation history with FTS5 search."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id TEXT UNIQUE NOT NULL,
        kind TEXT NOT NULL,
        question TEXT,
        transcription TEXT,
        reply TEXT,
        frame_path TEXT,
        audio_path TEXT,
        timings_json TEXT,
        cancelled INTEGER DEFAULT 0,
        created_at REAL NOT NULL,
        is_pinned INTEGER DEFAULT 0,
        note TEXT,
        agent_meta TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_turns_pinned ON turns(is_pinned);
    CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts
        USING fts5(turn_id, question, reply, content='turns',
                   content_rowid='id', tokenize='porter');
    CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
        INSERT INTO turns_fts(rowid, turn_id, question, reply)
            VALUES (new.id, new.turn_id, new.question, new.reply);
    END;
    CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
        INSERT INTO turns_fts(turns_fts, rowid, turn_id, question, reply)
            VALUES('delete', old.id, old.turn_id, old.question, old.reply);
    END;

    -- Spatial-temporal visual memory ("world model"): a persistent, searchable
    -- log of what the camera has seen over time. Keyframes are captioned by the
    -- VLM and indexed (FTS now; embedding column reserved for vector recall).
    CREATE TABLE IF NOT EXISTS visual_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        caption TEXT,
        objects TEXT,
        frame_file TEXT,
        phash INTEGER,
        source TEXT,
        embedding BLOB
    );
    CREATE INDEX IF NOT EXISTS idx_vmem_ts ON visual_memory(ts DESC);
    CREATE VIRTUAL TABLE IF NOT EXISTS vmem_fts
        USING fts5(caption, objects, content='visual_memory',
                   content_rowid='id', tokenize='porter');
    CREATE TRIGGER IF NOT EXISTS vmem_ai AFTER INSERT ON visual_memory BEGIN
        INSERT INTO vmem_fts(rowid, caption, objects)
            VALUES (new.id, new.caption, new.objects);
    END;
    CREATE TRIGGER IF NOT EXISTS vmem_ad AFTER DELETE ON visual_memory BEGIN
        INSERT INTO vmem_fts(vmem_fts, rowid, caption, objects)
            VALUES('delete', old.id, old.caption, old.objects);
    END;

    -- Proactive "watch" rules: natural-language conditions the VLM evaluates on
    -- scene-change keyframes and alerts on (ambient-agent monitoring).
    CREATE TABLE IF NOT EXISTS watch_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        active INTEGER DEFAULT 1,
        created_at REAL NOT NULL,
        last_fired REAL DEFAULT 0,
        fire_count INTEGER DEFAULT 0
    );
    """

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self._con = sqlite3.connect(str(path), check_same_thread=False,
                                    isolation_level=None)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.executescript(self.SCHEMA)
        # idempotent migration for pre-existing DBs
        try:
            self._con.execute("ALTER TABLE turns ADD COLUMN agent_meta TEXT")
        except sqlite3.OperationalError:
            pass

    def record(self, turn: dict) -> None:
        agent_meta_json = ""
        if turn.get("agent_mode") or turn.get("agent_steps"):
            agent_meta_json = json.dumps({
                "agent_mode": bool(turn.get("agent_mode")),
                "agent_steps": turn.get("agent_steps") or [],
            }, default=str)
        with self.lock:
            self._con.execute(
                """INSERT OR REPLACE INTO turns
                   (turn_id, kind, question, transcription, reply,
                    frame_path, audio_path, timings_json, cancelled,
                    created_at, agent_meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    turn["turn_id"], turn["kind"],
                    turn.get("question") or "",
                    turn.get("transcription") or "",
                    turn.get("reply") or "",
                    turn.get("frame_url") or "",
                    turn.get("audio_url") or "",
                    json.dumps(turn.get("timings") or {}),
                    1 if turn.get("cancelled") else 0,
                    turn.get("ts") or time.time(),
                    agent_meta_json,
                ),
            )

    def recent(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self._con.execute(
                """SELECT turn_id, kind, question, transcription, reply,
                          frame_path, audio_path, timings_json, cancelled,
                          created_at, is_pinned, note, agent_meta
                   FROM turns ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def search(self, q: str, limit: int = 50) -> list[dict]:
        if not q.strip():
            return self.recent(limit)
        with self.lock:
            try:
                rows = self._con.execute(
                    """SELECT t.turn_id, t.kind, t.question, t.transcription,
                              t.reply, t.frame_path, t.audio_path,
                              t.timings_json, t.cancelled, t.created_at,
                              t.is_pinned, t.note, t.agent_meta
                       FROM turns_fts JOIN turns t ON t.id = turns_fts.rowid
                       WHERE turns_fts MATCH ?
                       ORDER BY t.created_at DESC LIMIT ?""",
                    (q + "*", limit),
                ).fetchall()
            except sqlite3.OperationalError:
                pat = f"%{q}%"
                rows = self._con.execute(
                    """SELECT turn_id, kind, question, transcription, reply,
                              frame_path, audio_path, timings_json, cancelled,
                              created_at, is_pinned, note, agent_meta
                       FROM turns
                       WHERE question LIKE ? OR reply LIKE ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (pat, pat, limit),
                ).fetchall()
        return [self._row(r) for r in rows]

    def set_pin(self, turn_id: str, pinned: bool) -> None:
        with self.lock:
            self._con.execute(
                "UPDATE turns SET is_pinned=? WHERE turn_id=?",
                (1 if pinned else 0, turn_id),
            )

    def pinned(self) -> list[dict]:
        with self.lock:
            rows = self._con.execute(
                """SELECT turn_id, kind, question, transcription, reply,
                          frame_path, audio_path, timings_json, cancelled,
                          created_at, is_pinned, note, agent_meta
                   FROM turns WHERE is_pinned=1
                   ORDER BY created_at DESC""",
            ).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        with self.lock:
            return self._con.execute("SELECT COUNT(*) FROM turns").fetchone()[0]

    # --- visual memory (spatial-temporal "world model") ------------------
    def vmem_record(self, ts: float, caption: str, objects: str,
                    frame_file: str, phash: int, source: str) -> int:
        # pHash is unsigned 64-bit; SQLite INTEGER is signed 64-bit. Fold.
        if phash is not None and phash >= (1 << 63):
            phash -= (1 << 64)
        with self.lock:
            cur = self._con.execute(
                """INSERT INTO visual_memory
                   (ts, caption, objects, frame_file, phash, source)
                   VALUES (?,?,?,?,?,?)""",
                (ts, caption, objects, frame_file, phash, source),
            )
            return cur.lastrowid

    def vmem_recent(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self._con.execute(
                "SELECT id, ts, caption, objects, frame_file, source "
                "FROM visual_memory ORDER BY ts DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self._vrow(r) for r in rows]

    def vmem_search(self, q: str, limit: int = 30) -> list[dict]:
        q = (q or "").strip()
        if not q:
            return self.vmem_recent(limit)
        # Build a tolerant FTS query: OR the significant tokens, prefix-match.
        toks = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in q.split()]
        toks = [t for t in toks if len(t) > 2 and t not in _STOPWORDS]
        match = " OR ".join(f"{t}*" for t in toks) if toks else q
        with self.lock:
            try:
                rows = self._con.execute(
                    "SELECT v.id, v.ts, v.caption, v.objects, v.frame_file, "
                    "v.source FROM vmem_fts f JOIN visual_memory v "
                    "ON v.id = f.rowid WHERE vmem_fts MATCH ? "
                    "ORDER BY rank LIMIT ?", (match, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                pat = f"%{q}%"
                rows = self._con.execute(
                    "SELECT id, ts, caption, objects, frame_file, source "
                    "FROM visual_memory WHERE caption LIKE ? OR objects LIKE ? "
                    "ORDER BY ts DESC LIMIT ?", (pat, pat, limit),
                ).fetchall()
        return [self._vrow(r) for r in rows]

    def vmem_count(self) -> int:
        with self.lock:
            return self._con.execute(
                "SELECT COUNT(*) FROM visual_memory").fetchone()[0]

    def vmem_entities(self, scan_rows: int = 150) -> list[dict]:
        """Object-centric registry: aggregate the captioner's per-keyframe object
        lists into distinct entities with sighting count + last-seen + a frame."""
        with self.lock:
            rows = self._con.execute(
                "SELECT ts, objects, frame_file FROM visual_memory "
                "WHERE objects != '' ORDER BY ts DESC LIMIT ?", (scan_rows,),
            ).fetchall()
        agg: dict[str, dict] = {}
        for ts, objs, frame in rows:
            seen_here = set()
            for o in (objs or "").split(","):
                label = re.sub(r"[^a-z0-9 ]", "", o.strip().lower()).strip()
                if len(label) < 2 or label in seen_here:
                    continue
                seen_here.add(label)
                e = agg.get(label)
                furl = f"/vmem/{frame}" if frame else ""
                if e is None:
                    agg[label] = {"label": label, "count": 1,
                                  "last_ts": ts, "frame_url": furl}
                else:
                    e["count"] += 1
                    if ts > e["last_ts"]:
                        e["last_ts"] = ts
                        e["frame_url"] = furl
        return sorted(agg.values(), key=lambda e: e["last_ts"], reverse=True)

    def vmem_graph(self, scan_rows: int = 300, top_n: int = 30,
                   min_edge: int = 2) -> dict:
        """Co-occurrence link graph: nodes = top entities, edges = how often two
        entities appeared in the same keyframe. Powers the Palantir link-chart."""
        with self.lock:
            rows = self._con.execute(
                "SELECT objects FROM visual_memory WHERE objects != '' "
                "ORDER BY ts DESC LIMIT ?", (scan_rows,),
            ).fetchall()
        node_count: dict[str, int] = {}
        edge_w: dict[tuple, int] = {}
        for (objs,) in rows:
            labs = sorted({re.sub(r"[^a-z0-9 ]", "", o.strip().lower()).strip()
                           for o in (objs or "").split(",")})
            labs = [l for l in labs if len(l) >= 2]
            for l in labs:
                node_count[l] = node_count.get(l, 0) + 1
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    k = (labs[i], labs[j])
                    edge_w[k] = edge_w.get(k, 0) + 1
        top = {l for l, _ in sorted(node_count.items(),
                                    key=lambda x: -x[1])[:top_n]}
        nodes = [{"id": l, "count": node_count[l]} for l in top]
        edges = [{"a": a, "b": b, "w": w} for (a, b), w in edge_w.items()
                 if a in top and b in top and w >= min_edge]
        return {"nodes": nodes, "edges": edges}

    def entity_detail(self, label: str, scan_rows: int = 400) -> dict:
        """Object-centric dossier for one entity: sightings over time + the
        entities it co-occurs with (the Palantir 'links' between objects)."""
        label = re.sub(r"[^a-z0-9 ]", "", (label or "").lower()).strip()
        with self.lock:
            rows = self._con.execute(
                "SELECT ts, objects, frame_file, caption FROM visual_memory "
                "WHERE objects != '' ORDER BY ts DESC LIMIT ?", (scan_rows,),
            ).fetchall()
        sightings: list[dict] = []
        related: dict[str, int] = {}
        first_ts = last_ts = None
        count = 0
        last_caption = ""
        for ts, objs, frame, caption in rows:
            labs = set()
            for o in (objs or "").split(","):
                lab = re.sub(r"[^a-z0-9 ]", "", o.strip().lower()).strip()
                if len(lab) >= 2:
                    labs.add(lab)
            if label not in labs:
                continue
            count += 1
            if last_ts is None:
                last_ts = ts
                last_caption = caption or ""
            first_ts = ts
            if len(sightings) < 16 and frame:
                sightings.append({"ts": ts, "frame_url": f"/vmem/{frame}"})
            for lab in labs:
                if lab != label:
                    related[lab] = related.get(lab, 0) + 1
        rel = sorted(related.items(), key=lambda x: -x[1])[:10]
        return {
            "label": label, "count": count,
            "first_ts": first_ts, "last_ts": last_ts,
            "last_caption": last_caption,
            "sightings": sightings,
            "related": [{"label": k, "count": v} for k, v in rel],
        }

    @staticmethod
    def _vrow(r) -> dict:
        return {"id": r[0], "ts": r[1], "caption": r[2] or "",
                "objects": r[3] or "", "frame_url": f"/vmem/{r[4]}" if r[4] else "",
                "source": r[5] or ""}

    # --- watch rules (proactive monitoring) ------------------------------
    def watch_add(self, text: str) -> int:
        with self.lock:
            cur = self._con.execute(
                "INSERT INTO watch_rules (text, active, created_at) "
                "VALUES (?,1,?)", (text, time.time()))
            return cur.lastrowid

    def watch_list(self, only_active: bool = False) -> list[dict]:
        sql = ("SELECT id, text, active, created_at, last_fired, fire_count "
               "FROM watch_rules")
        if only_active:
            sql += " WHERE active=1"
        sql += " ORDER BY created_at DESC"
        with self.lock:
            rows = self._con.execute(sql).fetchall()
        return [{"id": r[0], "text": r[1], "active": bool(r[2]),
                 "created_at": r[3], "last_fired": r[4], "fire_count": r[5]}
                for r in rows]

    def watch_set_active(self, rid: int, active: bool) -> None:
        with self.lock:
            self._con.execute("UPDATE watch_rules SET active=? WHERE id=?",
                              (1 if active else 0, rid))

    def watch_remove(self, rid: int) -> None:
        with self.lock:
            self._con.execute("DELETE FROM watch_rules WHERE id=?", (rid,))

    def watch_mark_fired(self, rid: int) -> None:
        with self.lock:
            self._con.execute(
                "UPDATE watch_rules SET last_fired=?, fire_count=fire_count+1 "
                "WHERE id=?", (time.time(), rid))

    @staticmethod
    def _row(r) -> dict:
        meta = {}
        if len(r) > 12 and r[12]:
            try:
                meta = json.loads(r[12])
            except (TypeError, ValueError):
                meta = {}
        return {
            "turn_id": r[0], "kind": r[1], "question": r[2],
            "transcription": r[3], "reply": r[4], "frame_url": r[5],
            "audio_url": r[6],
            "timings": json.loads(r[7] or "{}"),
            "cancelled": bool(r[8]), "ts": r[9],
            "is_pinned": bool(r[10]), "note": r[11] or "",
            "agent_mode": bool(meta.get("agent_mode")),
            "agent_steps": meta.get("agent_steps") or [],
        }


MEMORY = Memory(DB_PATH)


# ----- camera streamer + pHash ------------------------------------------------
def phash_frame(jpg_bytes: bytes) -> int:
    """8x8 perceptual hash → 64-bit int. Tolerates small lighting/exposure
    changes; flips bits hard on real scene change. Decoded IN-PROCESS via PIL:
    on the 8GB box, free RAM routinely sits <100 MB and forking this large
    process for an ffmpeg subprocess stalls for seconds (it was timing out and
    breaking every interactive turn). PIL decode is fork-free and ~1 ms."""
    try:
        im = Image.open(io.BytesIO(jpg_bytes)).convert("L").resize(
            (8, 8), Image.BILINEAR)
    except Exception:  # noqa: BLE001 — corrupt/partial frame → no hash
        return 0
    pix = np.asarray(im, dtype=np.uint8).flatten()
    if pix.size != 64:
        return 0
    mean = pix.mean()
    h = 0
    for b in (pix > mean).astype(np.uint8):
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


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
                print(f"[camera] error: {exc}; retry in 2s", flush=True)
                time.sleep(2)

    # Auto focus/exposure/white-balance + sharpness. The C615 powers up in auto
    # but a prior manual session can leave it locked (focus stuck -> blurry,
    # unreadable text). Re-assert on every capture start so it survives reboots.
    CAM_CTRLS = (
        "focus_automatic_continuous=1",  # autofocus ON (the big fix)
        "auto_exposure=3",               # aperture-priority (auto) exposure
        "white_balance_automatic=1",     # auto white balance (kills blue cast)
        "sharpness=160",
        "backlight_compensation=1",
    )

    def _apply_v4l2_controls(self) -> None:
        for c in self.CAM_CTRLS:
            try:
                subprocess.run(["v4l2-ctl", "-d", CAM_DEVICE, "--set-ctrl", c],
                               capture_output=True, timeout=3)
            except Exception as exc:
                print(f"[camera] v4l2 set {c} failed: {exc}", flush=True)

    def _capture_once(self) -> None:
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "v4l2", "-input_format", "mjpeg",
             "-video_size", f"{CAM_W}x{CAM_H}",
             "-framerate", str(CAM_FPS),
             "-i", CAM_DEVICE,
             "-c", "copy", "-f", "mjpeg", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        # apply camera controls shortly after the device opens (UVC controls are
        # independent of the running stream); off-thread so capture isn't delayed.
        threading.Timer(1.0, self._apply_v4l2_controls).start()
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

# Frame cache: keep last successful 512x384 frame to reuse for repeated
# queries on the same scene.
_FRAME_CACHE: dict = {"phash": None, "path": None, "ts": 0.0}
_FRAME_CACHE_LOCK = threading.Lock()


def capture_frame_for_vlm(out_jpg: Path, crop: tuple | None = None,
                          allow_reuse: bool = False) -> dict:
    """Returns {phash, reused: bool, scene_changed: bool}."""
    raw = CAMERA.get_latest()
    cur_hash = phash_frame(raw)
    with SETTINGS_LOCK:
        cache_enabled = SETTINGS["scene_cache_enabled"]
    reused = False
    scene_changed = True
    with _FRAME_CACHE_LOCK:
        prev_hash = _FRAME_CACHE["phash"]
        prev_path = _FRAME_CACHE["path"]
    if (
        allow_reuse and cache_enabled and prev_hash is not None
        and prev_path is not None and Path(prev_path).exists()
        and crop is None
    ):
        dist = hamming(cur_hash, prev_hash)
        if dist <= PHASH_REUSE_DIST:
            # Reuse the cached frame — link to it. Cheaper than re-encoding.
            out_jpg.write_bytes(Path(prev_path).read_bytes())
            reused = True
            scene_changed = False
    if not reused:
        # In-process crop+scale (PIL, fork-free) — see phash_frame: forking this
        # large process for ffmpeg stalls under the 8GB memory ceiling. Crop is
        # against the real decoded size (more correct than CAM_W/H constants).
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if crop is not None:
            nx, ny, nw, nh = crop
            W, H = im.size
            cx = max(0, int(nx * W))
            cy = max(0, int(ny * H))
            cw = max(64, min(W - cx, int(nw * W)))
            ch = max(64, min(H - cy, int(nh * H)))
            im = im.crop((cx, cy, cx + cw, cy + ch))
        im.resize((VLM_W, VLM_H), Image.LANCZOS).save(
            str(out_jpg), "JPEG", quality=88)
        with _FRAME_CACHE_LOCK:
            if prev_hash is None:
                scene_changed = True
            else:
                scene_changed = hamming(cur_hash, prev_hash) > PHASH_LIVE_GATE
            _FRAME_CACHE["phash"] = cur_hash
            _FRAME_CACHE["path"] = str(out_jpg)
            _FRAME_CACHE["ts"] = time.time()
    return {"phash": cur_hash, "reused": reused, "scene_changed": scene_changed}


# ----- shared audio bus (single ALSA capture, fan-out to subscribers) --------
class AudioBus:
    """Single continuous ALSA capture; raw PCM int16 chunks fanned out.

    All audio consumers (wake-word listener, voice recorder, RMS meter)
    subscribe here. This is the only thing that opens the mic; no more
    ffmpeg-arecord conflicts.
    """
    CHUNK_SAMPLES = SAMPLE_RATE * 80 // 1000  # 80ms @ 16kHz = 1280 samples
    CHUNK_BYTES = CHUNK_SAMPLES * 2           # int16

    def __init__(self) -> None:
        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.running:
            try:
                self._capture_once()
            except Exception as exc:
                print(f"[audio] {exc}; retry in 2s", flush=True)
                time.sleep(2)

    def _capture_once(self) -> None:
        # arecord (not ffmpeg) — ffmpeg buffers stdout in a way that delays
        # chunks indefinitely for small reads; arecord writes raw PCM
        # immediately, period-by-period, which is exactly what we want.
        self.proc = subprocess.Popen(
            ["arecord", "-D", MIC_DEVICE,
             "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1",
             "-t", "raw", "--buffer-size=65536",
             f"--period-size={self.CHUNK_SAMPLES}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
        )
        accum = bytearray()
        while self.running:
            chunk = self.proc.stdout.read(self.CHUNK_BYTES)
            if not chunk:
                self.proc.wait()
                return
            accum.extend(chunk)
            # arecord may emit slightly more or less than CHUNK_BYTES per
            # period — drain all complete chunks from the accumulator.
            while len(accum) >= self.CHUNK_BYTES:
                frame = bytes(accum[:self.CHUNK_BYTES])
                del accum[:self.CHUNK_BYTES]
                samples = np.frombuffer(frame, dtype=np.int16).copy()
                with self.lock:
                    subs = list(self.subscribers)
                for q in subs:
                    with contextlib.suppress(queue.Full):
                        q.put_nowait(samples)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            with contextlib.suppress(ValueError):
                self.subscribers.remove(q)


AUDIO = AudioBus()


# ----- wake-word listener -----------------------------------------------------
class WakeWordListener:
    """openWakeWord on top of AudioBus. Fires callback on detection."""

    def __init__(self) -> None:
        self.enabled = False
        self.model = None
        self.callback = None
        self.thread: threading.Thread | None = None
        self.last_fire_ts = 0.0
        self.score = 0.0       # last predicted score, for visual indicator
        self.active = False    # currently listening (separate from enabled)

    def start(self, callback) -> bool:
        if self.thread is not None:
            return False
        self.callback = callback
        # Lazy import to keep server bootable without openwakeword installed
        from openwakeword.model import Model as OWWModel
        # Use ONNX backend (we have onnxruntime); tflite also works but ONNX
        # is what we already use elsewhere.
        self.model = OWWModel(
            wakeword_models=[WAKE_MODEL_NAME],
            inference_framework="onnx",
        )
        self.enabled = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.enabled = False
        self.active = False

    def _run(self) -> None:
        q = AUDIO.subscribe()
        self.active = True
        try:
            sustained = 0
            while self.enabled:
                try:
                    chunk = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                # openwakeword expects 80ms chunks of int16 mono @ 16kHz
                scores = self.model.predict(chunk)
                max_score = 0.0
                for _kw, score in scores.items():
                    max_score = max(max_score, float(score))
                self.score = max_score
                now = time.monotonic()
                if max_score > WAKE_THRESHOLD:
                    sustained += 1
                else:
                    sustained = 0
                if (sustained >= WAKE_REFRACT_FRAMES
                        and now - self.last_fire_ts > WAKE_COOLDOWN_S):
                    self.last_fire_ts = now
                    sustained = 0
                    try:
                        self.callback(max_score)
                    except Exception as exc:
                        print(f"[wake] callback error: {exc}", flush=True)
        finally:
            AUDIO.unsubscribe(q)
            self.active = False


WAKE = WakeWordListener()


# ----- voice recorder (subscribes to AudioBus, applies VAD) -------------------
def record_voice_via_bus(out_wav: Path, max_seconds: int = 6,
                         silence_dur: float = 1.2) -> None:
    """Record from AudioBus with simple energy-based VAD: stop on silence."""
    q = AUDIO.subscribe()
    recorded: list[np.ndarray] = []
    start = time.monotonic()
    last_voice = start
    SILENCE_THRESH = 600  # RMS threshold (int16)
    MIN_SAMPLES = SAMPLE_RATE  # at least 1s before silence-cut allowed
    try:
        total_samples = 0
        while time.monotonic() - start < max_seconds:
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                continue
            recorded.append(chunk)
            total_samples += len(chunk)
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            if rms > SILENCE_THRESH:
                last_voice = time.monotonic()
            if (total_samples >= MIN_SAMPLES
                    and time.monotonic() - last_voice > silence_dur):
                break
        if not recorded:
            audio = np.zeros(SAMPLE_RATE, dtype=np.int16)
        else:
            audio = np.concatenate(recorded)
    finally:
        AUDIO.unsubscribe(q)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())


# ----- audio level monitor (orb waveform) -------------------------------------
class AudioMonitor:
    """Continuously emits RMS level samples to subscribers (browser orb)."""

    def __init__(self) -> None:
        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        q = AUDIO.subscribe()
        try:
            while self.running:
                try:
                    chunk = q.get(timeout=1)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                level = min(1.0, rms / 8000.0)
                with self.lock:
                    subs = list(self.subscribers)
                for sq in subs:
                    with contextlib.suppress(queue.Full):
                        sq.put_nowait(level)
        finally:
            AUDIO.unsubscribe(q)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=40)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            with contextlib.suppress(ValueError):
                self.subscribers.remove(q)


AUDIO_MONITOR = AudioMonitor()


# ----- STT / TTS helpers ------------------------------------------------------
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


# ----- sentence-streaming TTS -------------------------------------------------
SENTENCE_RE = re.compile(r"([.!?](?:\s|$)|\n\n+|\n(?=[A-Z*\-•\d]))")


class StreamingTTS:
    """Splits a streaming VLM reply into sentences, synth each via Piper in
    a single worker thread (preserving order), and reports segment URLs to
    a callback. Client plays them gaplessly via Web Audio API."""

    def __init__(self, turn_id: str, on_segment) -> None:
        self.turn_id = turn_id
        self.on_segment = on_segment
        self.work = SESSION_DIR / turn_id
        self.work.mkdir(parents=True, exist_ok=True)
        self.buffer = ""
        self.next_idx = 0
        self.q: queue.Queue = queue.Queue()
        self.done_evt = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.segments: list[dict] = []
        self.lock = threading.Lock()

    def add_delta(self, delta: str) -> None:
        with self.lock:
            self.buffer += delta
            self._flush(force=False)

    def finish(self) -> None:
        with self.lock:
            self._flush(force=True)
        self.q.put(None)

    def _flush(self, force: bool) -> None:
        while True:
            m = SENTENCE_RE.search(self.buffer)
            if m:
                end = m.end()
                sentence = self.buffer[:end].strip()
                self.buffer = self.buffer[end:]
            elif force and self.buffer.strip():
                sentence = self.buffer.strip()
                self.buffer = ""
            else:
                return
            if not sentence:
                continue
            self.q.put((self.next_idx, sentence))
            self.next_idx += 1
            if not force:
                continue
            if force and not self.buffer.strip():
                return

    def _worker(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                self.done_evt.set()
                return
            idx, text = item
            wav = self.work / f"seg_{idx:03d}.wav"
            try:
                synthesize(text, wav)
            except Exception as exc:
                print(f"[tts seg {idx}] error: {exc}", flush=True)
                continue
            seg = {
                "index": idx,
                "text": text,
                "url": f"/audio_seg/{self.turn_id}/{idx:03d}",
                "duration_s": _wav_duration_s(wav),
            }
            with self.lock:
                self.segments.append(seg)
            try:
                self.on_segment(seg)
            except Exception as exc:
                print(f"[tts cb] error: {exc}", flush=True)


def _wav_duration_s(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


# ----- conversation memory (in-process, last 3 pairs for VLM context) ---------
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


# ----- streaming VLM (httpx) --------------------------------------------------
# Rolling snapshot of the last interactive VLM inference (tok/s, TTFT, prefill).
# Surfaced in /metrics + /nano so the HUD proves we're driving the GPU hard.
_VLM_PERF: dict = {}


def stream_vlm(question: str, frame_jpg: Path,
               cancel: threading.Event, on_token,
               include_history: bool = True,
               connect_timeout_s: float = 5.0,
               read_timeout_s: float = 20.0,
               priority: bool = True) -> tuple[str, dict]:
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
        # KV-prefix reuse: cache the shared system+history prefix across turns so
        # llama.cpp only prefills the new tokens (frontier on-device latency win).
        "cache_prompt": True,
        "timings_per_token": True,
        "stream_options": {"include_usage": True},
    }
    full: list[str] = []
    usage: dict = {}
    stalled = False
    req_start = None
    first_tok_at = None
    timeout = httpx.Timeout(
        connect=connect_timeout_s, read=read_timeout_s,
        write=connect_timeout_s, pool=connect_timeout_s,
    )
    deadline_at = time.monotonic() + max(read_timeout_s * 4, 45.0)
    # serialize GPU inference; background callers skip when busy (priority=False)
    if not jarvis_tools.VLM_BUSY.acquire(blocking=priority):
        return "", {"skipped": True}
    req_start = time.monotonic()
    try:
        with httpx.stream(
            "POST", VLM_URL, json=body, timeout=timeout,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if cancel.is_set():
                    break
                if time.monotonic() > deadline_at:
                    stalled = True
                    break
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith(":"):
                    continue
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
                if obj.get("timings"):
                    usage["timings"] = obj["timings"]
                # NB: with stream_options.include_usage, llama.cpp sends a final
                # chunk with "choices": [] — `.get(...,[{}])` doesn't catch an
                # empty list, so guard with `or [{}]` to avoid an IndexError that
                # would crash the whole stream (broke the captioner + perf).
                delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta is None:
                    continue
                if first_tok_at is None:
                    first_tok_at = time.monotonic()
                full.append(delta)
                try:
                    on_token(delta)
                except Exception:  # noqa: BLE001
                    pass  # a TTS/consumer failure must never abort the VLM stream
    except (httpx.TimeoutException, httpx.RemoteProtocolError,
            httpx.NetworkError, httpx.HTTPStatusError):
        stalled = True
    finally:
        jarvis_tools.VLM_BUSY.release()
    if stalled:
        usage["stalled"] = True
    # record interactive inference perf for the diagnostics HUD (frontier proof)
    if priority and not usage.get("skipped"):
        tm = usage.get("timings") or {}
        perf: dict = {}
        # tokens generated: prefer server count, else the number of streamed deltas
        gen_n = (tm.get("predicted_n") or usage.get("completion_tokens")
                 or len(full))
        if tm.get("predicted_per_second"):
            perf["tok_s"] = round(tm["predicted_per_second"], 1)
        elif gen_n and first_tok_at:                   # robust wall-clock fallback
            dt = time.monotonic() - first_tok_at
            if dt > 0:
                perf["tok_s"] = round(gen_n / dt, 1)
        if first_tok_at and req_start:
            perf["ttft_ms"] = round((first_tok_at - req_start) * 1000)
        if tm.get("prompt_ms"):
            perf["prefill_ms"] = round(tm["prompt_ms"])
        if tm.get("prompt_n") is not None:
            perf["prompt_n"] = tm["prompt_n"]          # tokens actually prefilled
        if tm.get("cache_n") is not None:
            perf["cache_n"] = tm["cache_n"]            # KV-cache prefix reused
        perf["prompt_tokens"] = usage.get("prompt_tokens")
        perf["completion_tokens"] = gen_n
        perf["ts"] = time.time()
        if perf.get("tok_s"):
            _VLM_PERF.update(perf)
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


# ----- turn worker ------------------------------------------------------------
def run_turn(ctx: TurnCtx, payload: dict) -> None:
    tid = ctx.tid
    LAST_USER_ACTIVITY["ts"] = time.time()
    work = SESSION_DIR / tid
    work.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    transcription = ""
    kind = ctx.kind
    try:
        with SETTINGS_LOCK:
            rec_default = SETTINGS["record_seconds"]
        seconds = max(2, min(15, int(payload.get("seconds", rec_default))))
        crop = payload.get("crop")
        if crop and len(crop) == 4:
            crop = tuple(float(c) for c in crop)
        else:
            crop = None

        t = time.monotonic()
        if kind == "talk":
            ctx.emit({"phase": "recording", "seconds": seconds})
            record_voice_via_bus(work / "user.wav", max_seconds=seconds)
            timings["record_s"] = round(time.monotonic() - t, 2); t = time.monotonic()
            ctx.emit({"phase": "transcribing"})
            transcription = transcribe(work / "user.wav")
            timings["transcribe_s"] = round(time.monotonic() - t, 2); t = time.monotonic()
            question = transcription or "Describe what you see briefly."
            allow_reuse = False
        elif kind == "text":
            question = (payload.get("text") or "").strip()
            if not question:
                raise ValueError("text mode requires text")
            allow_reuse = bool(payload.get("reuse_frame", True))
        elif kind == "snap":
            question = (
                "Describe what you actually see in one or two sentences. "
                "If the scene is genuinely empty or fully black, say so."
            )
            allow_reuse = False
        elif kind == "point":
            point = payload.get("point")
            if not point or len(point) != 2:
                raise ValueError("point mode requires point [x, y] 0..1")
            px, py = float(point[0]), float(point[1])
            crop = (max(0.0, px - 0.15), max(0.0, py - 0.15),
                    min(0.30, 1.0 - max(0.0, px - 0.15)),
                    min(0.30, 1.0 - max(0.0, py - 0.15)))
            question = ("The user tapped a specific point in the image. "
                        "Describe what is at this point or in this region. "
                        "Be specific and brief. If unclear, say so.")
            allow_reuse = False
        elif kind == "regenerate":
            with HISTORY_LOCK:
                if len(HISTORY) >= 2:
                    HISTORY.pop()
                    last_user = HISTORY.pop()
                    question = last_user["content"] if isinstance(last_user["content"], str) else "Try again."
                else:
                    raise ValueError("nothing to regenerate")
            allow_reuse = False
        else:
            raise ValueError(f"unknown kind: {kind}")

        # --- power / wake voice commands: text-matched on the transcript so they
        #     work even in ECO (VLM stopped). Skip the camera + VLM entirely. ---
        if kind in ("talk", "text"):
            pa = _route_power_command(question)
            if pa:
                reply = _power_reply(pa)
                ctx.emit({"phase": "thinking"})
                ctx.emit({"phase": "token", "delta": reply})
                ptts = StreamingTTS(tid, lambda seg: ctx.emit({"phase": "audio_segment", **seg}))
                ptts.add_delta(reply)
                ctx.emit({"phase": "speaking"})
                ptts.finish()
                ptts.done_evt.wait(timeout=20)
                try:
                    synthesize(reply, work / "reply.wav")
                except Exception:  # noqa: BLE001
                    pass
                set_power(pa)   # eco/wake run async; shutdown/reboot fire after a grace
                history_append("user", question)
                history_append("assistant", reply)
                ctx.result = {"turn_id": tid, "kind": kind, "question": question,
                              "transcription": transcription, "reply": reply,
                              "cancelled": False, "timings": timings,
                              "usage": {"power_action": pa},
                              "audio_url": f"/audio/{tid}.wav", "ts": time.time()}
                try:
                    MEMORY.record(ctx.result)
                except Exception:  # noqa: BLE001
                    pass
                ctx.emit({"phase": "done", "result": ctx.result})
                return   # finally: sets ctx.done
        # --- in ECO, any other request auto-wakes first (VLM reloads ~20s) ---
        if POWER["state"] == "eco":
            wake_if_eco(ctx.emit)

        ctx.emit({"phase": "capturing", "transcription": transcription,
                  "question": question, "crop": crop})
        cap = capture_frame_for_vlm(work / "frame.jpg", crop=crop,
                                    allow_reuse=allow_reuse)
        timings["capture_s"] = round(time.monotonic() - t, 2); t = time.monotonic()
        if cap.get("reused"):
            ctx.emit({"phase": "frame_reused"})

        ctx.emit({"phase": "thinking"})
        reply_buf: list = []
        tts = StreamingTTS(tid, lambda seg: ctx.emit({
            "phase": "audio_segment", **seg,
        }))

        with SETTINGS_LOCK:
            agent_on = bool(SETTINGS.get("agent_mode_enabled"))
            agent_max_steps = int(SETTINGS.get("agent_max_steps", 3))

        agent_steps: list = []
        if agent_on:
            # Agent path: plan -> act -> observe -> re-plan, then TTS the
            # final answer by sentence chunks (re-using the streaming TTS).
            def _on_step(step: dict) -> None:
                agent_steps.append(step)
                ctx.emit({
                    "phase": "agent_step",
                    "step": step.get("step"),
                    "tool": step.get("tool"),
                    "args": step.get("args"),
                    "ok": (step.get("result") or {}).get("ok", False),
                })

            ctx.emit({"phase": "agent_start", "max_steps": agent_max_steps})
            try:
                ar = jarvis_tools.agentic_loop(
                    jarvis_tools.TOOLS._ctx,
                    question,
                    frame_path=work / "frame.jpg",
                    max_steps=agent_max_steps,
                    use_frame=True,
                    on_step=_on_step,
                )
            except Exception as ae:
                ar = {"final": f"(agent error: {ae})", "steps": [],
                      "stopped": f"error:{type(ae).__name__}",
                      "usage_total": {}}
            reply = (ar.get("final") or "").strip()
            usage = ar.get("usage_total") or {}
            usage["agent_stopped"] = ar.get("stopped")
            usage["agent_steps"] = len(ar.get("steps") or [])

            # Chunk the final answer by sentence-ish boundaries so the
            # existing streaming TTS produces gapless audio.
            for chunk in re.findall(r"[^.!?\n]+[.!?\n]?", reply):
                chunk = chunk.strip()
                if not chunk:
                    continue
                reply_buf.append(chunk + " ")
                ctx.emit({"phase": "token", "delta": chunk + " "})
                tts.add_delta(chunk + " ")
                if ctx.cancel.is_set():
                    break
        else:
            def on_tok(delta: str) -> None:
                reply_buf.append(delta)
                ctx.emit({"phase": "token", "delta": delta})
                tts.add_delta(delta)

            reply, usage = stream_vlm(question, work / "frame.jpg",
                                      ctx.cancel, on_tok)
        timings["vlm_s"] = round(time.monotonic() - t, 2); t = time.monotonic()

        if ctx.cancel.is_set():
            ctx.emit({"phase": "cancelled"})
            tts.finish()
        else:
            history_append("user", question)
            history_append("assistant", reply)
            ctx.emit({"phase": "speaking"})
            tts.finish()
            # Wait briefly for TTS worker to finish so we can also save a
            # consolidated reply.wav for compat with the legacy /audio/
            # endpoint and the timings breakdown.
            tts.done_evt.wait(timeout=30)
            timings["tts_s"] = round(time.monotonic() - t, 2)
            try:
                synthesize(reply, work / "reply.wav")
            except Exception:
                pass

        ctx.result = {
            "turn_id": tid, "kind": kind, "question": question,
            "transcription": transcription, "reply": reply,
            "cancelled": ctx.cancel.is_set(), "timings": timings,
            "usage": usage,
            "audio_url": f"/audio/{tid}.wav",
            "frame_url": f"/frame/{tid}.jpg",
            "crop": list(crop) if crop else None,
            "phash": cap.get("phash"),
            "frame_reused": cap.get("reused", False),
            "scene_changed": cap.get("scene_changed", True),
            "segments": list(tts.segments),
            "agent_mode": agent_on,
            "agent_steps": agent_steps,
            "ts": time.time(),
        }
        # Persist
        try:
            MEMORY.record(ctx.result)
        except Exception as exc:
            print(f"[memory] record error: {exc}", flush=True)
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


# ----- investigate (Iron-Man vision: locate -> zoom -> identify -> web) -------
def run_investigate(ctx: TurnCtx, payload: dict) -> None:
    """Drive the jarvis_tools.run_investigate pipeline, streaming each phase as
    an SSE event over the existing turn machinery (/events/<turn_id>)."""
    LAST_USER_ACTIVITY["ts"] = time.time()
    try:
        tctx = jarvis_tools.TOOLS._ctx
        if tctx is None:
            ctx.emit({"phase": "error", "error": "tool context not ready"})
            ctx.result = {"error": "tool context not ready"}
            return
        subject = (payload.get("subject") or "").strip()
        point = payload.get("point")
        region = payload.get("region")
        web = bool(payload.get("web", True))
        result = jarvis_tools.run_investigate(
            tctx, subject=subject, point=point, region=region, web=web,
            on_event=lambda ev: ctx.emit(ev),
        )
        ctx.result = result
        # persist a lightweight turn record for history/memory
        try:
            ident = result.get("identification", {})
            MEMORY.record({
                "turn_id": ctx.tid, "kind": "investigate",
                "question": f"investigate: {subject}" if subject else "investigate",
                "transcription": "",
                "reply": (f"{ident.get('name','')} "
                          f"({ident.get('confidence','')}) — "
                          f"{ident.get('details','')}").strip(),
                "frame_url": result.get("zoom_url"),
                "audio_url": "", "timings": {}, "ts": time.time(),
            })
        except Exception as exc:
            print(f"[investigate] record error: {exc}", flush=True)
    except Exception as e:  # noqa: BLE001
        ctx.emit({"phase": "error", "error": str(e)})
        ctx.result = {"error": str(e)}
    finally:
        ctx.done.set()


# ----- live mode --------------------------------------------------------------
class LiveMode:
    def __init__(self) -> None:
        self.lock = threading.Lock()    # subscribers list only
        self.running = False
        self.subscribers: list[queue.Queue] = []
        self.observations: collections.deque = collections.deque(maxlen=50)
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self.last_phash: int | None = None

    def start(self) -> bool:
        if self.running:
            return False
        self.running = True
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._broadcast({"event": "start", "ts": time.time()})
        return True

    def stop(self) -> bool:
        if not self.running:
            return False
        self.running = False
        self._stop_evt.set()
        self._broadcast({"event": "stop", "ts": time.time()})
        return True

    def is_running(self) -> bool:
        return self.running

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        recent = list(self.observations)[-10:]
        running_now = self.running
        with self.lock:
            self.subscribers.append(q)
        for obs in recent:
            with contextlib.suppress(queue.Full):
                q.put_nowait({"event": "observation", **obs})
        with contextlib.suppress(queue.Full):
            q.put_nowait({"event": "state", "running": running_now})
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            with contextlib.suppress(ValueError):
                self.subscribers.remove(q)

    def _broadcast(self, ev: dict) -> None:
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            with contextlib.suppress(queue.Full):
                q.put_nowait(ev)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                tid = ("live-" + time.strftime("%Y%m%d-%H%M%S")
                       + "-" + uuid.uuid4().hex[:4])
                work = SESSION_DIR / tid
                work.mkdir(parents=True, exist_ok=True)
                cap = capture_frame_for_vlm(work / "frame.jpg",
                                            allow_reuse=False)
                # Scene-change gate: if the new frame is nearly identical to
                # the previous live frame, skip the VLM call entirely.
                if (self.last_phash is not None
                    and hamming(cap["phash"], self.last_phash)
                        <= PHASH_LIVE_GATE):
                    self._broadcast({"event": "skip", "reason": "scene_unchanged",
                                     "ts": time.time()})
                else:
                    self.last_phash = cap["phash"]
                    cancel = threading.Event()
                    reply, meta = stream_vlm(
                        "In one short sentence, describe what is happening or "
                        "visible in the scene RIGHT NOW. Be concrete. If "
                        "nothing has notably changed, say what is most salient.",
                        work / "frame.jpg", cancel, lambda d: None,
                        include_history=False, read_timeout_s=15.0,
                        priority=False,
                    )
                    if meta.get("skipped"):
                        pass  # VLM busy with interactive work; skip this tick
                    elif meta.get("stalled") and not reply:
                        self._broadcast({"event": "error",
                                         "error": "VLM stalled, skipping",
                                         "ts": time.time()})
                    else:
                        obs = {"ts": time.time(), "turn_id": tid,
                               "text": reply,
                               "frame_url": f"/frame/{tid}.jpg",
                               "scene_changed": True}
                        self.observations.append(obs)
                        self._broadcast({"event": "observation", **obs})
            except Exception as exc:
                self._broadcast({"event": "error", "error": str(exc),
                                 "ts": time.time()})
            with SETTINGS_LOCK:
                interval = SETTINGS["live_interval_s"]
            if self._stop_evt.wait(interval):
                return


LIVE = LiveMode()


# ----- visual memory: ambient keyframe captioner (the "world model") ----------
_VMEM_CAPTION_PROMPT = (
    "You are building a searchable memory of what a camera sees over time. "
    "Describe this scene factually in ONE sentence, then list the distinct "
    "notable objects/people visible. Only mention what is actually visible; "
    "do not invent. Reply in EXACTLY this format:\n"
    "SCENE: <one factual sentence>\n"
    "OBJECTS: <comma-separated nouns, or 'none'>"
)


def _parse_caption(text: str) -> tuple[str, str]:
    scene, objects = "", ""
    for line in (text or "").splitlines():
        m = re.match(r"\s*SCENE\s*[:\-]\s*(.+)", line, re.I)
        if m:
            scene = m.group(1).strip()
        m = re.match(r"\s*OBJECTS?\s*[:\-]\s*(.+)", line, re.I)
        if m:
            objects = m.group(1).strip()
    if not scene:
        scene = re.sub(r"\s+", " ", (text or "")).strip()[:200]
    if objects.lower() in ("none", "n/a", ""):
        objects = ""
    return scene, objects


class VisualMemory:
    """Ambient, scene-gated keyframe captioner that persists a searchable log of
    what the camera has seen. The slow side of the dual-loop: it yields the VLM
    to interactive turns and only fires on real scene change."""

    def __init__(self) -> None:
        self.enabled = True
        self.interval_s = 30.0
        self.min_user_idle_s = 12.0
        self.last_phash: int | None = None
        self.last_cap_ts = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="vmem",
                                        daemon=True)
        self._thread.start()

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)

    def capture_now(self, source: str = "manual", priority: bool = True) -> dict | None:
        """Force-capture a keyframe into visual memory. Returns the record, or
        None if the VLM was busy (background, priority=False) and skipped."""
        raw = CAMERA.get_latest()
        ph = phash_frame(raw)
        fname = uuid.uuid4().hex + ".jpg"
        fpath = VMEM_DIR / fname
        # 512x384 capture for legible captions — in-process (PIL), fork-free
        # (forking under the 8GB ceiling stalls; see phash_frame).
        Image.open(io.BytesIO(raw)).convert("RGB").resize(
            (VLM_W, VLM_H), Image.LANCZOS).save(str(fpath), "JPEG", quality=88)
        cancel = threading.Event()
        text, _u = stream_vlm(_VMEM_CAPTION_PROMPT, fpath, cancel,
                              lambda d: None, include_history=False,
                              read_timeout_s=20.0, priority=priority)
        if _u.get("skipped"):
            return None  # VLM busy with interactive work; skip this cycle
        scene, objects = _parse_caption(text)
        rid = MEMORY.vmem_record(time.time(), scene, objects, fname, ph, source)
        self.last_phash = ph
        self.last_cap_ts = time.time()
        return {"id": rid, "caption": scene, "objects": objects,
                "frame_url": f"/vmem/{fname}", "frame_path": str(fpath),
                "source": source}

    def _run(self) -> None:
        # small initial delay so the camera + VLM are up
        if self._stop.wait(15.0):
            return
        while not self._stop.is_set():
            try:
                if (self.enabled
                        and time.time() - LAST_USER_ACTIVITY["ts"]
                            > self.min_user_idle_s
                        and _VLM_UP["flag"]):
                    raw = CAMERA.get_latest(timeout=2.0)
                    ph = phash_frame(raw)
                    changed = (self.last_phash is None
                               or hamming(ph, self.last_phash) > PHASH_LIVE_GATE)
                    if changed:
                        rec = self.capture_now(source="ambient", priority=False)
                        if rec:  # None when the VLM was busy and we skipped
                            try:
                                WATCHER.evaluate(Path(rec["frame_path"]))
                            except Exception as wexc:
                                print(f"[watch] {wexc}", flush=True)
            except Exception as exc:
                print(f"[vmem] {exc}", flush=True)
            if self._stop.wait(self.interval_s):
                return


VMEM = VisualMemory()


# ----- proactive watcher: evaluate natural-language alert rules on keyframes ---
class Watcher:
    """Ambient monitoring: the VLM checks active natural-language conditions
    against scene-change keyframes and fires notifications. One batched VLM
    call covers all active rules; each rule is debounced by a cooldown."""

    COOLDOWN_S = 120.0

    def evaluate(self, frame_path: Path) -> list[dict]:
        rules = MEMORY.watch_list(only_active=True)
        now = time.time()
        due = [r for r in rules if now - (r["last_fired"] or 0) > self.COOLDOWN_S]
        if not due:
            return []
        numbered = "\n".join(f"{i+1}) {r['text']}" for i, r in enumerate(due))
        prompt = (
            "You are a vigilant monitor watching a live camera. For EACH numbered "
            "condition below, answer strictly YES or NO based ONLY on what is "
            "actually visible in this image right now. Do not guess.\n\n"
            f"{numbered}\n\n"
            "Reply one line per condition, exactly like '1: YES' or '2: NO'. "
            "Nothing else."
        )
        cancel = threading.Event()
        text, _u = stream_vlm(prompt, frame_path, cancel, lambda d: None,
                              include_history=False, read_timeout_s=20.0,
                              priority=False)
        if _u.get("skipped"):
            return []  # VLM busy; re-evaluate next cycle
        fired = []
        for i, r in enumerate(due):
            m = re.search(rf"{i+1}\s*[:\-]\s*(yes|no)", text or "", re.I)
            if m and m.group(1).lower() == "yes":
                self._fire(r)
                fired.append(r)
        return fired

    def _fire(self, rule: dict) -> None:
        note = {"kind": "alert", "text": rule["text"], "rule_id": rule["id"]}
        try:
            ndir = SESSION_DIR / "notifications"
            ndir.mkdir(parents=True, exist_ok=True)
            wav = ndir / f"watch-{rule['id']}-{int(time.time())}.wav"
            synthesize(f"Heads up. {rule['text']}", wav)
            if wav.exists():
                note["audio_url"] = f"/audio_note/{wav.name}"
        except Exception as exc:  # noqa: BLE001
            note["tts_error"] = str(exc)
        try:
            jarvis_tools.TOOLS.emit_notification(note)
        except Exception:
            pass
        MEMORY.watch_mark_fired(rule["id"])
        print(f"[watch] FIRED rule #{rule['id']}: {rule['text']!r}", flush=True)


WATCHER = Watcher()


# ----- wake-word integration (triggers a voice turn when "Hey Jarvis") --------
def _wake_callback(score: float) -> None:
    """When wake word detected, fire a talk-mode turn."""
    tid = ("wake-" + time.strftime("%Y%m%d-%H%M%S")
           + "-" + uuid.uuid4().hex[:4])
    ctx = register_turn(tid, "talk")
    threading.Thread(
        target=run_turn,
        args=(ctx, {"kind": "talk", "seconds": 6, "wake_score": score}),
        daemon=True,
    ).start()
    # Broadcast wake event to all live subscribers so the UI lights up
    LIVE._broadcast({"event": "wake", "turn_id": tid, "score": score,
                     "ts": time.time()})


def set_wake_enabled(enabled: bool) -> bool:
    if enabled and not WAKE.enabled:
        ok = WAKE.start(_wake_callback)
        with SETTINGS_LOCK:
            SETTINGS["wake_word_enabled"] = bool(ok)
        return ok
    if not enabled and WAKE.enabled:
        WAKE.stop()
        with SETTINGS_LOCK:
            SETTINGS["wake_word_enabled"] = False
        return True
    return True


# ----- metrics ----------------------------------------------------------------
_VLM_UP = {"flag": False}


def _vlm_health_poller() -> None:
    while True:
        try:
            r = httpx.get(VLM_HEALTH, timeout=httpx.Timeout(
                connect=2, read=2, write=2, pool=2,
            ))
            _VLM_UP["flag"] = (r.status_code == 200)
        except Exception:
            _VLM_UP["flag"] = False
        time.sleep(3)


def read_meminfo() -> dict:
    m: dict = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            m[k.strip()] = v.strip()
    return m


# ---- Self-healing VLM memory watchdog ---------------------------------------
# The 8GB ceiling is the real limiter: when MemAvailable collapses, llama.cpp
# stalls on the mmproj vision prefill (needs ~800MB CONTIGUOUS; CMA fragments)
# and returns 0 tokens. Auto-refresh jarvis-vlm to reclaim memory — but ONLY
# when the box is idle, never mid-inference, and rate-limited, so it can't
# interrupt the user. This is what keeps the edge box self-sustaining.
VLM_AUTOREFRESH = {
    "enabled": True,
    "min_avail_mb": 160,    # refresh when MemAvailable stays below this
    "consecutive": 3,       # require N consecutive low readings (~60s)
    "idle_s": 30,           # only when no user activity for this long
    "cooldown_s": 1800,     # at most once / 30 min
    "last_refresh_ts": 0.0,
    "refreshes": 0,
    "low_streak": 0,
}


def _mem_avail_mb() -> int:
    try:
        return int(re.findall(r"\d+", read_meminfo().get("MemAvailable", "0"))[0]) // 1024
    except Exception:  # noqa: BLE001
        return 9999


def _vlm_memory_watchdog() -> None:
    cfg = VLM_AUTOREFRESH
    while True:
        time.sleep(20)
        try:
            # auto-eco: drop to cool/low-power after a long idle (any request wakes it)
            if (POWER["state"] == "full" and not POWER["transition"]
                    and not PERCEPTION["on"]
                    and POWER["auto_eco_idle_s"]
                    and time.time() - LAST_USER_ACTIVITY["ts"] > POWER["auto_eco_idle_s"]):
                print("[power] idle — auto-entering eco", flush=True)
                _power_transition("eco")
                continue
            # self-heal is for FULL only; eco/perception intentionally stop the VLM
            if (not cfg["enabled"] or PERCEPTION["on"]
                    or POWER["state"] != "full" or POWER["transition"]):
                cfg["low_streak"] = 0
                continue
            avail = _mem_avail_mb()
            if avail >= cfg["min_avail_mb"]:
                cfg["low_streak"] = 0
                continue
            cfg["low_streak"] += 1
            now = time.time()
            if (cfg["low_streak"] < cfg["consecutive"]
                    or now - LAST_USER_ACTIVITY["ts"] < cfg["idle_s"]
                    or now - cfg["last_refresh_ts"] < cfg["cooldown_s"]):
                continue
            # Hold the inference lock so no turn can hit the VLM mid-restart.
            if not jarvis_tools.VLM_BUSY.acquire(blocking=False):
                continue
            try:
                print(f"[watchdog] MemAvailable={avail}MB low for "
                      f"{cfg['low_streak']} checks — refreshing jarvis-vlm",
                      flush=True)
                subprocess.run(
                    ["sudo", "-n", "systemctl", "restart", "jarvis-vlm"],
                    check=False, capture_output=True, timeout=30)
                cfg["last_refresh_ts"] = time.time()
                cfg["refreshes"] += 1
                cfg["low_streak"] = 0
                deadline = time.time() + 60   # wait for llama-server to come back
                while time.time() < deadline:
                    time.sleep(3)
                    try:
                        if httpx.get(VLM_HEALTH, timeout=httpx.Timeout(
                                connect=2, read=2, write=2, pool=2)).status_code == 200:
                            break
                    except Exception:  # noqa: BLE001
                        pass
                print(f"[watchdog] jarvis-vlm refreshed; MemAvailable now "
                      f"{_mem_avail_mb()}MB", flush=True)
            finally:
                jarvis_tools.VLM_BUSY.release()
        except Exception:  # noqa: BLE001
            pass


def read_soc_temp_c() -> float:
    temps = []
    for f in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(f.read_text().strip()) / 1000.0)
        except Exception:
            pass
    return round(max(temps), 1) if temps else 0.0


# ----- Full Jetson telemetry (jtop-grade, read-only, zero GPU cost) -----------
# Parses a persistent `tegrastats` stream into a rich snapshot: per-core CPU
# load+freq, GPU (GR3D) util, EMC util, every thermal zone, and the INA3221
# power rails (instantaneous + running average). Plus nvpmodel power mode and
# CPU governor. This is how we make ALL Nano diagnostics visible and prove we
# are driving the hardware at its limits. Degrades gracefully if unavailable.
class TegraStats:
    _CPU_RE = re.compile(r"(\d+|off)%?@?(\d+)?")
    _TEMP_RE = re.compile(r"(\w+?)@([\d.]+)C")
    _PWR_RE = re.compile(r"(VDD_\w+|VIN_\w+|POM_\w+|NC\w*)\s+(\d+)mW/(\d+)mW")
    _THROTTLE_C = 85.0  # Orin Nano SoC throttle threshold

    def __init__(self):
        self.lock = threading.Lock()
        self.data: dict = {}
        self.meta: dict = {}
        self.ok = False
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._last_net = None  # (t, rx_bytes, tx_bytes) for throughput delta

    @staticmethod
    def _net_bytes():
        rx = tx = 0
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, _, rest = line.partition(":")
            if name.strip() == "lo" or not rest.strip():
                continue
            f = rest.split()
            rx += int(f[0]); tx += int(f[8])
        return rx, tx

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="tegrastats",
                                        daemon=True)
        self._thread.start()

    def _run(self):
        self._refresh_meta()
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except (FileNotFoundError, OSError):
            self.ok = False
            return
        n = 0
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            try:
                parsed = self._parse(line)
            except Exception:  # noqa: BLE001 — never let a bad line kill telemetry
                continue
            if parsed:
                try:  # network throughput (delta over the ~1s tegrastats tick)
                    rx, tx = self._net_bytes()
                    tnow = time.time()
                    if self._last_net:
                        dt = tnow - self._last_net[0]
                        if dt > 0:
                            parsed["net_rx_kbs"] = round((rx - self._last_net[1]) / dt / 1024, 1)
                            parsed["net_tx_kbs"] = round((tx - self._last_net[2]) / dt / 1024, 1)
                    self._last_net = (tnow, rx, tx)
                except Exception:  # noqa: BLE001
                    pass
                with self.lock:
                    self.data = parsed
                    self.ok = True
            n += 1
            if n % 20 == 0:   # power mode/governor rarely change — sample slowly
                self._refresh_meta()

    def _parse(self, line: str) -> dict:
        d: dict = {}
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            d["ram_used_mb"], d["ram_total_mb"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"SWAP (\d+)/(\d+)MB", line)
        if m:
            d["swap_used_mb"], d["swap_total_mb"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"CPU \[([^\]]+)\]", line)
        if m:
            cores = []
            for c in m.group(1).split(","):
                mc = re.match(r"(\d+)%@(\d+)", c.strip())
                if mc:
                    cores.append({"load": int(mc.group(1)), "freq": int(mc.group(2))})
                else:
                    cores.append({"load": 0, "freq": 0, "off": True})
            d["cpu"] = cores
        m = re.search(r"GR3D_FREQ (\d+)%", line)
        if m:
            d["gpu_util"] = int(m.group(1))
        m = re.search(r"EMC_FREQ (\d+)%", line)
        if m:
            d["emc_util"] = int(m.group(1))
        temps = {n: float(v) for n, v in self._TEMP_RE.findall(line)}
        if temps:
            d["temps"] = temps
            d["temp_max_c"] = round(max(temps.values()), 1)
            d["throttle_headroom_c"] = round(self._THROTTLE_C - max(temps.values()), 1)
        rails = {}
        for name, now, avg in self._PWR_RE.findall(line):
            rails[name] = {"now": int(now), "avg": int(avg)}
        if rails:
            d["power"] = rails
            vin = rails.get("VDD_IN") or rails.get("VIN_SYS_5V0") or rails.get("POM_5V_IN")
            if vin:
                d["power_total_mw"], d["power_avg_mw"] = vin["now"], vin["avg"]
            else:
                d["power_total_mw"] = sum(r["now"] for r in rails.values())
                d["power_avg_mw"] = sum(r["avg"] for r in rails.values())
        return d

    def _refresh_meta(self):
        meta = {}
        try:
            out = subprocess.run(["sudo", "-n", "nvpmodel", "-q"],
                                 capture_output=True, text=True, timeout=5).stdout
            mm = re.search(r"NV Power Mode:\s*(\S+)", out)
            if mm:
                meta["power_mode"] = mm.group(1)
            mn = re.search(r"^\s*(\d+)\s*$", out, re.M)
            if mn:
                meta["power_mode_id"] = int(mn.group(1))
        except Exception:  # noqa: BLE001
            pass
        try:
            gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            if gov.exists():
                meta["governor"] = gov.read_text().strip()
        except Exception:  # noqa: BLE001
            pass
        try:
            du = shutil.disk_usage("/")
            meta["disk_total_gb"] = round(du.total / 1e9, 1)
            meta["disk_used_gb"] = round(du.used / 1e9, 1)
            meta["disk_pct"] = round(du.used / du.total * 100)
        except Exception:  # noqa: BLE001
            pass
        try:  # jetson_clocks pins min freq up to max (it doesn't rename the governor)
            base = Path("/sys/devices/system/cpu/cpu0/cpufreq")
            meta["jetson_clocks"] = (
                (base / "scaling_min_freq").read_text().strip()
                == (base / "scaling_max_freq").read_text().strip())
        except Exception:  # noqa: BLE001
            meta["jetson_clocks"] = False
        if meta:
            with self.lock:
                self.meta.update(meta)

    def snapshot(self) -> dict:
        with self.lock:
            d = dict(self.data)
            d["meta"] = dict(self.meta)
            d["ok"] = self.ok
        return d


TEGRA = TegraStats()


_JCLK_STORE = "/tmp/jarvis_jetson_clocks_before.txt"


def set_jetson_clocks(on: bool) -> dict:
    """Lock all clocks to max (jetson_clocks) for peak throughput, or restore
    DVFS. Safe on this thermally-headroomed box (MAXN_SUPER, ~28C margin); it
    trades a little idle power for zero clock-ramp latency. We snapshot the
    pre-boost state so --restore is clean; if that snapshot is gone, fall back
    to resetting each core's scaling_min_freq to the hardware minimum."""
    try:
        if on:
            subprocess.run(["sudo", "-n", "jetson_clocks", "--store", _JCLK_STORE],
                           check=False, capture_output=True, timeout=20)
            subprocess.run(["sudo", "-n", "jetson_clocks"], check=True,
                           capture_output=True, timeout=20)
        elif os.path.exists(_JCLK_STORE):
            subprocess.run(["sudo", "-n", "jetson_clocks", "--restore", _JCLK_STORE],
                           check=True, capture_output=True, timeout=20)
        else:
            for c in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq"):
                try:
                    lo = (c / "cpuinfo_min_freq").read_text().strip()
                    subprocess.run(["sudo", "-n", "tee", str(c / "scaling_min_freq")],
                                   input=lo, text=True, check=True,
                                   capture_output=True, timeout=5)
                except Exception:  # noqa: BLE001
                    pass
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e)}
    TEGRA._refresh_meta()
    return {"ok": True, "jetson_clocks": on}


# ---- Perception mode: real-time NanoOWL detection -----------------------------
# On 8GB the VLM (~3.8GB) and OWL (~1GB + TRT) cannot co-reside, so this is a
# MODE SWITCH: stop one, start the other. The dashboard survives because
# jarvis-voice Wants= (not Requires=) jarvis-vlm. Transition runs in a thread;
# the UI polls owl_up/perception_mode in /metrics.
PERCEPTION = {"on": False, "transition": "", "ts": 0.0}
OWL_HEALTH = "http://127.0.0.1:8086/health"


def _perception_transition(on: bool) -> None:
    try:
        if on:
            PERCEPTION["transition"] = "stopping VLM"
            subprocess.run(["sudo", "-n", "systemctl", "stop", "jarvis-vlm"],
                           check=False, capture_output=True, timeout=30)
            _VLM_UP["flag"] = False
            PERCEPTION["transition"] = "freeing memory"
            subprocess.run(["sudo", "-n", "sh", "-c",
                            "sync; echo 3 > /proc/sys/vm/drop_caches; "
                            "echo 1 > /proc/sys/vm/compact_memory"],
                           check=False, capture_output=True, timeout=20)
            PERCEPTION["on"] = True   # set before start so the watchdog won't fight
            PERCEPTION["transition"] = "starting OWL"
            subprocess.run(["sudo", "-n", "systemctl", "start", "jarvis-owl"],
                           check=False, capture_output=True, timeout=90)
            deadline = time.time() + 90
            while time.time() < deadline:
                try:
                    if httpx.get(OWL_HEALTH, timeout=httpx.Timeout(2, 2, 2, 2)).status_code == 200:
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(3)
            PERCEPTION["transition"] = ""
        else:
            PERCEPTION["transition"] = "stopping OWL"
            subprocess.run(["sudo", "-n", "systemctl", "stop", "jarvis-owl"],
                           check=False, capture_output=True, timeout=30)
            PERCEPTION["on"] = False
            PERCEPTION["transition"] = "starting VLM"
            subprocess.run(["sudo", "-n", "systemctl", "start", "jarvis-vlm"],
                           check=False, capture_output=True, timeout=30)
            PERCEPTION["transition"] = ""
    except Exception as e:  # noqa: BLE001
        PERCEPTION["transition"] = "error"
        print(f"[perception] transition error: {e}", flush=True)


def set_perception(on: bool) -> dict:
    PERCEPTION["ts"] = time.time()
    threading.Thread(target=_perception_transition, args=(on,), daemon=True).start()
    return {"ok": True, "on": on, "transition": "switching"}


# ---- Power management: FULL <-> ECO (cool, voice-wakeable) + OFF/REBOOT ------
# FULL  = MAXN_SUPER + VLM up + captioner on (~20W/70C).
# ECO   = 7W + VLM STOPPED + captioner paused, but dashboard + mic + wake word
#         stay alive so "Hey Jarvis, wake up" works (~7-10W, cool).
# OFF   = poweroff — only the physical button brings it back (no voice/network).
# The 8GB box can't run VLM+OWL; eco/perception/full are one state machine.
POWER = {"state": "full", "transition": "", "ts": 0.0, "auto_eco_idle_s": 900}


def _power_transition(target: str) -> None:
    # NOTE: the VLM is ~all of the power/heat (stopping it drops ~20W → ~7.6W and
    # cools the SoC), so eco = stop the VLM. We do NOT switch nvpmodel: the 7W
    # mode requires a REBOOT on this board, so it isn't a runtime lever. nvpmodel
    # stays MAXN_SUPER; ⚡Turbo (jetson_clocks) is the separate high-usage boost.
    try:
        if target == "eco":
            POWER["transition"] = "entering eco"
            POWER["state"] = "eco"          # set first so the watchdog stands down
            VMEM.set_enabled(False)
            subprocess.run(["sudo", "-n", "systemctl", "stop", "jarvis-vlm"],
                           check=False, capture_output=True, timeout=30)
            _VLM_UP["flag"] = False
            POWER["transition"] = ""
        elif target == "full":
            POWER["transition"] = "waking"
            LAST_USER_ACTIVITY["ts"] = time.time()   # reset idle timer up front
            subprocess.run(["sudo", "-n", "systemctl", "start", "jarvis-vlm"],
                           check=False, capture_output=True, timeout=30)
            deadline = time.time() + 90
            while time.time() < deadline:
                try:
                    if httpx.get(VLM_HEALTH, timeout=httpx.Timeout(2, 2, 2, 2)).status_code == 200:
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(3)
            VMEM.set_enabled(True)
            POWER["state"] = "full"
            POWER["transition"] = ""
            LAST_USER_ACTIVITY["ts"] = time.time()   # fresh idle window so auto-eco won't re-fire
    except Exception as e:  # noqa: BLE001
        POWER["transition"] = "error"
        print(f"[power] transition error: {e}", flush=True)


def set_power(action: str) -> dict:
    """action: eco | wake | shutdown | reboot."""
    POWER["ts"] = time.time()
    if action in ("shutdown", "reboot"):
        verb = "poweroff" if action == "shutdown" else "reboot"
        POWER["transition"] = "shutting down" if action == "shutdown" else "rebooting"
        # fire after a short grace so the HTTP response reaches the client first
        threading.Timer(1.5, lambda: subprocess.run(
            ["sudo", "-n", "systemctl", verb], check=False)).start()
        return {"ok": True, "action": action, "transition": POWER["transition"]}
    if action in ("eco", "wake"):
        target = "eco" if action == "eco" else "full"
        if POWER["state"] == target and not POWER["transition"]:
            return {"ok": True, "state": target, "noop": True}
        threading.Thread(target=_power_transition, args=(target,), daemon=True).start()
        return {"ok": True, "state": POWER["state"],
                "transition": "entering eco" if target == "eco" else "waking"}
    return {"ok": False, "error": "unknown action"}


def wake_if_eco(emit=None) -> bool:
    """If asleep, wake to FULL and block until the VLM is healthy. Returns True
    if a wake happened (caller should treat the turn as resuming after wake)."""
    if POWER["state"] != "eco":
        return False
    if emit:
        emit({"phase": "waking"})
    _power_transition("full")   # synchronous: stops returning until VLM is up
    return True


# Voice power commands are text-matched on the STT transcript (NOT the VLM) so
# they work in ECO where the VLM is stopped. Short phrases only — a long sentence
# is treated as a real question, not a command.
_POWER_PHRASES = [
    ("wake", ("wake up", "wakey", "wake jarvis", "good morning", "full power",
              "boost", "come back", "are you there", "you awake")),
    ("eco", ("eco mode", "power save", "power saver", "go to sleep", "sleep mode",
             "go eco", "stand down", "cool down", "low power", "take a nap", "go to eco")),
    ("shutdown", ("shut down", "shutdown", "power off", "power down",
                  "turn yourself off", "turn off jarvis")),
    ("reboot", ("reboot", "restart yourself", "restart the system", "reboot yourself")),
]


def _route_power_command(text: str) -> str | None:
    t = (text or "").lower().strip().strip(".!?,")
    if not t or len(t) > 60:   # long input = a real question, not a power command
        return None
    for action, phrases in _POWER_PHRASES:
        if any(ph in t for ph in phrases):
            return action
    return None


def _power_reply(action: str) -> str:
    return {
        "wake": "Waking up, sir. One moment while I come fully online.",
        "eco": "Entering eco mode, sir. I'll keep listening — say wake up when you need me.",
        "shutdown": "Powering down, sir. You'll need the physical button to bring me back.",
        "reboot": "Rebooting now, sir. Back in a moment.",
    }.get(action, "")


def gather_metrics() -> dict:
    _ns = TEGRA.snapshot()
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
        "memory_count": MEMORY.count(),
        "live_running": LIVE.running,
        "live_observations": len(LIVE.observations),
        "wake_enabled": WAKE.enabled,
        "wake_score": round(WAKE.score, 3) if WAKE.enabled else 0.0,
        "agent_mode_enabled": bool(SETTINGS.get("agent_mode_enabled")),
        "tool_count": len(jarvis_tools.TOOLS.catalog()),
        "vmem_count": MEMORY.vmem_count(),
        "vmem_enabled": VMEM.enabled,
        "owl_up": jarvis_tools._owl_available(),
        "watch_count": len(MEMORY.watch_list(only_active=True)),
        "uptime_s": round(time.time() - _BOOT_TS),
        # headline Nano telemetry for the always-on rail (full detail at /nano)
        "gpu_util": _ns.get("gpu_util"),
        "emc_util": _ns.get("emc_util"),
        "power_w": (round(_ns["power_total_mw"] / 1000.0, 1)
                    if _ns.get("power_total_mw") else None),
        "cpu_load_avg": (round(sum(c["load"] for c in _ns["cpu"]) / len(_ns["cpu"]))
                         if _ns.get("cpu") else None),
        "power_mode": (_ns.get("meta") or {}).get("power_mode"),
        "jetson_clocks": (_ns.get("meta") or {}).get("jetson_clocks"),
        "throttle_headroom_c": _ns.get("throttle_headroom_c"),
        "vlm_tok_s": _VLM_PERF.get("tok_s"),
        "vlm_ttft_ms": _VLM_PERF.get("ttft_ms"),
        "vlm_autorefresh": VLM_AUTOREFRESH["enabled"],
        "vlm_refreshes": VLM_AUTOREFRESH["refreshes"],
        "perception_mode": PERCEPTION["on"],
        "perception_transition": PERCEPTION["transition"],
        "power_state": POWER["state"],
        "power_transition": POWER["transition"],
    }


# ----- export -----------------------------------------------------------------
def export_markdown() -> str:
    lines = ["# Jarvis session", "",
             f"_exported {time.strftime('%Y-%m-%d %H:%M:%S')}_", ""]
    msgs = history_messages()
    if msgs:
        lines.append("## Current conversation"); lines.append("")
        for m in msgs:
            role = "User" if m["role"] == "user" else "Jarvis"
            content = m["content"] if isinstance(m["content"], str) else "(image)"
            lines.append(f"**{role}.** {content}"); lines.append("")
    pins = MEMORY.pinned()
    if pins:
        lines.append("## Pinned"); lines.append("")
        for p in pins:
            lines.append(f"- _{p.get('question','')}_  →  {p.get('reply','')}")
    return "\n".join(lines) + "\n"


# ----- Training-dataset export ------------------------------------------------
# Turns the on-device visual memory + grounded Q&A into a portable, standards-
# aligned dataset (Open-X / LeRobot-friendly JSONL + frames) with a consent /
# provenance card. This is OBSERVATIONAL vision-language data: there is no
# embodiment and no action labels (a fixed camera does not act). Action-
# conditioned trajectories require the Zip robot drive loop — see DATASET_CARD.
DATASET_EXPORT_VERSION = "jarvis-lab/export-v1"


def _objects_to_list(raw) -> list:
    """visual_memory.objects is stored either as a JSON array or a delimited
    string depending on the caption-parser path — normalise to a list."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(o).strip() for o in raw if str(o).strip()]
    s = str(raw).strip()
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(o).strip() for o in v if str(o).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
    return [p.strip() for p in re.split(r"[;,\n]", s) if p.strip()]


def _dataset_card(name: str, info: dict, consent: dict) -> str:
    c = info["counts"]
    shape = info["features"]["observation.image"]["shape"]
    return f"""# Jarvis-Lab dataset — {name}

_Exported {time.strftime('%Y-%m-%d %H:%M:%S')} from an on-device Qwen2.5-VL-3B
perception loop running fully offline on a Jetson Orin Nano Super (8 GB)._

## What this is

A **vision-language** dataset auto-annotated **at the edge, at capture time** —
every frame ships with a VLM caption and an object list at zero marginal
labeling cost. Two streams:

| File | Records | Contents |
|---|---|---|
| `data/vision_language.jsonl` | {c['vl_records']} | observational keyframes: image + caption + objects + timestamp |
| `data/visual_qa.jsonl` | {c['vqa_records']} | grounded Q&A: image + question + answer |

Frames: {c['frames_copied']} copied, {c['frames_missing']} missing. Image
shape (HxWxC): {shape}.

## Tier — be honest about value

This is **Tier-3 observational vision-language** data, *not* Tier-1 action-
labeled trajectories. A fixed camera does not act, so **`action` is `null`**
throughout. It suits VLM / world-model pretraining, perception, and grounded-
VQA fine-tuning. For action-conditioned (VLA) training, the same auto-
annotation loop must run on the **Zip robot** drive path, recording
`frame -> commanded velocity/heading` as the action label.

## Format

Open-X / LeRobot-friendly JSONL. `meta/info.json` declares the feature schema
and splits; convert to RLDS or a LeRobot `LeRobotDataset` by reading the JSONL
plus the `frames/` images. Each VL row is modeled as a one-step episode.

## Provenance & consent

- Collection mode: **{consent['collection_mode']}**
- Location: {consent['location_label']}
- Operator: {consent['operator']}
- Capture device: {consent['capture_device']}
- Consent obtained: **{consent['consent_obtained']}**
- License: **{consent['license']}**
- PII: {consent['contains_pii']}
- Notes: {consent['notes'] or '—'}

> Review frames for personally identifying content before redistribution.
"""


def export_dataset(opts: dict | None = None) -> dict:
    """Export visual memory + grounded VQA to a portable training dataset.

    opts: since_days (float|None), limit (int, cap 50000),
          include_frames (bool), include_turns (bool), consent (dict).
    Returns a summary dict (also persisted as meta/info.json)."""
    opts = opts or {}
    now = time.time()
    since_days = opts.get("since_days")
    cutoff = (now - float(since_days) * 86400.0) if since_days else None
    limit = max(1, min(50000, int(opts.get("limit", 5000))))
    include_frames = bool(opts.get("include_frames", True))
    include_turns = bool(opts.get("include_turns", True))
    consent_in = opts.get("consent") or {}

    name = f"jarvis-vl-{time.strftime('%Y%m%d-%H%M%S')}"
    root = EXPORT_DIR / name
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    frames_dir = root / "frames"
    if include_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    counts = {"vl_records": 0, "vqa_records": 0,
              "frames_copied": 0, "frames_missing": 0}

    # ---- observational vision-language stream (from visual_memory) ----------
    if cutoff is not None:
        q = ("SELECT id, ts, caption, objects, frame_file, phash, source "
             "FROM visual_memory WHERE ts >= ? ORDER BY ts ASC LIMIT ?")
        args = (cutoff, limit)
    else:
        q = ("SELECT id, ts, caption, objects, frame_file, phash, source "
             "FROM visual_memory ORDER BY ts ASC LIMIT ?")
        args = (limit,)
    with MEMORY.lock:
        vl_rows = MEMORY._con.execute(q, args).fetchall()

    with (root / "data" / "vision_language.jsonl").open("w", encoding="utf-8") as f:
        for idx, (_rid, ts, caption, objects, frame_file, phash, source) in enumerate(vl_rows):
            frame_rel = None
            if frame_file:
                src = VMEM_DIR / frame_file
                if src.exists():
                    if include_frames:
                        try:
                            shutil.copy2(src, frames_dir / frame_file)
                            counts["frames_copied"] += 1
                        except OSError:
                            counts["frames_missing"] += 1
                    frame_rel = f"frames/{frame_file}"
                else:
                    counts["frames_missing"] += 1
            rec = {
                "index": idx,
                "episode_index": idx,   # observational: each keyframe = 1-step episode
                "frame_index": 0,
                "timestamp": ts,
                "observation.image": frame_rel,
                "language.caption": caption or "",
                "language.objects": _objects_to_list(objects),
                "phash": phash,
                "source": source or "ambient",
                "action": None,         # no embodiment — see DATASET_CARD.md
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["vl_records"] += 1

    # ---- grounded visual-QA stream (from turns that have a frame) -----------
    if include_turns:
        with MEMORY.lock:
            t_rows = MEMORY._con.execute(
                "SELECT turn_id, kind, question, transcription, reply, created_at "
                "FROM turns WHERE reply IS NOT NULL AND reply != '' "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        with (root / "data" / "visual_qa.jsonl").open("w", encoding="utf-8") as f:
            for tid, kind, question, transcription, reply, cts in t_rows:
                if cutoff is not None and (cts or 0) < cutoff:
                    continue
                src = SESSION_DIR / tid / "frame.jpg"
                if not src.exists():
                    continue  # grounded VQA needs the image it was asked about
                fname = f"turn-{tid}.jpg"
                frame_rel = f"frames/{fname}"
                if include_frames:
                    try:
                        shutil.copy2(src, frames_dir / fname)
                        counts["frames_copied"] += 1
                    except OSError:
                        counts["frames_missing"] += 1
                        frame_rel = None
                rec = {
                    "turn_id": tid,
                    "kind": kind,
                    "timestamp": cts,
                    "image": frame_rel,
                    "question": (question or transcription or "").strip(),
                    "answer": (reply or "").strip(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["vqa_records"] += 1

    # ---- consent / provenance (the market is rights-sensitive) --------------
    consent = {
        "collection_mode": consent_in.get("collection_mode", "stationary_camera"),
        "location_label": consent_in.get("location_label", "unspecified"),
        "operator": consent_in.get("operator", "unspecified"),
        "capture_device": consent_in.get("capture_device", "Logitech C615 @ 512x384"),
        "consent_obtained": bool(consent_in.get("consent_obtained", False)),
        "license": consent_in.get("license", "unspecified — review before redistribution"),
        "contains_pii": consent_in.get("contains_pii", "unknown — review frames before sharing"),
        "notes": consent_in.get("notes", ""),
    }
    (root / "meta" / "consent.json").write_text(
        json.dumps(consent, indent=2), encoding="utf-8")

    # ---- machine-readable dataset info (LeRobot-ish) ------------------------
    info = {
        "codebase_version": DATASET_EXPORT_VERSION,
        "exported_at": now,
        "robot_type": "observation_only",
        "fps": None,
        "total_episodes": counts["vl_records"],
        "total_frames": counts["vl_records"],
        "total_vqa": counts["vqa_records"],
        "features": {
            "observation.image": {"dtype": "image", "shape": [VLM_H, VLM_W, 3],
                                  "names": ["height", "width", "channel"]},
            "language.caption": {"dtype": "string"},
            "language.objects": {"dtype": "list[string]"},
            "action": {"dtype": "null",
                       "note": "no embodiment — observational data. Action "
                               "labels require the Zip robot drive loop."},
        },
        "splits": {"train": f"0:{counts['vl_records']}"},
        "counts": counts,
        "consent": consent,
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")

    (root / "DATASET_CARD.md").write_text(
        _dataset_card(name, info, consent), encoding="utf-8")

    total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return {"ok": True, "name": name, "path": str(root),
            "counts": counts, "bytes": total_bytes,
            "card_url": f"/dataset/card/{name}",
            "download_url": f"/dataset/dl/{name}.zip"}


def list_dataset_exports() -> list[dict]:
    out = []
    for d in sorted(EXPORT_DIR.glob("jarvis-vl-*"), reverse=True):
        if not d.is_dir():
            continue
        info_p = d / "meta" / "info.json"
        try:
            info = json.loads(info_p.read_text()) if info_p.exists() else {}
        except (json.JSONDecodeError, OSError):
            info = {}
        out.append({
            "name": d.name,
            "exported_at": info.get("exported_at"),
            "counts": info.get("counts", {}),
            "card_url": f"/dataset/card/{d.name}",
            "download_url": f"/dataset/dl/{d.name}.zip",
        })
    return out


# ----- HTML (loaded from disk for editability) --------------------------------
HTML = (LAB / "scripts/jarvis_ui.html").read_text() if (LAB / "scripts/jarvis_ui.html").exists() else ""


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

    def _stream_notifications(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = jarvis_tools.TOOLS.subscribe_notifications()
        try:
            while True:
                try:
                    ev = q.get(timeout=20)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); continue
                self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            jarvis_tools.TOOLS.unsubscribe_notifications(q)

    def do_GET(self):
        p = self.path
        if p == "/":
            self._send_bytes(HTML.encode(), "text/html; charset=utf-8")
        elif p == "/metrics":
            self._send_json(200, gather_metrics())
        elif p == "/nano":
            _snap = TEGRA.snapshot()
            _snap["vlm"] = dict(_VLM_PERF)
            _snap["autorefresh"] = {
                "enabled": VLM_AUTOREFRESH["enabled"],
                "refreshes": VLM_AUTOREFRESH["refreshes"],
                "min_avail_mb": VLM_AUTOREFRESH["min_avail_mb"],
                "last_refresh_ts": VLM_AUTOREFRESH["last_refresh_ts"],
            }
            self._send_json(200, _snap)
        elif p == "/history":
            self._send_json(200, {"messages": history_messages()})
        elif p == "/settings":
            with SETTINGS_LOCK:
                self._send_json(200, dict(SETTINGS))
        elif p == "/presets":
            self._send_json(200, {"presets": PRESETS})
        elif p == "/memory/recent":
            self._send_json(200, {"items": MEMORY.recent(50)})
        elif p.startswith("/memory/search"):
            q = ""
            if "?" in p:
                qs = p.split("?", 1)[1]
                for kv in qs.split("&"):
                    if kv.startswith("q="):
                        from urllib.parse import unquote
                        q = unquote(kv[2:])
            self._send_json(200, {"items": MEMORY.search(q, 50), "q": q})
        elif p == "/memory/pinned":
            self._send_json(200, {"items": MEMORY.pinned()})
        elif p == "/memory/graph":
            self._send_json(200, MEMORY.vmem_graph())
        elif p.startswith("/memory/entity?"):
            from urllib.parse import unquote
            label = ""
            for kv in p.split("?", 1)[1].split("&"):
                if kv.startswith("label="):
                    label = unquote(kv[6:])
            self._send_json(200, MEMORY.entity_detail(label))
        elif p == "/memory/entities":
            ents = MEMORY.vmem_entities()
            # surface the most-sighted entities first, then most-recent — keeps
            # the registry focused (free-text object lists are noisy).
            ents.sort(key=lambda e: (e["count"], e["last_ts"]), reverse=True)
            self._send_json(200, {"entities": ents[:60], "count": len(ents)})
        elif p == "/watch/rules":
            self._send_json(200, {"rules": MEMORY.watch_list()})
        elif p == "/memory/visual/recent":
            self._send_json(200, {"items": MEMORY.vmem_recent(60),
                                  "enabled": VMEM.enabled,
                                  "count": MEMORY.vmem_count()})
        elif p.startswith("/memory/visual/search"):
            from urllib.parse import unquote
            q = ""
            if "?" in p:
                for kv in p.split("?", 1)[1].split("&"):
                    if kv.startswith("q="):
                        q = unquote(kv[2:])
            self._send_json(200, {"items": MEMORY.vmem_search(q, 40), "q": q})
        elif p == "/dataset/exports":
            self._send_json(200, {"items": list_dataset_exports()})
        elif p.startswith("/dataset/card/"):
            name = p.split("?", 1)[0].split("/")[-1]
            if "/" in name or ".." in name:
                self.send_response(400); self.end_headers(); return
            card = EXPORT_DIR / name / "DATASET_CARD.md"
            if not card.exists():
                self.send_response(404); self.end_headers(); return
            self._send_bytes(card.read_bytes(), "text/markdown; charset=utf-8")
        elif p.startswith("/dataset/dl/"):
            # /dataset/dl/<name>.zip — zip the export dir on demand and stream it
            name = p.split("?", 1)[0].split("/")[-1].replace(".zip", "")
            if "/" in name or ".." in name:
                self.send_response(400); self.end_headers(); return
            src = EXPORT_DIR / name
            if not src.is_dir():
                self.send_response(404); self.end_headers(); return
            try:
                zpath = Path(shutil.make_archive(str(src), "zip",
                                                 root_dir=str(EXPORT_DIR),
                                                 base_dir=name))
                self._send_bytes(zpath.read_bytes(), "application/zip",
                                 {"Content-Disposition":
                                  f'attachment; filename={name}.zip'})
            except OSError as e:  # noqa: BLE001
                self._send_json(500, {"error": str(e)})
        elif p == "/export/markdown":
            body = export_markdown().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition",
                             "attachment; filename=jarvis-session.md")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/stream.mjpeg":
            self._stream_mjpeg()
        elif p.split("?")[0] == "/snapshot.jpg":
            try:
                self._send_bytes(CAMERA.get_latest(), "image/jpeg",
                                 {"Cache-Control": "no-store"})
            except TimeoutError:
                self.send_response(503); self.end_headers()
        elif p == "/audio_meter":
            self._stream_audio_meter()
        elif p == "/live/stream":
            self._stream_live()
        elif p.startswith("/events/"):
            tid = p.split("/", 2)[2]
            self._stream_events(tid)
        elif p.startswith("/audio_seg/"):
            # /audio_seg/<turn_id>/<NNN>
            parts = p.split("/")
            if len(parts) >= 4:
                tid = parts[2]; idx = parts[3]
                self._send_file(SESSION_DIR / tid / f"seg_{idx}.wav", "audio/wav")
            else:
                self.send_response(404); self.end_headers()
        elif p.startswith("/audio/"):
            tid = p.split("/")[-1].replace(".wav", "")
            self._send_file(SESSION_DIR / tid / "reply.wav", "audio/wav")
        elif p.startswith("/frame/"):
            tid = p.split("/")[-1].replace(".jpg", "")
            self._send_file(SESSION_DIR / tid / "frame.jpg", "image/jpeg")
        elif p.startswith("/vmem/"):
            # /vmem/<file>.jpg  — visual-memory keyframes
            f = p.split("?", 1)[0].split("/")[-1]
            if "/" in f or ".." in f:
                self.send_response(400); self.end_headers(); return
            self._send_file(VMEM_DIR / f, "image/jpeg")
        elif p.startswith("/inv/"):
            # /inv/<inv_id>/<file>.jpg  — investigate artifacts (full/zoom)
            parts = p.split("?", 1)[0].split("/")
            if len(parts) >= 4:
                d, f = parts[2], parts[3]
                if "/" in d or ".." in d or "/" in f or ".." in f:
                    self.send_response(400); self.end_headers(); return
                self._send_file(SESSION_DIR / "inv" / d / f, "image/jpeg")
            else:
                self.send_response(404); self.end_headers()
        elif p == "/tools":
            self._send_json(200, {"tools": jarvis_tools.TOOLS.catalog()})
        elif p.startswith("/tools/calls"):
            # recent tool-call audit log
            limit = 50
            if "?" in p:
                from urllib.parse import parse_qs
                qs = parse_qs(p.split("?", 1)[1])
                try:
                    limit = max(1, min(500, int(qs.get("limit", ["50"])[0])))
                except ValueError:
                    pass
            with MEMORY.lock:
                rows = MEMORY._con.execute(
                    "SELECT name, args_json, result_json, ok, ms, created_at "
                    "FROM tool_calls ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            self._send_json(200, {"items": [
                {"name": r[0], "args": json.loads(r[1] or "null"),
                 "result": json.loads(r[2] or "null"),
                 "ok": bool(r[3]), "ms": r[4], "ts": r[5]}
                for r in rows
            ]})
        elif p.startswith("/audio_tool/"):
            # /audio_tool/<dir>/<file>.wav  served from session_dir/tools/<dir>/<file>
            parts = p.split("/")
            if len(parts) >= 4:
                d = parts[2]; f = parts[3]
                # strict: no traversal
                if "/" in d or ".." in d or "/" in f or ".." in f:
                    self.send_response(400); self.end_headers(); return
                self._send_file(SESSION_DIR / "tools" / d / f, "audio/wav")
            else:
                self.send_response(404); self.end_headers()
        elif p == "/reminders":
            with MEMORY.lock:
                rows = MEMORY._con.execute(
                    "SELECT id, text, fire_at, fired, fired_at FROM reminders "
                    "ORDER BY fire_at DESC LIMIT 200",
                ).fetchall()
            self._send_json(200, {"items": [
                {"id": r[0], "text": r[1], "fire_at": r[2],
                 "fired": bool(r[3]), "fired_at": r[4]}
                for r in rows
            ]})
        elif p.startswith("/notifications"):
            since = 0
            if "?" in p:
                from urllib.parse import parse_qs
                qs = parse_qs(p.split("?", 1)[1])
                try:
                    since = int(qs.get("since_id", ["0"])[0])
                except ValueError:
                    pass
            self._send_json(200, {
                "items": jarvis_tools.TOOLS.notifications(since_id=since),
            })
        elif p == "/events/notifications":
            self._stream_notifications()
        elif p.startswith("/audio_note/"):
            # /audio_note/reminder-<id>.wav  served from session_dir/notifications/
            parts = p.split("/")
            if len(parts) >= 3:
                f = parts[2]
                if "/" in f or ".." in f:
                    self.send_response(400); self.end_headers(); return
                self._send_file(SESSION_DIR / "notifications" / f, "audio/wav")
            else:
                self.send_response(404); self.end_headers()
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
                          "scene_cache_enabled",
                          "agent_mode_enabled", "agent_max_steps"):
                    if k in payload:
                        SETTINGS[k] = payload[k]
            if "wake_word_enabled" in payload:
                set_wake_enabled(bool(payload["wake_word_enabled"]))
            self._send_json(200, {"ok": True})
        else:
            self.send_response(404); self.end_headers()

    def do_DELETE(self):
        p = self.path
        if p == "/history":
            history_clear()
            self._send_json(200, {"ok": True})
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
            threading.Thread(target=run_turn, args=(ctx, payload),
                             daemon=True).start()
            self._send_json(200, {"turn_id": tid})
        elif p == "/investigate":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            tid = "inv-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            ctx = register_turn(tid, "investigate")
            threading.Thread(target=run_investigate, args=(ctx, payload),
                             daemon=True).start()
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
                SETTINGS["preset"] = "jarvis"
                SETTINGS["max_tokens"] = 240
                SETTINGS["temperature"] = 0.2
                SETTINGS["record_seconds"] = 6
                SETTINGS["live_interval_s"] = 8
                SETTINGS["scene_cache_enabled"] = True
            self._send_json(200, {"ok": True})
        elif p == "/watch/rules":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            txt = (payload.get("text") or "").strip()
            if not txt:
                self._send_json(400, {"error": "text required"}); return
            rid = MEMORY.watch_add(txt)
            self._send_json(200, {"ok": True, "id": rid,
                                  "rules": MEMORY.watch_list()})
        elif p.startswith("/watch/rules/") and p.endswith("/delete"):
            rid = int(p.split("/")[3])
            MEMORY.watch_remove(rid)
            self._send_json(200, {"ok": True, "rules": MEMORY.watch_list()})
        elif p.startswith("/watch/rules/") and p.endswith("/toggle"):
            rid = int(p.split("/")[3])
            cur = {r["id"]: r for r in MEMORY.watch_list()}.get(rid)
            if cur is None:
                self._send_json(404, {"error": "no such rule"}); return
            MEMORY.watch_set_active(rid, not cur["active"])
            self._send_json(200, {"ok": True, "rules": MEMORY.watch_list()})
        elif p == "/watch/test":
            try:
                rec = VMEM.capture_now(source="manual")
                fired = WATCHER.evaluate(Path(rec["frame_path"]))
                self._send_json(200, {"ok": True, "frame_url": rec["frame_url"],
                                      "fired": [{"id": r["id"], "text": r["text"]}
                                                for r in fired]})
            except Exception as e:  # noqa: BLE001
                self._send_json(500, {"ok": False, "error": str(e)})
        elif p == "/dataset/export":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                opts = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            try:
                self._send_json(200, export_dataset(opts))
            except Exception as e:  # noqa: BLE001
                self._send_json(500, {"ok": False, "error": str(e)})
        elif p == "/perception":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            self._send_json(200, set_perception(bool(payload.get("on"))))
        elif p == "/power":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            self._send_json(200, set_power((payload.get("action") or "").strip()))
        elif p == "/nano/autorefresh":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            if "enabled" in payload:
                VLM_AUTOREFRESH["enabled"] = bool(payload["enabled"])
            if "min_avail_mb" in payload:
                VLM_AUTOREFRESH["min_avail_mb"] = max(
                    64, min(1024, int(payload["min_avail_mb"])))
            self._send_json(200, {"ok": True,
                                  "enabled": VLM_AUTOREFRESH["enabled"],
                                  "min_avail_mb": VLM_AUTOREFRESH["min_avail_mb"],
                                  "refreshes": VLM_AUTOREFRESH["refreshes"]})
        elif p == "/nano/jetson_clocks":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            self._send_json(200, set_jetson_clocks(bool(payload.get("on"))))
        elif p == "/memory/visual/capture":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
            # bg=true → background caption (priority=False): yields to interactive
            # turns and skips when the VLM is busy (used by the live ticker refresh)
            bg = bool(body.get("bg"))
            try:
                rec = VMEM.capture_now(source="ticker" if bg else "manual",
                                       priority=not bg)
                self._send_json(200, {"ok": True, "item": rec})
            except Exception as e:  # noqa: BLE001
                self._send_json(500, {"ok": False, "error": str(e)})
        elif p == "/memory/visual/enable":
            VMEM.set_enabled(True)
            self._send_json(200, {"enabled": True})
        elif p == "/memory/visual/disable":
            VMEM.set_enabled(False)
            self._send_json(200, {"enabled": False})
        elif p == "/live/start":
            ok = LIVE.start()
            self._send_json(200, {"ok": ok, "running": LIVE.is_running()})
        elif p == "/live/stop":
            ok = LIVE.stop()
            self._send_json(200, {"ok": ok, "running": LIVE.is_running()})
        elif p == "/wake/start":
            ok = set_wake_enabled(True)
            self._send_json(200, {"ok": ok, "enabled": WAKE.enabled})
        elif p == "/wake/stop":
            ok = set_wake_enabled(False)
            self._send_json(200, {"ok": ok, "enabled": WAKE.enabled})
        elif p == "/agent/enable":
            with SETTINGS_LOCK:
                SETTINGS["agent_mode_enabled"] = True
            self._send_json(200, {"agent_mode_enabled": True})
        elif p == "/agent/disable":
            with SETTINGS_LOCK:
                SETTINGS["agent_mode_enabled"] = False
            self._send_json(200, {"agent_mode_enabled": False})
        elif p.startswith("/memory/") and p.endswith("/pin"):
            tid = p.split("/")[2]
            MEMORY.set_pin(tid, True)
            self._send_json(200, {"ok": True})
        elif p.startswith("/memory/") and p.endswith("/unpin"):
            tid = p.split("/")[2]
            MEMORY.set_pin(tid, False)
            self._send_json(200, {"ok": True})
        elif p == "/agent":
            # POST /agent  body: {question, max_steps?, use_frame?}
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            question = (payload.get("question") or "").strip()
            if not question:
                self._send_json(400, {"error": "question required"}); return
            max_steps = max(1, min(8, int(payload.get("max_steps", 3))))
            # use_frame defaults to "auto" — heuristic on the question text.
            use_frame = payload.get("use_frame", "auto")
            try:
                result = jarvis_tools.run_agent(
                    question, max_steps=max_steps, use_frame=use_frame,
                )
                self._send_json(200, result)
            except Exception as e:  # noqa: BLE001
                self._send_json(500, {
                    "error": str(e), "type": type(e).__name__,
                })
        elif p.startswith("/tool/"):
            # /tool/<name>?confirm=1   body: {args: {...}}
            from urllib.parse import urlsplit, parse_qs
            sp = urlsplit(p)
            name = sp.path[len("/tool/"):]
            qs = parse_qs(sp.query or "")
            confirmed = (qs.get("confirm", ["0"])[0] in ("1", "true", "yes"))
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"}); return
            args = payload.get("args", payload) or {}
            t0 = time.monotonic()
            result = jarvis_tools.TOOLS.call(name, args, confirmed=confirmed)
            ms = round((time.monotonic() - t0) * 1000.0, 1)
            try:
                jarvis_tools.TOOLS.log_call(MEMORY, name, args, result, ms)
            except Exception:
                pass
            status = 200 if result.get("ok") else (
                403 if "requires confirmation" in (result.get("error") or "")
                else 400
            )
            result["ms"] = ms
            self._send_json(status, result)
        else:
            self.send_response(404); self.end_headers()


# ----- server -----------------------------------------------------------------
class JarvisServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128
    allow_reuse_address = True


def main():
    CAMERA.start()
    AUDIO.start()
    AUDIO_MONITOR.start()
    threading.Thread(target=_vlm_health_poller, daemon=True).start()
    threading.Thread(target=_vlm_memory_watchdog, daemon=True).start()
    TEGRA.start()
    VMEM.start()

    # wire up the tool registry
    jarvis_tools.TOOLS.migrate(MEMORY)
    jarvis_tools.TOOLS.set_context(jarvis_tools.ToolContext(
        memory=MEMORY,
        camera=CAMERA,
        audio=AUDIO,
        live=LIVE,
        wake=WAKE,
        visual_memory=VMEM,
        settings=SETTINGS,
        settings_lock=SETTINGS_LOCK,
        presets=PRESETS,
        set_wake_enabled=set_wake_enabled,
        capture_frame=capture_frame_for_vlm,
        stream_vlm=stream_vlm,
        transcribe=transcribe,
        synthesize=synthesize,
        record_voice=record_voice_via_bus,
        lab_root=LAB,
        session_dir=SESSION_DIR,
        vlm_w=VLM_W, vlm_h=VLM_H,
        cam_w=CAM_W, cam_h=CAM_H,
        mic_device=MIC_DEVICE,
    ))
    jarvis_tools.TOOLS.start_reminder_loop(MEMORY)
    print(f"jarvis: {len(jarvis_tools.TOOLS.catalog())} tools registered", flush=True)

    # Boot-to-eco: if the VLM service isn't running at startup (it's disabled from
    # auto-start), we're cool/idle until the first request or "wake up". Use the
    # service state, not a health ping (which races with llama-server warmup).
    try:
        _vlm_running = subprocess.run(
            ["systemctl", "is-active", "jarvis-vlm"],
            capture_output=True, text=True, timeout=5).stdout.strip() == "active"
    except Exception:  # noqa: BLE001
        _vlm_running = False
    if not _vlm_running:
        POWER["state"] = "eco"
        VMEM.set_enabled(False)
        print("jarvis: booted in ECO (VLM stopped) — wake on request", flush=True)

    print(f"jarvis listening on http://{LISTEN_HOST}:{LISTEN_PORT}/", flush=True)
    JarvisServer((LISTEN_HOST, LISTEN_PORT), H).serve_forever()


if __name__ == "__main__":
    main()
