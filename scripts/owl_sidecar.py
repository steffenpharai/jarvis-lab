#!/usr/bin/env python3
"""NanoOWL open-vocabulary detection sidecar.

Runs INSIDE the dustynv/nanoowl container. Exposes a tiny HTTP API the host
Jarvis calls to locate ANY object named in free text — the open-vocab
"find/track anything" capability (OWL-ViT patch32 distilled + TensorRT).

POST /detect  {"image_b64": "...", "query": "a bird, a red car", "threshold": 0.1}
  -> {"detections": [{"label","score","box_xyxy":[..0-1..],"bbox":[x,y,w,h 0-1]}],
      "best": {<top detection or null>}}
GET  /health  -> {"ok": true, "model": "..."}

Coordinates are normalised 0-1 to the input image.
"""
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import PIL.Image

from nanoowl.owl_predictor import OwlPredictor

ENGINE = os.environ.get("OWL_ENGINE",
                        "/data/owl_image_encoder_patch32.engine")
MODEL = os.environ.get("OWL_MODEL", "google/owlvit-base-patch32")
PORT = int(os.environ.get("OWL_PORT", "8086"))

print(f"[owl] loading {MODEL} (engine={ENGINE}) ...", flush=True)
PREDICTOR = OwlPredictor(MODEL, image_encoder_engine=ENGINE)
_LOCK = threading.Lock()
print("[owl] ready", flush=True)


def detect(image: PIL.Image.Image, query: str, threshold: float) -> list:
    texts = [t.strip() for t in query.split(",") if t.strip()] or [query]
    with _LOCK:
        enc = PREDICTOR.encode_text(texts)
        out = PREDICTOR.predict(image=image, text=texts, text_encodings=enc,
                                threshold=threshold, pad_square=False)
    W, H = image.size
    dets = []
    boxes = out.boxes.detach().cpu().tolist()
    labels = out.labels.detach().cpu().tolist()
    scores = out.scores.detach().cpu().tolist()
    for box, li, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        nx1, ny1, nx2, ny2 = x1 / W, y1 / H, x2 / W, y2 / H
        dets.append({
            "label": texts[int(li)] if 0 <= int(li) < len(texts) else str(li),
            "score": round(float(sc), 3),
            "box_xyxy": [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
            "bbox": [round(max(0.0, nx1), 4), round(max(0.0, ny1), 4),
                     round(min(1.0, nx2 - nx1), 4), round(min(1.0, ny2 - ny1), 4)],
        })
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model": MODEL})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/detect":
            self._json(404, {"error": "not found"}); return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"}); return
        query = (body.get("query") or "").strip()
        if not query:
            self._json(400, {"error": "query required"}); return
        thr = float(body.get("threshold", 0.1))
        try:
            raw = base64.b64decode(body["image_b64"])
            img = PIL.Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": f"bad image: {e}"}); return
        try:
            dets = detect(img, query, thr)
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": f"{type(e).__name__}: {e}"}); return
        self._json(200, {"detections": dets[:20],
                         "best": dets[0] if dets else None})


if __name__ == "__main__":
    print(f"[owl] serving on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
