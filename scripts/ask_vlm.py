#!/usr/bin/env python3
"""Ask the local Jarvis VLM a question about the current camera frame.

Usage:
    ask_vlm.py "what is in this scene?"
    ask_vlm.py --image path.jpg "..."
    ask_vlm.py --no-capture "tell me a joke"   # text-only
"""
import argparse, base64, json, subprocess, sys, time, urllib.request

VLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
SYSTEM = (
    "You are Jarvis, the user's personal AI agent. You see what the camera "
    "sees and you hear what the user says. Answer briefly, conversationally, "
    "and helpfully. If asked to read a sign, transcribe it exactly. If unsure, "
    "say so. Keep responses under 60 words unless explicitly asked for more."
)

def capture_frame() -> str:
    p = subprocess.run(["/home/zip/jarvis-lab/scripts/capture_frame.sh"],
                       capture_output=True, text=True, check=True)
    return p.stdout.strip()

def jpeg_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"

def ask(prompt: str, image_path: str | None) -> None:
    user_content: list = []
    if image_path:
        user_content.append({"type": "image_url",
                             "image_url": {"url": jpeg_to_data_url(image_path)}})
    user_content.append({"type": "text", "text": prompt})

    body = {
        "model": "jarvis-vlm:latest",
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        VLM_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first_token_t = None
    tokens = 0
    print("[Jarvis] ", end="", flush=True)
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
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
            if first_token_t is None:
                first_token_t = time.monotonic()
            tokens += 1
            print(delta, end="", flush=True)
    t_end = time.monotonic()
    if first_token_t is None:
        print("\n[no tokens received]")
        sys.exit(2)
    print()
    cold = (first_token_t - t0) * 1000
    gen = t_end - first_token_t
    rate = tokens / gen if gen > 0 else 0.0
    print(f"\n[bench] cold_first_token={cold:.0f}ms  tokens={tokens}  "
          f"gen={gen:.2f}s  rate={rate:.2f} tok/s", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--image", help="path to JPEG; default = capture from /dev/video0")
    ap.add_argument("--no-capture", action="store_true",
                    help="text-only, no camera frame")
    args = ap.parse_args()

    if args.no_capture:
        image_path = None
    elif args.image:
        image_path = args.image
    else:
        image_path = capture_frame()
        print(f"[captured] {image_path}", file=sys.stderr)
    ask(args.prompt, image_path)

if __name__ == "__main__":
    main()
