# Jarvis Lab

Wearable conversational AI for Zip: a Jetson Orin Nano Super (8GB) running
in a backpack, paired to the user via Bluetooth audio (Pixel Buds 2 / PC
speakers in dev), with a camera. The user talks to it like Jarvis from
Iron Man — and can ask it about anything in the scene.

## Use case

- Always-on conversational voice agent ("Hey Jarvis, ...")
- On-demand vision: "what does that storefront say?" → the VLM looks at
  the current frame and answers
- General scene QA: people, cars, dogs, stores, signs, anything visible

## Why separate from `~/zip` and `~/.openclaw`

- `~/zip` = mobile-robot stack (drive/perception/mapping). Wrong shape
  for a wearable.
- `~/.openclaw` = the earlier OpenClaw agent attempt. Too heavy for 8GB
  (~8-16k base prompt) and too generic for this use case.
- `jarvis-lab` = the new application stack. VLM-first, voice-loop second.

## Top-level layout

- `models/qwen2.5-vl-3b/` — the VLM weights (GGUF) + vision projector
- `scripts/` — runtime helpers: VLM server, capture, ask, benchmark
- `test_images/` — three reference frames for repeatable benchmarking
- `logs/` — runtime logs
- `ARCHITECTURE.md` — the load-bearing decisions and why
