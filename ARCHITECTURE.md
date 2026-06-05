# Jarvis Lab — Architecture (verified 2026-06-05)

## Hardware

- **Jetson Orin Nano Super 8GB** on backpack power (target: 100Wh pack, ~6h)
- **USB UVC camera** Logitech C615 (MJPEG up to 1280x720)
- **Bluetooth audio** -> Pixel Buds 2 (dev = PC speakers via Jetson)
- **Mic** -> TBD: USB lavalier favored (BT HFP is 8/16kHz NB-SCO, wrecks ASR)
- **Network** -> phone tether for cloud-frontier escalation when online

## VRAM budget (Orin Nano 8GB shared LPDDR5)

After cleanup + multi-user.target: **~6.8 GB RAM available**, GPU effectively free.
Peak with VLM loaded: 3.86 GB on GPU, ~3 GB available RAM.

**Decision: Mode A — VLM is the only reasoner.**

Resident set (verified):
- Qwen2.5-VL-3B Q4_K_M + Q8_0 mmproj  ~3.86 GB GPU
- Headroom for KV cache, activations  ~700 MB

Future: voice loop (~2 GB Parakeet, openWakeWord, Kokoro) coexists if model
shrinks slightly or KV cache stays small. Current 4096 ctx is the sweet spot.

## The VLM — Qwen2.5-VL-3B-Instruct, Q4_K_M GGUF, native llama.cpp CUDA

VERIFIED on Orin Nano Super (2026-06-05):
- Build:        `b1-308f61c` from upstream main, CUDA sm_87, FlashAttention on
- Weights:      `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (1.9 GB)
- Vision proj:  `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf` (845 MB)
- License:      Qwen Research (commercial-OK with conditions)

VERIFIED throughput on real C615 frames:

| Image | Vision encoder | Total prefill | Generation |
|---|---:|---:|---:|
| 512x384 (default) | 580-1054 ms | ~1.0 s | 22.7 tok/s |
| 1280x720 (HD)     | 5359-5516 ms | ~5.6 s | 22.5 tok/s |

Quality verified:
- Scene description: accurate room/object descriptions
- OCR: read "BEWARE" from a small sign in the C615 frame
- Hallucination risk: hallucinated "12:30" sign in one frame; mitigation = system prompt asks model to say "I am unsure" if uncertain

Fallback: InternVL3-2B Q4_K_M, same runtime, smaller, MIT licensed.

## TEGRA CMA PREFLIGHT — REQUIRED

On Jetson Orin Nano 8GB with iGPU shared LPDDR5, contiguous CUDA buffers > 500MB
fail with `NvMapMemAllocInternalTagged error 12` if CmaFree is low, even when
total free RAM is plenty. The mmproj load needs ~800 MB contiguous.

ALWAYS run before loading the VLM:

    sudo sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches
    echo 1 | sudo tee /proc/sys/vm/compact_memory

Baked into `scripts/start_vlm_native.sh`. Before: CmaFree=6948 kB ABORT.
After: CmaFree=219204 kB SUCCESS.

## The pipeline (initial — wearable always-on)

```
Mic -> openWakeWord -> Parakeet-v3 (chunked STT) -----+
                                                       v
                                                Qwen2.5-VL-3B
USB UVC cam -> capture_frame.sh @ 512x384 ---------> /v1/chat/completions
                                                       |
                                                       v
                                                  Kokoro TTS -> BT audio
```

Resolution policy:
- Default: 512x384 capture, ~1s prefill -> conversational
- On request: 1280x720 capture, ~5.6s prefill -> OCR / fine detail mode
- Set via JARVIS_FRAME_W / JARVIS_FRAME_H env vars in capture_frame.sh

## End-to-end Jarvis turn budget (target)

- Wake-word + Parakeet STT:    ~600 ms
- VLM prefill + first token:  ~1000 ms  (512x384)
- VLM generation ~30 tokens:  ~1300 ms
- Kokoro TTS first audio:      ~500 ms
- TOTAL "what?" -> first spoken word: ~3.4 s

## Hybrid escalation (later)

For "tell me about that storefront" where local VLM extracts name but needs
world knowledge: tool-call to Places API or cloud frontier VLM over phone
tether. MCP-first design pattern (RuView SENSE-BRIDGE template).

## What v1 ships

1. [x] Native llama.cpp llama-server with Qwen2.5-VL-3B GGUF + mmproj GPU
2. [x] CLI tool: ask_vlm.py captures frame + asks
3. [x] Verified benchmark on real C615 frames
4. [x] CMA preflight in startup script
5. [ ] systemd unit for auto-start (next)
6. [ ] Voice loop bolt-on (next)
