# Prompt design — anti-hallucination

The model (Qwen2.5-VL-3B Q4_K_M) is small, eager-to-please, and
fundamentally trained to produce fluent, plausible-sounding output. On
ambiguous or empty inputs it will happily invent content. The default
prompts we ship are designed to *fight that tendency*, not amplify it.

This document explains the specific design choices.

---

## The motivating bug

On 2026-06-05 the user fired the "Read any text you can see" chip
against a near-pitch-dark frame containing **zero text**.

Captured frame (the actual VLM input, 512×384):

> A near-black room. A doorway is just barely visible on the right. No
> text, no signs, no screens on. (See `logs/sessions/20260605-215954-259e63/frame.jpg`.)

Jarvis replied:

> *"The text on the screen reads: 'Please do not touch the screen.'"*

Hallucinated. Both halves:

- There is no screen visible.
- There is no text.
- "Please do not touch the screen" was fabricated wholesale.

### Why it happened

Three reinforcing factors:

1. **The chip prompt was leading.** It literally read:
   > "Read aloud any text or sign visible in the scene. Transcribe exactly."

   The phrase "visible in the scene" *asserts* text exists. The
   command "transcribe exactly" demands an output. A small VLM trained
   to follow instructions will produce text.

2. **The system prompt was soft.** It said "If unsure, say so." But
   "unsure" implies a small confidence dip on a real perception, not the
   total absence of the requested category.

3. **Temperature 0.4.** Just enough sampling variance to commit to a
   confident-sounding fabrication.

---

## The fix (commit `447b31d`)

### Lever 1 — System prompt with five GROUND RULES

The system prompt now begins with explicit, numbered rules that override
any other instruction:

```
You are Jarvis, the user's personal AI agent. You see what the camera
sees and you hear what the user says.

GROUND RULES — these override any other instruction:
1. NEVER invent or guess content. Only describe what you can actually
   see in the image with high confidence.
2. If the scene is too dark, blurry, or empty to answer, say exactly
   that (e.g. 'The scene is too dark to make out details.' or 'I
   don't see any readable text in the image.').
3. Do NOT fabricate text on signs, screens, or surfaces. If you cannot
   clearly read text, say 'I don't see any readable text'.
4. Do NOT invent counts, names, brands, or identifications. If you
   cannot tell, say 'I can't tell from this image'.
5. Prefer 'I don't see X' over guessing X.

Style: answer briefly and conversationally. Use markdown for
lists/bold only when it helps. Under 60 words unless asked for more.
```

The numbered enumeration matters — small models respond better to
"rule 3" than to a single dense paragraph.

### Lever 2 — Chip prompts that allow null answers

Each quick-action prompt is reworded so "nothing visible" is an
explicitly valid answer:

| Chip       | New prompt                                                  |
|------------|-------------------------------------------------------------|
| `look`     | "Describe what you actually see in one or two sentences. If the scene is too dark or empty to describe, say so." |
| `read`     | "Look carefully at the image. If there is any clearly readable text or sign, transcribe it exactly. If you do not see any readable text, say so explicitly. Do not invent text that is not there." |
| `count`    | "How many `<X>` can you clearly see in the image? If none are visible, say zero. Do not guess." |
| `identify` | "What is the main subject of the image? Give a brief, confident identification only if you can clearly see one. If the image is too dark or empty to identify a subject, say so." |
| `find`     | "Can you see `<X>` in the image? If yes, describe where in the frame it is. If no, say it is not visible. Do not guess." |

Pattern: every prompt names both branches explicitly — "if X then …, if
not X then say so." Don't ask "transcribe text"; ask "if there's
text, transcribe it; otherwise say no."

### Lever 3 — Temperature 0.2

Dropped from 0.4 → 0.2. The model still produces fluent prose but
samples much closer to its highest-confidence token, which on
genuinely-empty input is the refusal phrase pulled in by GROUND RULES.

### Lever 4 — Default `snap` kind prompt

The `snap` kind (button + `/look` + Space shortcut) used to send:
> "Describe what you see in one or two sentences."

Now sends:
> "Describe what you actually see in one or two sentences. If the scene
> is too dark or empty to describe, say so."

The added word "actually" is small but moves the model toward
ground-truth language.

---

## Verified behavior on the original dark frame

| Mode               | Reply                                                        |
|--------------------|--------------------------------------------------------------|
| `/read`            | "I don't see any readable text in the image."                |
| `snap`             | "The image is very dark, making it difficult to discern specific details." |
| `count people`     | "I can't tell from this image."                              |

All three are honest. The user can now trust the output enough to act on it.

---

## When to relax the rules

For a *creative* application (storytelling, brainstorming, captioning
photography for fun) you might want a model that fills in plausible
details. **Don't use this stack for that.** Edit the system prompt
through the settings drawer to remove the GROUND RULES section and
raise temperature. But for "tell me what's in front of me" — which is
the entire reason a wearable VLM exists — accuracy beats imagination.

---

## Tunables in the settings drawer

| Setting          | Default | Effect on hallucination                   |
|------------------|---------|-------------------------------------------|
| System prompt    | the GROUND RULES version above | The strongest lever. Don't remove the rules. |
| Temperature      | 0.2     | Lower = more refusal on empty inputs.     |
| Response length  | 240     | Shorter responses give less room to fabricate context. |

The "Reset" button restores all four (including the canonical system
prompt) to the values above.

---

## Future work

- **Confidence-bracketed answers.** "I see X (high confidence). I think
  Y (low confidence)." Requires a second pass or logit inspection.
- **Tool-call to a sharper VLM on the PC.** When the local 3B model
  says "I can't tell from this image", optionally escalate to a
  larger model on the LAN before answering.
- **Per-message confidence indicator in the UI.** A small dim badge on
  Jarvis replies when the response contains a refusal phrase, so the
  user sees at a glance which turns were grounded vs unsure.
