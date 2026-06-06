# Next Session — Agentic Jarvis for 2026

Where to take this from here. The current build is a working
wearable VLM with persistent memory, sentence-streaming TTS,
Hey Jarvis wake word, perceptual frame caching, live narration,
and a frontier-grade dashboard. Everything in §11 of `ARCHITECTURE.md`
that was marked "future" has shipped *except* the genuinely
agentic layer.

This document is the roadmap for that layer — turning Jarvis from
an excellent *conversational* VLM into something closer to
**Iron Man's Jarvis: a personal agent that takes action**.

---

## The shape of "true Jarvis"

Iron Man's Jarvis isn't really about voice or vision. It's about
**agency over Tony's world**: controls the suit, runs simulations,
files patents, calls Pepper, monitors threats. Built on top of
perception and conversation, but the value is *acting on the user's
behalf in a long-running context*.

To get there from where we are, we need five things we don't have
yet:

1. **Tool use** — Jarvis can call functions, not just describe scenes
2. **Cloud-frontier escalation** — when local 3B falls short, hand
   off to Claude 4 / GPT-5 / Gemini with the same context
3. **Long-term memory with semantic recall** — beyond the 3-pair
   conversation buffer; remember "the coffee shop on 6th" forever
4. **Persistent agent identity** — Jarvis knows the user's name,
   preferences, schedule, contacts, goals; learns over time
5. **Multi-modal reasoning loop** — not one VLM call but a
   plan → act → observe → re-plan loop with tool calls in between

---

## Sprint A — Tool use on the local VLM (1-2 sessions)

Qwen2.5-VL doesn't natively output function calls in OpenAI format,
but llama.cpp + jinja templating supports tool use via prompted
schema. The pattern:

1. System prompt advertises the tool catalog (JSON schema + descriptions)
2. Model is asked to either reply in natural language OR emit a
   `<tool_call>{"name":"X","args":{...}}</tool_call>` token sequence
3. Server detects the tool call, dispatches to the tool handler,
   re-prompts the model with the tool result, loops until the model
   answers without a tool call

**Implementation path:**
- Add a `ToolRegistry` class to `jarvis_voice.py`
- Each tool is a Python function with a JSON schema + docstring
- New `tool_use` mode in `run_turn` enables the loop
- Update system prompts to advertise available tools

**First batch of tools to implement:**
- `web_search(query)` — duckduckgo or similar lightweight API
- `get_time()` — return current time, timezone
- `get_weather(location)` — OpenWeatherMap (free tier)
- `do_math(expression)` — Python eval in a sandbox
- `set_reminder(when, what)` — write to SQLite reminders table
- `recall(query)` — semantic search of long-term memory (see Sprint C)

**UI:** show tool calls inline in the conversation with a small
collapsible "tool used: web_search → 312 chars" card.

---

## Sprint B — Cloud-frontier escalation (1 session)

A single MCP-style HTTP tool: `/escalate` that takes the user's
question + the current frame and forwards to Claude 4 / GPT-5 /
Gemini 3 via API. Returns the higher-quality answer.

**Trigger options:**
- Manual: per-message button "Ask the big model" on any Jarvis reply
- Auto: when the local reply contains a refusal phrase
  ("I can't tell from this image", "I don't see any readable text")
  AND the user has opted in
- Slash command: `/ask-frontier <question>`

**Implementation path:**
- Add `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` env
- New endpoint `POST /escalate {question, turn_id?}` that includes
  the frame from `turn_id` (or current snapshot) and forwards to
  the configured provider
- Result rendered in the same Jarvis bubble style with a small
  "via Claude 4 Opus" badge
- Token cost displayed (use the same `prompt_tokens / completion_tokens`
  pattern from the local path)

**Provider choice:**
- Default to **Claude 4** for everyday questions (best instruction
  following + vision)
- **GPT-5** for code-heavy queries
- **Gemini 3** for grounded web answers

This is the highest-leverage feature for actual daily use. Local
Qwen2.5-VL is great for "what is this" but a frontier model is
required for "tell me everything about this storefront."

---

## Sprint C — Long-term memory + semantic recall (1-2 sessions)

The 3-pair conversation buffer is great for "what about it" follow-ups
but loses everything after a few turns. True Jarvis needs:

**A. Vector store over the existing SQLite turns table.**
- Add `embedding BLOB` column to `turns`
- On each turn complete, embed the (question, reply) pair using a
  small local model (e.g. `all-MiniLM-L6-v2` ONNX, 25 MB, runs on CPU)
- New tool: `recall(query, k=5)` does semantic search via numpy
  cosine similarity over all embeddings
- New endpoint `GET /memory/recall?q=...` for the UI

**B. Episode summarization.**
- Background worker scans recent turns, summarizes 10-20 turn
  episodes via the local VLM into a single "episode" record
- Episodes are themselves embedded and searchable
- Reduces the memory haystack from "all turns ever" to "all episodes"
  which is much faster to recall over

**C. Entity extraction.**
- After each turn, extract people / places / things via the VLM in
  a structured `{"entities": [{"name": "Pepper", "type": "person"}]}`
- New `entities` table; surface as a sidebar in the UI

**D. Manual notes.**
- A "Jarvis, remember this" command saves a free-form note to a
  `notes` table — also embedded and searchable

This unblocks "what was that thing I asked you about last Tuesday"
which is when Jarvis stops being a tool and starts being a companion.

---

## Sprint D — Persistent agent identity (1 session)

A simple but powerful sub-sprint. The system prompt should be
*per-user* and learn over time.

**Implementation path:**
- `/profile` slash command shows the current learned preferences
- Editable text area: name, calling style, locations of interest,
  recurring contexts ("I'm a software engineer", "I have a 3-year-old
  daughter named Nora", "I commute by bike")
- Inject this profile into every system prompt as a separate section
- The VLM can write to the profile via a tool call:
  `update_profile(key, value)` — gated by the user via a small toast
  ("Jarvis wants to remember: 'commutes by bike'. Allow?")

The Persona Presets we already have (focused / inspector / companion /
curator) become *modes on top of* the persistent identity, not
replacements for it.

---

## Sprint E — Multi-modal reasoning loop (2+ sessions)

The biggest one. Instead of one VLM call per turn, structure each
turn as a loop:

```
turn:
  1. plan: ask the VLM (or escalate) "what should I do for this?"
  2. for each step in plan:
     a. act: call the chosen tool with the chosen args
     b. observe: append the tool result to working memory
     c. re-plan: ask the VLM "given what we have, what next?"
  3. terminate when the VLM emits a final answer
```

This is the standard ReAct / agentic loop. Frontier products
(Claude 4 agents, GPT-5 operators, Manus, Devin) all use variations.

**Hardware constraints:**
- Each loop step is a VLM call (~3-5s on the local 3B at 4k ctx)
- A 5-step plan = 15-25s
- For escalated frontier calls, parallel network latency dominates

**UI implication:**
- The chat pattern stops being "one question → one answer"
- We need: a "step" timeline within each turn showing plan / tool
  call / observation / next plan
- Maybe collapsible to keep the conversation feed clean

This is where the "Jarvis" feeling really emerges — when the user
says "find me a thai place that's still open" and Jarvis searches,
checks hours, surfaces options, and offers to call — all from one
sentence.

---

## Sprint F — Voice loop polish (~1 session, parallel)

These aren't agentic but they make daily use feel premium:

- **Bluetooth A2DP** to Pixel Buds 2 (already on the deferred list)
  — needs hardware testing; bluez 5 + pipewire on JetPack 6 should
  Just Work but the HFP mic path is known-bad (use C615 mic instead)
- **VAD-tuned wake-to-record** — current 1.2s silence cutoff might
  be too eager for "Hey Jarvis ... uh ... what time is it?"; consider
  a longer initial-grace window
- **Interrupt-while-speaking** — let user say something during TTS
  playback to cancel it (Web Audio cancel + stop the segment queue)
- **Continuous voice mode** — like ChatGPT advanced voice; once
  enabled, no need to say "Hey Jarvis" each turn

---

## Sprint G — Hardware-aware features (deferred until backpack deploy)

These wait until the Jetson is actually in a backpack:

- IMU integration → "Jarvis, what direction am I facing?"
- GPS / location → "where am I?" + Places API integration
- Battery monitoring → "how much battery do we have left?"
- Thermal-aware throttling → spawn lighter prompts when SoC > 70°C
- Phone tether detection → cloud-frontier escalation gated on
  network availability

---

## Order I'd run them

1. **Sprint B (cloud frontier)** — highest daily-use value, smallest
   surface area; gives us a fallback that makes everything else
   shippable
2. **Sprint A (local tool use)** — most agentic feel, foundational
   for D and E
3. **Sprint D (persistent identity)** — small, high-leverage,
   unlocks "Jarvis knows me"
4. **Sprint C (long-term memory + recall)** — needs A's tool layer
   to be useful; semantic search is a separate research question
5. **Sprint E (agentic loop)** — needs A, B, C; the capstone
6. **Sprint F (voice polish)** — can run in parallel with B/A
7. **Sprint G (hardware)** — when the box is in the backpack
