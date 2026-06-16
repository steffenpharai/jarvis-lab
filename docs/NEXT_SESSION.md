# Next Session — three.js JARVIS animation + HUD animations

**Focus:** make the companion **orb** feel alive (the thing you talk to, like
Iron Man's JARVIS) and polish the other animations. This is a *creative/visual*
session — everything below renders on the **viewer's GPU**, so it costs the
Jetson nothing; iterate freely.

Everything visual lives in **one file**: [`scripts/jarvis_ui.html`](../scripts/jarvis_ui.html).

---

## Where the animations are

### The companion orb (the centerpiece) — `#presence`
- Markup: `<canvas class=presence id=presence>` inside `#feedwrap` (top-center of the feed).
- Code: the three.js `<script type="module">` near the **bottom** of the file
  (search `three.js holographic voice-presence orb`). Today it is:
  - an **additive fresnel `ShaderMaterial`** icosahedron **core** that deforms
    with voice level (vertex-shader noise displacement),
  - a **particle halo** (additive points on a shell, fewer on mobile),
  - a **glow sprite**,
  - color **eased per voice state** (`COLORS`: idle rust / listening cyan /
    thinking amber / speaking rust-orange / alert red),
  - an idle breathing baseline, rotation, scale-with-level.
- **State bridge:** `window.jarvisVoice = { level: 0..1, state: 'idle'|'listening'
  |'thinking'|'speaking'|'alert' }`. `level` comes from the `/audio_meter` SSE;
  `state` is set by `setLive()`/the turn lifecycle. Drive *new* behaviour off this.
- `window.__orbReady` = true when the orb initialised (DOM-verifiable).

### Other animations (mostly CSS `@keyframes` in the `<style>` block)
- **Edge-glow presence** (`.ambient`, `ambientBreathe/Listen/Think/Speak/Alert`)
  — full-viewport border that shifts with voice state.
- **Boot/power-on** sequence (`.boot`, `bootSpin/bootPulse/bootFill`) — once per
  session via sessionStorage `jarvis_booted`.
- **Radar scan sweep** over the feed (`.scansweep`, `radarSpin`).
- **Targeting reticle** (`.reticle`, `reticleScan`) — the investigate locate box.
- **3D point-cloud** module (separate three.js module, `window.openPointCloud`)
  — has a control suite (color modes / density / live re-scan / depth / fps).
- Misc: `pulse`, `breathe`, `wake-ring`, `wake-breathe`, `tap-pulse`,
  `scene-flash`, `dtflash` (detection ticker), `msgIn`, `toastIn`, `stageScan`.

---

## Hard constraints (don't relearn these)

- **Viewer-GPU only.** three.js loads via a **CDN importmap** (`three@0.169.0`);
  the Jetson only serves the HTML. Zero extra Jetson load — animate freely.
  *(Offline/AP caveat: vendor `three.module.js` locally if the viewer has no net.)*
- **`WebGLRenderer`, NOT WebGPU** — phone/browser reliability.
- **No post-processing / bloom.** Use **additive blending** for glow (bloom was
  rejected for robustness). The orb already does this.
- **Feature-detect → silent fallback.** No WebGL → orb hides, `__orbReady=false`,
  the CSS HUD carries on. Keep this pattern for any new WebGL.
- **Perf guards:** DPR capped (2 desktop / 1.5 mobile), fewer particles on mobile,
  `requestAnimationFrame` **paused on `visibilitychange`**. Match these.
- **`prefers-reduced-motion`** is honoured throughout — gate new motion behind it.
- **Design language:** Apple "Liquid Glass" + Palantir density + restraint. Single
  rust accent (`--accent`) + semantic state colors. Neon-everywhere reads amateur.

---

## Verification (important)

- **`preview_screenshot` HANGS on this page** — the persistent SSE streams
  (`/audio_meter`, notifications) + continuous WebGL RAF never let the
  network/compositor go idle. **Verify via `preview_eval`** (DOM, computed style,
  `canvas.getContext(...).getImageData()` pixel sampling, `window.__orbReady`).
- `preview_start('jarvis-voice')` (the server resets between Claude-Code sessions).
- To capture a still if ever needed: freeze the RAF + swap `#liveimg` to
  `/snapshot.jpg`, then restore.
- Deploy loop: edit here → `scp` to `zip-jetson:~/jarvis-lab/scripts/` →
  `sudo systemctl restart jarvis-voice` (HTML is read from disk at startup) →
  reload the preview. Commit on the PC clone, push, then `git reset --hard` on the
  Jetson.

---

## Ideas to explore (not prescriptive)

**The orb** — make it read as a presence reacting to *you*:
- Distinct per-state behaviour beyond color: a **thinking** particle swirl/vortex,
  **speaking** amplitude that tracks the actual TTS envelope (lip-sync feel),
  a **wake "bloom"** burst when "Hey Jarvis" fires, **listening** ripple.
- Richer core: layered shells, iris/aperture, ferrofluid spikes, refraction.
- **Power-state transitions** (now that power mgmt exists): **eco** = orb dims +
  slows to a slow pulse; **waking** = spin-up/ignition; **full** = alive. Hook off
  `power_state` in `/metrics` (or extend `window.jarvisVoice`).
- Investigate: a focused **scan beam** from the orb to the located bbox.

**Other animations:**
- Smooth **state-transition** choreography (idle→listening→thinking→speaking).
- Eco/wake visual on the feed (the orb + edge-glow telling the power story).
- Polish the boot sequence; subtle entrance/exit for panels & overlays.

Keep it tasteful: restraint + physics-based easing > more neon.
