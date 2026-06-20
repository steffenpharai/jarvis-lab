# Zip — Android Companion

Phone-only Android virtual AI companion. The camera, microphone, and screen are the body;
the on-device perception pipeline + behavior engine + LLM are the mind. Built section-by-section
against [`../looi-complete-spec.md`](../looi-complete-spec.md) (renamed from "LOOI" → "Zip"
throughout the codebase; package is `com.zip.companion`).

## Status

**All 24 spec sections implemented. Debug APK builds clean. All 21 unit tests pass.**

| §  | Area                  | Status |
|----|-----------------------|--------|
| 01 | Project setup         | ✅      |
| 02 | Architecture overview | ✅      |
| 03 | Data models           | ✅      |
| 04 | Perception pipeline   | ✅      |
| 05 | Behavior engine       | ✅      |
| 06 | Memory system         | ✅      |
| 07 | LLM integration       | ✅      |
| 08 | Voice pipeline        | ✅      |
| 09 | Design system         | ✅      |
| 10 | Character rendering   | ✅      |
| 11 | Screens               | ✅      |
| 12 | Navigation            | ✅      |
| 13 | Animation system      | ✅      |
| 14 | Onboarding            | ✅      |
| 15 | Chat feature          | ✅      |
| 16 | Personality screen    | ✅      |
| 17 | Ambient mode          | ✅      |
| 18 | Needs system UI       | ✅      |
| 19 | Background workers    | ✅      |
| 20 | Notifications         | ✅      |
| 21 | Settings              | ✅      |
| 22 | Testing strategy      | ✅      |
| 23 | Performance           | ✅      |
| 24 | Build & release       | ✅      |

## Quick start

### Prereqs (already installed on this machine)
- **Android SDK**: `C:\Users\phara\AppData\Local\Android\Sdk`
- **Java 17 (JBR)**: `C:\Program Files\Android\Android Studio\jbr`
- **Gradle 8.9** (auto-fetched by the wrapper)

`local.properties` is pre-populated with the SDK path.

### Build from the command line
```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

cd C:\Startup\zip\zip-android
.\gradlew.bat assembleDebug             # → app\build\outputs\apk\debug\app-debug.apk
.\gradlew.bat testDebugUnitTest         # runs all unit tests
.\gradlew.bat installDebug              # installs onto a running emulator/device
```

### Run from Android Studio
Open `C:\Startup\zip\zip-android\` → sync → run on emulator or device.
Target: Pixel 10 Fold Pro for hardware testing; any API 34/35 AVD for sim work.

### Sim-first development
`PerceptionEventGenerator` (debug-only) drives the whole behavior chain without a real
camera. Four scenarios: `AMBIENT`, `SOCIAL`, `OBJECTS`, `RAPID_FIRE`. Call directly from
the CompanionViewModel or wire to a dev panel to validate the FSM, needs decay, LLM trigger
paths, and animation feedback loop end-to-end on the emulator before going to hardware.

### Bring your own assets
The APK boots without these — features degrade gracefully when missing.

```
app/src/main/assets/models/
  ├── object_labeler.tflite            (MediaPipe EfficientDet-Lite0)
  ├── face_landmarker.task             (MediaPipe Face Landmarker)
  ├── gesture_recognizer.task          (MediaPipe Gesture Recognizer)
  └── embedding_model.tflite           (all-MiniLM-L6-v2 converted)

app/src/main/assets/
  └── hey_zip_android.ppn              (Porcupine custom keyword — console.picovoice.ai)

app/src/main/assets/lottie/           (Lottie JSONs — file names in ZipAnimation enum)
app/src/main/res/raw/                  (sound effects — names in ZipSound enum)
```

### API keys
Drop into `local.properties` when ready. The cloud LLM falls through to the on-device stub
and finally to a scripted phrase pool when keys are blank — speech bubble still works.

```
GEMINI_API_KEY=...
OPENAI_API_KEY=...           # optional fallback
PORCUPINE_ACCESS_KEY=...     # wake word — only when bundling hey_zip_android.ppn
```

## Architecture

Strict Clean Architecture, MVVM, Compose UI, Hilt DI. Three layers + four cross-cutting roots:

```
ui/             ← Composables + ViewModels (depends on domain)
domain/         ← Pure Kotlin: use cases, FSM, needs, repository interfaces, Result
data/           ← Room, DataStore, Retrofit, repository implementations
perception/     ← CameraX + MediaPipe analyzers + event bus
ai/             ← LLM routing (Gemini Nano / Flash / scripted) + prompt builder
voice/          ← Wake word + STT + emotional TTS
worker/         ← WorkManager periodic jobs (needs decay, summarization, evolution)
notification/   ← Proactive nudges with daily cap + sleep-window respect
```

Full package layout: [`../looi-complete-spec.md`](../looi-complete-spec.md) §02.

## Privacy non-negotiables

- **No camera frames leave the device.** Object/face/gesture analysis is 100% on-device.
- **No analytics by default.** Opt-in only.
- **Privacy mode** disables all memory and personality evolution.
- **Local-first preferences** via DataStore; nothing synced.
- `data_extraction_rules.xml` + `backup_rules.xml` disable auto-backup so memories don't
  silently leave the device.

## Testing

```powershell
.\gradlew.bat testDebugUnitTest
```

Coverage (key domain logic):
- `NeedsEngineTest` — decay, event spikes, mood EWMA, user events (9 tests)
- `PersonalityFSMTest` — state transitions, sleep override, timeouts (7 tests)
- `ActionSelectorTest` — direct event response, needs-driven, DoNothing baseline (5 tests)
- `AnalyzerCadenceTest` — perception throttling (4 tests)
- `KnownObjectCacheTest` — novelty detection (4 tests)

## Performance guardrails (§23)

- MediaPipe tasks lazy-init on first frame, GPU→CPU fallback in `tryBuild`.
- Cadence-gated frame routing (face 100ms / gesture 200ms / object 300ms staggered).
- `DROP_OLDEST` overflow on perception bus — freshness over completeness.
- TTS uses `QUEUE_FLUSH` (single voice).
- Lottie character falls back to a soft gradient blob when assets missing — boots fast.
- Behavior state persisted to DataStore; long-absence catch-up tick capped at 8h.

## Release (§24)

`assembleRelease` is wired but `signingConfig` is commented out — drop a keystore + add
`signingConfig` to `app/build.gradle.kts` before publishing. R8 + ProGuard are enabled in
release with rules for MediaPipe / Moshi / Room / Lottie / Porcupine / TFLite / Compose.
