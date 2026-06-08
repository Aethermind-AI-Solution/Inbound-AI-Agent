# LEARN.md — Bugs, Fixes & Decisions Log

Running log so a problem solved once is never re-debugged. **Check this before coding in an area; append after every bug fix or non-obvious decision.**

## Entry template

### [YYYY-MM-DD] <short title>  ·  area: <voice|data|nlu|fallback|auth|integration|config|infra>
Symptom:    what was observed
Root cause: why it happened
Fix:        what changed
Prevention: test added / rule / hook so it can't recur

---

## Pre-empted pitfalls

### [decision] Idempotency key origin · area: data
Symptom: risk of silent double-booking on `confirm_booking` retry.
Root cause: if the agent generates a fresh idempotency key per attempt, a retry books twice.
Fix: **`idempotency_key = hold_id`.** One hold → one confirm → one booking.
Prevention: `bookings.idempotency_key UNIQUE`; concurrency test in Step 2.

### [decision] Date resolution ownership · area: nlu
Symptom: risk of the LLM inventing a non-existent slot/date.
Root cause: letting the model resolve dates.
Fix: deterministic resolver decides dates against business_hours + window; LLM only proposes a phrase.
Prevention: resolver unit suite (Step 3).

### [decision] External systems in the turn loop · area: integration
Symptom: would blow the latency budget.
Root cause: slow/flaky external CRM/calendar APIs.
Fix: offer from Supabase mirror; verify at hold only; sync async.

---

## Latency baseline

### [2026-06-04] Voice spike — Step 0 baseline · area: voice
Stack: Pipecat 1.3.0 · Deepgram Nova-2 STT · Deepgram Aura TTS (aura-asteria-en) · SileroVAD · LocalAudioTransport (MacBook mic/speaker)

| Component | Latency |
|-----------|---------|
| Pipeline overhead (transcription→TTS push) | ~25ms |
| Deepgram TTS TTFB | 264–404ms (P50 ~300ms) |
| Deepgram STT (estimated from Deepgram docs) | ~300–500ms |
| **Total estimated turn latency** | **~600–900ms** |

Notes:
- Local mic test only (no Twilio telephony overhead yet).
- Audio feedback loop solved with: (a) 4s cooldown after each echo, (b) "you said" text filter.
- Pipecat v1.3.0: `PipelineTask` → deprecated, use `PipelineWorker`. `PipelineRunner` → use `WorkerRunner`. VAD import path: `pipecat.audio.vad.silero` (not `pipecat.vad.silero`).
- pyaudio on arm64 Mac: must build from source with `ARCHFLAGS="-arch arm64" pip install --no-cache-dir --no-binary :all: pyaudio`.

---

## Live call latency baseline (Twilio + Sarvam + Claude Opus)

### [2026-06-08] Twilio live call baseline · area: voice
Stack: Pipecat 1.3.0 · Sarvam STT · Sarvam TTS (bulbul:v2, voice: anushka) · Claude Opus 4 · SileroVAD · Twilio Media Streams · Cloudflare Tunnel

| Component | Latency |
|-----------|---------|
| Sarvam STT TTFB | 0.5–2.2s |
| Claude Opus LLM TTFB | 1.8–2.6s (avg ~2.0s) |
| Sarvam TTS TTFB | 0.5–1.1s (avg ~0.7s) |
| **Total perceived turn latency** | **~3–5s** |

Notes:
- 16 LLM calls in a ~2.5min Hindi booking call (greeting→service→datetime→slot→confirm→close).
- Opus is overkill for voice — switch to Claude Haiku 4.5 for production (10x cheaper, ~500ms TTFB).
- Anthropic prompt caching shows 0 cache hits — enable cache_control on the system prompt.
- Sarvam STT reconnects on every interruption (TTS disconnect/reconnect cycle visible in logs).

---

## Cost estimates

### [2026-06-08] Per-call cost model · area: infra

Based on a typical 2.5-minute Hindi booking call (16 LLM turns, ~25s caller audio, ~1900 TTS chars):

| Service | Unit price | Usage/call | Cost/call |
|---------|-----------|------------|-----------|
| **Twilio** phone number | $1.15/mo | — | ~$0.04/day |
| **Twilio** media streams | $0.004/min | 2.5 min | $0.01 |
| **Sarvam STT** | ₹0.60/min | 2.5 min | ₹1.50 ($0.018) |
| **Sarvam TTS** | ₹0.04/1K chars | 1.9K chars | ₹0.08 ($0.001) |
| **Claude Opus** (current) | $15/M input + $75/M output | ~16 turns, ~8K in + ~1K out tokens | ~$0.20 |
| **Claude Haiku 4.5** (recommended) | $0.80/M in + $4/M out | same | ~$0.01 |
| **Cloudflare Tunnel** | Free | — | $0 |

**Per-call total (with Opus): ~$0.23** / ~₹19
**Per-call total (with Haiku): ~$0.04** / ~₹3.4

**Monthly projections (500 calls/month):**
- With Opus: ~$116/month (~₹9,700)
- **With Haiku: ~$21/month (~₹1,750)** ← recommended

Switching to Haiku 4.5 is the single biggest cost lever. The LLM is 85% of per-call cost with Opus.

---

## Bug log
*(append real bugs below as they occur, newest first, using the template)*

### [2026-06-08] WorkerRunner.add_workers() must be awaited · area: infra
Symptom: Call connected (Twilio `start` event received, pipeline linked) but immediately stalled — no audio processed, call hung for ~40s then disconnected. Log showed `RuntimeWarning: coroutine 'WorkerRunner.add_workers' was never awaited`.
Root cause: `runner.add_workers(worker)` is a coroutine in Pipecat 1.3.0 but was called without `await`. The worker never actually started, so the pipeline never processed frames.
Fix: Changed `runner.add_workers(worker)` → `await runner.add_workers(worker)`.
Prevention: Always `await` `WorkerRunner.add_workers()`. The RuntimeWarning is the telltale — grep logs for "was never awaited" after any pipeline stall.

### [2026-06-08] TTS target_language_code stuck on en-IN after Hindi lock-in · area: voice
Symptom: After detecting Hindi and switching LLM to Hindi, Sarvam TTS `target_language_code` stayed `en-IN` in every reconnect config.
Root cause: `_lock_in()` sent `TTSUpdateSettingsFrame(delta=TTSSettings(voice=...))` but never set `language=` on the delta. Sarvam TTS has an alias `target_language_code → language`, so it kept the initial value.
Fix: Send both `language` and `voice` in the TTS delta: `TTSSettings(language=language, voice=tts_voice)`.
Prevention: Visible in TTS config logs — grep for `target_language_code` to verify it matches detected language.

### [2026-06-08] Sarvam `language_code` always returns configured language, not detected · area: voice
Symptom: Caller spoke Hindi ("Mujhe booking chahiye") but Sarvam returned `language_code='en-IN'`. Language detector locked in English.
Root cause: Sarvam STT's `language_code` in the response echoes the *configured* language, not the *detected* spoken language. When initialized with `language='en-IN'`, it transcribes Hindi as romanized Latin text and tags it `en-IN`.
Fix: Replaced `frame.language` detection with **romanized Hindi keyword matching** — a set of ~60 common Hindi words. If 2+ keywords match in one utterance, detect Hindi. Falls back to English after `MAX_DETECTION_TURNS` (2) if no Hindi keywords found.
Prevention: Cannot rely on Sarvam's `language_code` for language identification. Any STT-native detection strategy must be validated with live calls, not just unit tests.

### [2026-06-08] lingua ConfidenceValue unpacking error · area: voice
Symptom: `cannot unpack non-iterable lingua.ConfidenceValue object` at `language_detector.py:136` during live call.
Root cause: Code used `for lang, conf in confidence_values:` (tuple unpacking) but lingua returns `ConfidenceValue` objects with `.language` and `.value` attributes.
Fix: Changed to `for cv in confidence_values:` using `cv.language` and `cv.value`.
Prevention: lingua API not covered by mocked unit tests. Need integration test that exercises the actual lingua detector (or at minimum, document the attribute API).

### [2026-06-08] LLM uses wrong year for dates (2024 instead of 2026) · area: nlu
Symptom: Caller said "14th June" and LLM sent `date: '2024-06-14'` to `check_availability`, which rejected it as "in the past."
Root cause: System prompt had no current date. LLM defaulted to a year from its training data.
Fix: Added `Today's date: {date}` to `build_role_message()` (every node's system prompt) and to `collect_datetime_task()`. Also added "Use the current year when the caller says a date without a year."
Prevention: System prompt now dynamically includes today's date via `_today_ist()`.

### [2026-06-08] Romanized Hindi defeats lingua text-based detection · area: voice
Symptom: Sarvam transcribed Hindi speech as romanized Latin text ("Mujhe booking chahiye"). lingua saw Latin characters and detected English with high confidence.
Root cause: lingua works on script/character patterns. Romanized Hindi is indistinguishable from English at the character level.
Fix: For Sarvam STT, skipped lingua entirely and used keyword-based detection on the romanized text instead. lingua is still used as fallback for Deepgram STT (which returns text in native scripts).
Prevention: Any text-based language detection on romanized transliterations will fail. Must use keyword matching or audio-level detection for romanized STT providers.

### [2026-06-08] Pipecat deprecation warnings (PipelineTask, PipelineRunner, STTUpdateSettingsFrame) · area: infra
Symptom: Three deprecation warnings in server logs every call.
Root cause: Pipecat 1.3.0 renamed `PipelineTask` → `PipelineWorker`, `PipelineRunner` → `WorkerRunner`, and `STTUpdateSettingsFrame(settings={...})` → `STTUpdateSettingsFrame(delta=STTSettings(...))`.
Fix: Updated imports and usage in `server.py` and `language_detector.py`.
Prevention: Already noted in latency baseline above; now actually fixed.
