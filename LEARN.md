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

## Bug log
*(append real bugs below as they occur, newest first, using the template)*
