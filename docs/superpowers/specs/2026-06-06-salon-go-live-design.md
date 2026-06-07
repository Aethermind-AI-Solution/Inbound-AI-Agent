# Salon Go-Live Design Spec

**Date:** 2026-06-06
**Status:** Approved

## Goal

Build a custom FastAPI server (`server.py`) that receives inbound Twilio calls, runs the full dialogue state machine via a pipecat pipeline, and lets a caller walk through a booking flow over a real phone call. MVP scope: MockDataAdapter, English only, single concurrent call, local dev with Cloudflare tunnel.

## Architecture

One new file: `packages/voice-agent/server.py` — a standalone FastAPI application with three routes:

1. **`POST /incoming-call`** — Twilio webhook. Receives call metadata (From, CallSid) as form data. Returns TwiML XML instructing Twilio to open a bidirectional WebSocket media stream to `/ws`, passing the caller's phone number as a custom parameter.

2. **`WebSocket /ws`** — Receives Twilio's media stream. Parses the `start` message to extract `streamSid`, `callSid`, and `caller_phone`. Creates per-call instances of all pipeline components. Runs the pipecat pipeline for the call's lifetime.

3. **`GET /health`** — Returns `{"status": "ok"}`.

### Call Flow

```
Caller dials US number
  → Twilio sends POST /incoming-call
  → Server returns TwiML with <Stream url="wss://{TUNNEL_URL}/ws">
  → Twilio opens WebSocket to /ws
  → Server parses start message, creates pipeline:
      transport.input() → Deepgram STT → PipelineAdapter → Deepgram TTS → transport.output()
  → PipelineAdapter calls dm.start() → greeting plays
  → Caller speaks → STT transcribes → PipelineAdapter → dm.handle_event() → TTS responds
  → On END_CALL → TwilioFrameSerializer auto-hangs-up via Twilio REST API
  → WebSocket closes, pipeline cleans up
```

Each call gets its own pipeline instance. The server is stateless; all call state lives in the DialogueManager's in-memory context.

## Components

### TwiML Response

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{TUNNEL_URL}/ws">
      <Parameter name="caller_phone" value="{From}"/>
    </Stream>
  </Connect>
  <Pause length="40"/>
</Response>
```

The `<Pause length="40"/>` keeps the call alive while the WebSocket stream is active. Twilio ends the call when the stream disconnects or `<Pause>` expires.

The caller's phone number from the webhook `From` field is forwarded as a custom parameter so the WebSocket handler can use it for ANI soft-auth.

### WebSocket Handler

On WebSocket connect:

1. Accept the connection
2. Wait for Twilio's `start` event message (JSON)
3. Extract from `start` message:
   - `streamSid` — identifies the media stream
   - `callSid` — identifies the call (used as `call_id` for DialogueManager)
   - `customParameters.caller_phone` — the caller's phone number
4. Create `TwilioFrameSerializer` with `stream_sid`, `call_sid`, `account_sid`, `auth_token`, `auto_hang_up=True`
5. Create `FastAPIWebsocketTransport` with the serializer
6. Create `DeepgramSTTService` and `DeepgramTTSService`
7. Create `DialogueManager` with `MockDataAdapter`, `ClaudeNLUService`, `InMemoryCheckpointStore`, `InMemoryBudgetTracker`
8. Create `PipelineAdapter` wrapping the DialogueManager
9. Build pipeline: `transport.input() → STT → PipelineAdapter → TTS → transport.output()`
10. Create `PipelineTask` and `PipelineRunner`, run until call ends

### STT — Deepgram Nova 2

| Setting | Value |
|---|---|
| Model | `nova-2` |
| Language | `en` |
| `interim_results` | `true` |
| `endpointing` | `300` ms |
| `utterance_end_ms` | `1000` ms |
| `smart_format` | `true` |

Same settings as `run_local.py`.

### TTS — Deepgram Aura

| Setting | Value |
|---|---|
| Voice | `aura-asteria-en` |

Uses the same `DEEPGRAM_API_KEY` as STT.

### NLU — Claude Haiku

`ClaudeNLUService` with `AnthropicLLMClient`. Falls back to `StubNLU` if `ANTHROPIC_API_KEY` is not set.

### Transport — FastAPIWebsocketTransport

| Setting | Value |
|---|---|
| `audio_in_enabled` | `true` |
| `audio_out_enabled` | `true` |
| `add_wav_header` | `false` |
| Serializer | `TwilioFrameSerializer` |

No VAD needed — Twilio handles voice activity detection. Audio only flows when the caller speaks.

### Data & State

| Component | Implementation | Why |
|---|---|---|
| DataAdapter | `MockDataAdapter` | No Supabase needed for MVP |
| CheckpointStore | `InMemoryCheckpointStore` | Single instance, no resume needed |
| BudgetTracker | `InMemoryBudgetTracker` (Redis if `REDIS_URL` set) | Same pattern as `run_local.py` |
| TenantConfig | `make_tenant_config()` from test conftest | Reuse existing salon config fixture |

## Environment Variables

All existing in `.env` except `TUNNEL_URL`:

| Variable | Purpose | Required |
|---|---|---|
| `TWILIO_ACCOUNT_SID` | Auto-hang-up via REST API | Yes |
| `TWILIO_AUTH_TOKEN` | Auto-hang-up via REST API | Yes |
| `DEEPGRAM_API_KEY` | STT + TTS | Yes |
| `ANTHROPIC_API_KEY` | Claude Haiku NLU | Yes (falls back to StubNLU) |
| `REDIS_URL` | Redis budget tracker | No (falls back to in-memory) |
| `TUNNEL_URL` | Cloudflare tunnel hostname for TwiML WebSocket URL | Yes |

## Error Handling

**Pipeline creation failure** (missing API key, service unavailable): Log the error and return a TwiML error response:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, we are experiencing technical difficulties. Please try again later.</Say>
  <Hangup/>
</Response>
```

**WebSocket disconnect mid-call:** Pipecat's transport handles cleanup automatically via its frame lifecycle.

**DialogueManager error during a turn:** `PipelineAdapter` already catches exceptions, logs them, records usage, and sends `EndTaskFrame` to end the call gracefully (existing Step 5 behavior).

## How to Run

```bash
# Terminal 1: Start Cloudflare tunnel
cloudflared tunnel --url http://localhost:8765

# Terminal 2: Start the server
PYTHONPATH=. .venv/bin/python packages/voice-agent/server.py

# Then: In Twilio console, set the phone number's
# "A call comes in" webhook to: https://<tunnel-hostname>/incoming-call
```

Server binds to `0.0.0.0:8765`. Uvicorn with structured logging at INFO level.

## Success Criteria

1. Call your Twilio US number from any phone
2. Hear the greeting from the DialogueManager
3. Walk through a complete booking flow (service → datetime → slot → readback → confirm)
4. Call ends cleanly after confirmation
5. Fallback paths work (unknown service → callback capture)

## Not in Scope

- Supabase data adapter (real data) — future step
- Redis checkpoint (cross-instance resume) — production concern
- Production deployment (Docker, cloud hosting) — local dev only
- Call recording or Langfuse observability — can layer on later
- Hindi/bilingual support — English only for US number
- Concurrent call load testing — single call at a time is fine for MVP
- Custom tenant config loading from file/DB — uses test fixture config

## File Changes

- **Create:** `packages/voice-agent/server.py` (~120 lines)
- **No changes** to existing files
