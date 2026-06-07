# Salon Go-Live Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom FastAPI server that receives inbound Twilio calls, runs the full dialogue state machine via a pipecat pipeline, and lets a caller walk through a booking flow over a real phone call.

**Architecture:** One new file `packages/voice-agent/server.py` — a standalone FastAPI app with three routes: `POST /incoming-call` (returns TwiML), `WebSocket /ws` (runs pipecat pipeline per call), `GET /health`. Each incoming call gets its own pipeline instance with MockDataAdapter, Claude Haiku NLU, Deepgram STT+TTS, and the existing PipelineAdapter + DialogueManager.

**Tech Stack:** FastAPI, uvicorn, pipecat-ai (FastAPIWebsocketTransport, TwilioFrameSerializer, DeepgramSTTService, DeepgramTTSService), Twilio, Cloudflare tunnel

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `packages/voice-agent/server.py` | Create | FastAPI app with `/incoming-call`, `/ws`, `/health` routes. Per-call pipeline creation. Entry point with uvicorn. |
| `.env` | Modify | Add `TUNNEL_URL` variable |

No other files are modified. All existing components (`PipelineAdapter`, `DialogueManager`, `MockDataAdapter`, `ClaudeNLUService`, `InMemoryCheckpointStore`, `InMemoryBudgetTracker`, `make_tenant_config`) are used as-is.

---

### Task 1: FastAPI app skeleton with health and TwiML routes

**Files:**
- Create: `packages/voice-agent/server.py`

- [ ] **Step 1: Create the FastAPI app with env loading, health route, and TwiML webhook**

```python
"""
Twilio inbound call server — connects phone calls to the dialogue state machine.

Custom FastAPI server with three routes:
  POST /incoming-call — Twilio webhook, returns TwiML to open a media stream
  WebSocket /ws — receives Twilio media stream, runs pipecat pipeline
  GET /health — health check

Usage:
  # Terminal 1: Start Cloudflare tunnel
  cloudflared tunnel --url http://localhost:8765

  # Terminal 2: Start the server
  PYTHONPATH=. .venv/bin/python packages/voice-agent/server.py

  # Then: Set Twilio phone number webhook to https://<tunnel>/incoming-call
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, WebSocket
from fastapi.responses import Response

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("server")

app = FastAPI(title="Voice Booking Agent")

TUNNEL_URL = os.environ.get("TUNNEL_URL", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

TWIML_ERROR = """\
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, we are experiencing technical difficulties. Please try again later.</Say>
  <Hangup/>
</Response>"""


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/incoming-call")
async def incoming_call(From: str = Form("unknown"), CallSid: str = Form("unknown")):
    if not TUNNEL_URL:
        logger.error("TUNNEL_URL not set — cannot build TwiML response")
        return Response(content=TWIML_ERROR, media_type="application/xml")

    logger.info(f"Incoming call from {From} (CallSid={CallSid})")

    twiml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{TUNNEL_URL}/ws">
      <Parameter name="caller_phone" value="{From}"/>
    </Stream>
  </Connect>
  <Pause length="40"/>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


if __name__ == "__main__":
    if not DEEPGRAM_API_KEY:
        logger.error("DEEPGRAM_API_KEY not set")
        sys.exit(1)
    if not TUNNEL_URL:
        logger.warning("TUNNEL_URL not set — TwiML responses will fail")

    logger.info("")
    logger.info("=" * 55)
    logger.info("  VOICE BOOKING AGENT — Twilio server")
    logger.info(f"  Tunnel: {TUNNEL_URL or '(not set)'}")
    logger.info(f"  Webhook: https://{TUNNEL_URL}/incoming-call")
    logger.info("=" * 55)
    logger.info("")

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
```

- [ ] **Step 2: Verify the server starts and health check works**

Run:
```bash
PYTHONPATH=. TUNNEL_URL=test.example.com .venv/bin/python -c "
import asyncio
from httpx import AsyncClient, ASGITransport
from packages.voice_agent.server import app

async def test():
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as c:
        r = await c.get('/health')
        assert r.status_code == 200
        assert r.json() == {'status': 'ok'}
        print('Health OK')

        r = await c.post('/incoming-call', data={'From': '+1234567890', 'CallSid': 'CA123'})
        assert r.status_code == 200
        assert 'wss://test.example.com/ws' in r.text
        assert 'caller_phone' in r.text
        assert '+1234567890' in r.text
        print('TwiML OK')
        print(r.text)

asyncio.run(test())
"
```

Expected: Both assertions pass. TwiML contains the tunnel URL and caller phone parameter.

- [ ] **Step 3: Commit**

```bash
git add packages/voice-agent/server.py
git commit -m "feat(server): add FastAPI skeleton with health check and TwiML webhook"
```

---

### Task 2: WebSocket handler with Twilio media stream parsing and pipeline

**Files:**
- Modify: `packages/voice-agent/server.py`

This is the core task — the WebSocket endpoint that parses Twilio's `start` message, creates all pipeline components, and runs the pipecat pipeline for the call's lifetime.

- [ ] **Step 1: Add the WebSocket route and pipeline creation logic**

Add these imports at the top of `server.py`, after the existing imports:

```python
import json
import uuid
```

Add the WebSocket route after the `/incoming-call` route:

```python
@app.websocket("/ws")
async def websocket_twilio(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    call_sid = None
    try:
        stream_sid, call_sid, caller_phone = await _parse_twilio_start(websocket)
        logger.info(f"Call started: call_sid={call_sid}, caller={caller_phone}")
        await _run_pipeline(websocket, stream_sid, call_sid, caller_phone)
    except Exception:
        logger.exception(f"Error in call {call_sid or 'unknown'}")
    finally:
        logger.info(f"Call ended: {call_sid or 'unknown'}")


async def _parse_twilio_start(websocket: WebSocket) -> tuple[str, str, str]:
    """Wait for Twilio's 'start' event and extract stream/call metadata."""
    async for raw in websocket.iter_text():
        msg = json.loads(raw)
        if msg.get("event") == "connected":
            logger.info("Twilio connected event received")
            continue
        if msg.get("event") == "start":
            start = msg["start"]
            stream_sid = start["streamSid"]
            call_sid = start["callSid"]
            custom = start.get("customParameters", {})
            caller_phone = custom.get("caller_phone", "unknown")
            return stream_sid, call_sid, caller_phone
    raise ValueError("WebSocket closed before receiving Twilio start event")


async def _run_pipeline(
    websocket: WebSocket,
    stream_sid: str,
    call_sid: str,
    caller_phone: str,
) -> None:
    """Create and run the pipecat pipeline for a single call."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.deepgram.tts import DeepgramTTSService
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    from packages.voice_agent.data.mock_adapter import MockDataAdapter
    from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker
    from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore
    from packages.voice_agent.dialogue.manager import DialogueManager
    from packages.voice_agent.pipeline_adapter import PipelineAdapter
    from packages.voice_agent.tests.test_states.conftest import make_tenant_config

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        settings=DeepgramSTTService.Settings(
            language="en",
            model="nova-2",
            interim_results=True,
            endpointing=300,
            utterance_end_ms=1000,
            smart_format=True,
        ),
    )

    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY,
        settings=DeepgramTTSService.Settings(voice="aura-asteria-en"),
    )

    config = make_tenant_config()
    data = MockDataAdapter()
    checkpoint = InMemoryCheckpointStore()
    budget_tracker = InMemoryBudgetTracker()

    nlu = _create_nlu(config)

    dm = DialogueManager(
        config=config,
        data_adapter=data,
        nlu=nlu,
        checkpoint_store=checkpoint,
        caller_phone=caller_phone,
        call_id=call_sid,
        budget_tracker=budget_tracker,
    )
    adapter = PipelineAdapter(dm)

    pipeline = Pipeline([
        transport.input(),
        stt,
        adapter,
        tts,
        transport.output(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    runner = PipelineRunner()
    await runner.run(task)
```

Add the NLU helper function after `_run_pipeline`:

```python
def _create_nlu(config):
    """Create NLU service — Claude Haiku if API key is set, otherwise StubNLU."""
    if ANTHROPIC_API_KEY:
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        from packages.voice_agent.dialogue.nlu.llm_client import AnthropicLLMClient

        llm_client = AnthropicLLMClient(api_key=ANTHROPIC_API_KEY)
        return ClaudeNLUService(llm_client, config)
    else:
        from packages.voice_agent.dialogue.nlu.stub import StubNLUService

        logger.warning("ANTHROPIC_API_KEY not set — using StubNLU")
        return StubNLUService()
```

- [ ] **Step 2: Verify the server module loads without errors**

Run:
```bash
PYTHONPATH=. TUNNEL_URL=test.example.com .venv/bin/python -c "
from packages.voice_agent.server import app
print('Module loaded OK')
print('Routes:', [r.path for r in app.routes])
"
```

Expected: Module loads, routes include `/health`, `/incoming-call`, `/ws`.

- [ ] **Step 3: Commit**

```bash
git add packages/voice-agent/server.py
git commit -m "feat(server): add WebSocket handler with Twilio parsing and pipecat pipeline"
```

---

### Task 3: Add TUNNEL_URL to .env and update startup validation

**Files:**
- Modify: `.env`

- [ ] **Step 1: Add TUNNEL_URL placeholder to .env**

Add this line at the end of `.env`:

```
TUNNEL_URL=
```

The value will be filled in when starting the Cloudflare tunnel (e.g., `abc-xyz.trycloudflare.com`).

- [ ] **Step 2: Commit**

```bash
git add .env
git commit -m "chore: add TUNNEL_URL placeholder to .env"
```

Wait — `.env` contains secrets. Do NOT commit it. Check if `.gitignore` includes `.env`:

```bash
grep "\.env" .gitignore
```

If `.env` is in `.gitignore` (expected), skip this commit — the user manages `.env` locally. Just add the line manually.

---

### Task 4: End-to-end smoke test (manual)

This task is manual — it requires a real Twilio number, Cloudflare tunnel, and a phone to call from.

- [ ] **Step 1: Start the Cloudflare tunnel**

```bash
cloudflared tunnel --url http://localhost:8765
```

Note the tunnel URL printed (e.g., `https://abc-xyz.trycloudflare.com`). Copy the hostname (without `https://`).

- [ ] **Step 2: Set TUNNEL_URL in .env**

Edit `.env` and set:
```
TUNNEL_URL=abc-xyz.trycloudflare.com
```

(Replace with your actual tunnel hostname from Step 1.)

- [ ] **Step 3: Start the server**

```bash
PYTHONPATH=. .venv/bin/python packages/voice-agent/server.py
```

Expected output:
```
  VOICE BOOKING AGENT — Twilio server
  Tunnel: abc-xyz.trycloudflare.com
  Webhook: https://abc-xyz.trycloudflare.com/incoming-call
```

- [ ] **Step 4: Configure Twilio webhook**

In the Twilio console:
1. Go to Phone Numbers → Active Numbers → click your US number
2. Under "Voice Configuration" → "A call comes in":
   - Set to: **Webhook**
   - URL: `https://abc-xyz.trycloudflare.com/incoming-call`
   - HTTP: **HTTP POST**
3. Click "Save configuration"

- [ ] **Step 5: Call your Twilio number**

Call the number from your phone. You should:
1. Hear a brief pause while the WebSocket connects
2. Hear the greeting: "Welcome to Glamour Studio! I'm your AI assistant..."
3. Say "I want to book a haircut" → agent asks for date/time
4. Say "tomorrow at 10am" → agent offers available slots
5. Pick a stylist → agent reads back the booking
6. Say "yes" → agent confirms and says goodbye
7. Call ends automatically

- [ ] **Step 6: Check server logs**

Verify in the terminal:
- `Incoming call from +1XXXXXXXXXX (CallSid=CA...)`
- `WebSocket connection accepted`
- `Call started: call_sid=CA..., caller=+1XXXXXXXXXX`
- `Call ended: CA...`

- [ ] **Step 7: Test fallback path**

Call again and say gibberish (e.g., "asdkjfhaskdjfh") three times. The agent should:
1. Ask you to repeat
2. Eventually offer a callback
3. Capture your callback info
4. End the call

- [ ] **Step 8: Commit any fixes**

If any code changes were needed during testing:
```bash
git add packages/voice-agent/server.py
git commit -m "fix(server): adjustments from smoke test"
```

---

### Task 5: Update BUILD.md

**Files:**
- Modify: `BUILD.md`

- [ ] **Step 1: Mark Step 8 as done**

Change `## Step 8 — Salon go-live ☐` to `## Step 8 — Salon go-live ☑`

- [ ] **Step 2: Commit**

```bash
git add BUILD.md
git commit -m "docs: mark Step 8 (salon go-live) complete in BUILD.md"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ `POST /incoming-call` with TwiML + caller_phone parameter → Task 1
- ✅ `WebSocket /ws` with Twilio start parsing → Task 2
- ✅ `GET /health` → Task 1
- ✅ Pipeline: transport.input() → STT → PipelineAdapter → TTS → transport.output() → Task 2
- ✅ STT settings (nova-2, en, interim, endpointing 300, utterance_end_ms 1000, smart_format) → Task 2
- ✅ TTS settings (aura-asteria-en) → Task 2
- ✅ TwilioFrameSerializer with auto_hang_up → Task 2
- ✅ FastAPIWebsocketParams (audio_in, audio_out, no wav header) → Task 2
- ✅ MockDataAdapter, InMemoryCheckpointStore, InMemoryBudgetTracker → Task 2
- ✅ ClaudeNLUService with StubNLU fallback → Task 2
- ✅ make_tenant_config() → Task 2
- ✅ TUNNEL_URL env var → Task 1, Task 3
- ✅ Error TwiML response → Task 1
- ✅ Server on 0.0.0.0:8765 with uvicorn → Task 1
- ✅ How to run (cloudflared + server + Twilio config) → Task 4
- ✅ Success criteria (5 items) → Task 4
- ✅ BUILD.md update → Task 5

**2. Placeholder scan:** No TBD, TODO, or vague steps found.

**3. Type consistency:**
- `_parse_twilio_start` returns `tuple[str, str, str]` → used as `stream_sid, call_sid, caller_phone` in Task 2 ✅
- `_run_pipeline` signature matches call site in `websocket_twilio` ✅
- `_create_nlu` takes `config` (TenantConfig) and returns NLU service ✅
- `TwilioFrameSerializer` constructor args match pipecat source (`stream_sid`, `call_sid`, `account_sid`, `auth_token`) ✅
- `FastAPIWebsocketParams` fields match pipecat source (`audio_in_enabled`, `audio_out_enabled`, `add_wav_header`, `serializer`) ✅
- `DeepgramSTTService.Settings` fields match pipecat source (`language`, `model`, `interim_results`, `endpointing`, `utterance_end_ms`, `smart_format`) ✅
- `DeepgramTTSService.Settings` field matches pipecat source (`voice`) ✅
