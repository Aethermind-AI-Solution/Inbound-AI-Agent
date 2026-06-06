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

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, Form, WebSocket
from fastapi.responses import Response

if TYPE_CHECKING:
    from packages.voice_agent.config.models import TenantConfig
    from packages.voice_agent.dialogue.nlu.base import NLUService

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

TUNNEL_URL = os.environ.get("TUNNEL_URL", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    if not DEEPGRAM_API_KEY:
        logger.error("DEEPGRAM_API_KEY not set")
        sys.exit(1)
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.error("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must both be set")
        sys.exit(1)
    if not TUNNEL_URL:
        logger.warning("TUNNEL_URL not set — /incoming-call will return error TwiML")
    yield


app = FastAPI(title="Voice Booking Agent", lifespan=lifespan)

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
      <Parameter name="caller_phone" value="{escape(From, quote=True)}"/>
    </Stream>
  </Connect>
  <Pause length="40"/>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


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
        encoding="mulaw",
        sample_rate=8000,
        settings=DeepgramSTTService.Settings(
            language="en",
            model="nova-2-phonecall",
            interim_results=True,
            endpointing=300,
            utterance_end_ms=1000,
            smart_format=True,
        ),
    )

    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY,
        encoding="mulaw",
        sample_rate=8000,
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


def _create_nlu(config: TenantConfig) -> NLUService:
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


if __name__ == "__main__":
    logger.info("")
    logger.info("=" * 55)
    logger.info("  VOICE BOOKING AGENT — Twilio server")
    logger.info(f"  Tunnel: {TUNNEL_URL or '(not set)'}")
    logger.info(f"  Webhook: https://{TUNNEL_URL}/incoming-call")
    logger.info("=" * 55)
    logger.info("")

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
