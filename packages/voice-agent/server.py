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
