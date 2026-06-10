"""
Twilio inbound call server — connects phone calls to the voice booking agent.

Pipecat Flows pipeline with GPT-4o:
  transport.input() → STT → LanguageDetector → GuardrailProcessor → ContextAggregator.user()
  → GPT-4o LLM → TTS → transport.output() → ContextAggregator.assistant()

Usage:
  # Terminal 1: Start Cloudflare tunnel
  cloudflared tunnel --url http://localhost:8765

  # Terminal 2: Start the server
  PYTHONPATH=. .venv/bin/python packages/voice-agent/server.py

  # Then: Set Twilio phone number webhook to https://<tunnel>/incoming-call
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import time as time_mod
from collections import defaultdict
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
from fastapi import FastAPI, Form, Header, Request, WebSocket
from fastapi.responses import Response

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("server")

TUNNEL_URL = os.environ.get("TUNNEL_URL", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

MAX_CALLS_PER_WINDOW = 10
RATE_WINDOW_SECONDS = 900  # 15 minutes


def _mask_phone(phone: str) -> str:
    from packages.voice_agent.utils import mask_phone
    return mask_phone(phone)


def _validate_twilio_signature(url: str, params: dict[str, str], signature: str, auth_token: str) -> bool:
    sorted_params = urlencode(sorted(params.items()))
    data = url + sorted_params
    expected = base64.b64encode(
        hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    ).decode()
    return hmac.compare_digest(expected, signature)


_call_timestamps: dict[str, list[float]] = defaultdict(list)
_last_eviction: float = 0.0


def _check_rate_limit(phone: str) -> bool:
    global _last_eviction
    now = time_mod.monotonic()
    window_start = now - RATE_WINDOW_SECONDS

    if now - _last_eviction > RATE_WINDOW_SECONDS:
        stale = [k for k, v in _call_timestamps.items() if not v or v[-1] < window_start]
        for k in stale:
            del _call_timestamps[k]
        _last_eviction = now

    timestamps = _call_timestamps[phone]
    _call_timestamps[phone] = [t for t in timestamps if t > window_start]
    if len(_call_timestamps[phone]) >= MAX_CALLS_PER_WINDOW:
        return False
    _call_timestamps[phone].append(now)
    return True


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    if not DEEPGRAM_API_KEY:
        logger.error("DEEPGRAM_API_KEY not set")
        sys.exit(1)
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.error("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must both be set")
        sys.exit(1)
    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        logger.error("Either ANTHROPIC_API_KEY or OPENAI_API_KEY must be set")
        sys.exit(1)
    if not TUNNEL_URL:
        logger.warning("TUNNEL_URL not set — /incoming-call will return error TwiML")
    if not SARVAM_API_KEY:
        logger.info("SARVAM_API_KEY not set — Sarvam STT/TTS will not be available")
    yield


app = FastAPI(title="Voice Booking Agent", lifespan=lifespan)

TWIML_ERROR = """\
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, we are experiencing technical difficulties. Please try again later.</Say>
  <Hangup/>
</Response>"""

TWIML_RATE_LIMITED = """\
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>You've called several times recently. Please try again in a few minutes.</Say>
  <Hangup/>
</Response>"""


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/incoming-call")
async def incoming_call(
    request: Request,
    From: str = Form("unknown"),
    CallSid: str = Form("unknown"),
    x_twilio_signature: str = Header("", alias="X-Twilio-Signature"),
):
    if not TUNNEL_URL:
        logger.error("TUNNEL_URL not set — cannot build TwiML response")
        return Response(content=TWIML_ERROR, media_type="application/xml")

    if TWILIO_AUTH_TOKEN and x_twilio_signature:
        form_data = dict(await request.form())
        url = str(request.url)
        if not _validate_twilio_signature(url, form_data, x_twilio_signature, TWILIO_AUTH_TOKEN):
            logger.warning("Invalid Twilio signature from %s", _mask_phone(From))
            return Response(content=TWIML_ERROR, media_type="application/xml", status_code=403)

    logger.info("Incoming call from %s (CallSid=%s)", _mask_phone(From), CallSid)

    if not _check_rate_limit(From):
        logger.warning("Rate limited caller %s", _mask_phone(From))
        return Response(content=TWIML_RATE_LIMITED, media_type="application/xml")

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
        logger.info("Call started: call_sid=%s, caller=%s", call_sid, _mask_phone(caller_phone))
        await _run_pipeline(websocket, stream_sid, call_sid, caller_phone)
    except Exception:
        logger.exception("Error in call %s", call_sid or "unknown")
    finally:
        logger.info("Call ended: %s", call_sid or "unknown")


async def _parse_twilio_start(websocket: WebSocket) -> tuple[str, str, str]:
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
    import asyncio

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.workers.runner import WorkerRunner
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.deepgram.tts import DeepgramTTSService
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )
    from pipecat_flows import FlowManager

    from packages.voice_agent.data.mock_adapter import MockDataAdapter
    from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker
    from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore
    from packages.voice_agent.flows.guardrails import GuardrailProcessor
    from packages.voice_agent.flows.nodes import create_callback_capture_node, create_greeting_node

    config = _load_tenant_config()
    data = MockDataAdapter()
    budget_tracker = InMemoryBudgetTracker()
    checkpoint_store = InMemoryCheckpointStore()
    call_start = time_mod.monotonic()

    caller_info = await asyncio.to_thread(
        data.resolve_or_create_caller, config.meta.tenant_id, caller_phone
    )

    initial_language = config.persona.fallback_language
    preferred_language = getattr(caller_info, "preferred_language", None)
    if preferred_language and preferred_language in config.persona.languages:
        initial_language = preferred_language

    stt_initial_language = initial_language
    if config.pipeline.stt_provider == "sarvam" and not preferred_language:
        for lang in config.persona.languages:
            if lang.startswith("hi-"):
                stt_initial_language = lang
                break

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

    if config.pipeline.stt_provider == "sarvam":
        from pipecat.services.sarvam.stt import SarvamSTTService
        stt = SarvamSTTService(
            api_key=SARVAM_API_KEY,
            settings=SarvamSTTService.Settings(language=stt_initial_language),
        )
    else:
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            settings=DeepgramSTTService.Settings(
                language=initial_language.split("-")[0],
                model="nova-2-phonecall",
                interim_results=True,
                endpointing=700,
                utterance_end_ms=2000,
                smart_format=True,
            ),
        )

    initial_voice = config.pipeline.tts_voices.get(initial_language, "aura-asteria-en")
    if config.pipeline.tts_provider == "sarvam":
        from pipecat.services.sarvam.tts import SarvamTTSService
        tts = SarvamTTSService(
            api_key=SARVAM_API_KEY,
            settings=SarvamTTSService.Settings(voice=initial_voice),
        )
    else:
        tts = DeepgramTTSService(
            api_key=DEEPGRAM_API_KEY,
            settings=DeepgramTTSService.Settings(voice=initial_voice),
        )

    if config.pipeline.llm_provider == "openai":
        llm = OpenAILLMService(
            api_key=OPENAI_API_KEY,
            settings=OpenAILLMService.Settings(
                model="gpt-4o",
                max_completion_tokens=200,
                temperature=0.4,
            ),
        )
    else:
        llm = AnthropicLLMService(
            api_key=ANTHROPIC_API_KEY,
            settings=AnthropicLLMService.Settings(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                temperature=0.4,
            ),
        )

    guardrails = GuardrailProcessor(
        max_turns=config.guardrails.max_turns,
        max_seconds=config.guardrails.max_call_seconds,
        call_start=call_start,
    )

    from packages.voice_agent.flows.language_detector import LanguageDetectorProcessor

    language_detector = LanguageDetectorProcessor(
        config=config,
        flow_state={},  # will be replaced with flow_manager.state after it's created
        caller_id=caller_info.id,
        data_adapter=data,
        preferred_language=getattr(caller_info, "preferred_language", None),
        stt_provider=config.pipeline.stt_provider,
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(confidence=0.7, start_secs=0.3, stop_secs=0.3),
            ),
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        language_detector,
        guardrails,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
    guardrails.set_worker(worker)
    language_detector.set_worker(worker)

    from packages.voice_agent.flows.nodes import make_request_callback_tool
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=[make_request_callback_tool()],
    )

    flow_manager.state.update({
        "config": config,
        "data_adapter": data,
        "budget_tracker": budget_tracker,
        "checkpoint_store": checkpoint_store,
        "caller_phone": caller_phone,
        "caller_id": caller_info.id,
        "tenant_id": config.meta.tenant_id,
        "call_id": call_sid,
        "call_start": call_start,
        "turn_count": 0,
        "language": initial_language,
    })

    language_detector.set_flow_state(flow_manager.state)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_ref, client):
        logger.info("Client connected — checking budget and initializing flow")
        within_budget = await budget_tracker.check_budget(
            config.meta.tenant_id, config.guardrails.monthly_budget_inr
        )
        if not within_budget:
            logger.warning("Tenant %s over budget — routing to callback", config.meta.tenant_id)
            flow_manager.state["fallback_reason"] = "budget_exceeded"
            initial_node = create_callback_capture_node(flow_manager)
        else:
            initial_node = create_greeting_node(flow_manager)
        await flow_manager.initialize(initial_node)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_ref, client):
        elapsed = time_mod.monotonic() - call_start
        logger.info("Client disconnected — call lasted %.1fs", elapsed)
        try:
            await budget_tracker.record_usage(config.meta.tenant_id, elapsed)
        except Exception:
            logger.exception("Error recording call usage")
        await worker.cancel()

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


def _load_tenant_config():
    from packages.voice_agent.config.models import (
        AuthConfig,
        BookingModel,
        BusinessHours,
        EdgeProfile,
        EscalationChain,
        EscalationConfig,
        GuardrailsConfig,
        IntegrationConfig,
        LanguagePolicy,
        MetaConfig,
        PersonaConfig,
        PipelineConfig,
        Resource,
        Service,
        TenantConfig,
    )
    return TenantConfig(
        meta=MetaConfig(
            tenant_id="t1", sector="salon", config_version="1.0.0", status="live",
        ),
        persona=PersonaConfig(
            business_name="Glamour Salon",
            greeting={
                "en-IN": "Welcome to Glamour Salon!",
                "hi-IN": "ग्लैमर सैलून में आपका स्वागत है!",
            },
            ai_disclosure={
                "en-IN": "I'm an AI assistant.",
                "hi-IN": "आप एक AI असिस्टेंट से बात कर रहे हैं।",
            },
            tone="warm",
            languages=["en-IN", "hi-IN"],
            fallback_language="en-IN",
            language_policy=LanguagePolicy(greeting="default", match_caller=True),
        ),
        integration=IntegrationConfig(
            calendar_provider="native_supabase",
            write_back=False,
            system_of_record="native",
            external_unreachable_policy="capture",
        ),
        auth=AuthConfig(
            levels=["soft"],
            action_policy={"new_booking": "none"},
            soft_match_fields=["caller_number"],
        ),
        edge_profile=EdgeProfile(
            caller_demographic="general",
            endpointing_ms=500,
            barge_in=True,
            asr_lexicon=[],
            noise_profile="quiet",
            dtmf_fallback=False,
        ),
        escalation=EscalationConfig(
            chain=[EscalationChain(type="callback_queue")],
            trigger_on=["repeated_failure"],
        ),
        booking_model=BookingModel(
            resources=[
                Resource(id="r1", name="Priya", type="stylist", capacity=1, tags=[]),
                Resource(id="r2", name="Rahul", type="stylist", capacity=1, tags=[]),
            ],
            services=[
                Service(
                    id="s1", name="Haircut", resource_type="stylist",
                    duration_min=30, booking_mode="exclusive", custom_fields=[],
                ),
                Service(
                    id="s2", name="Hair Color", resource_type="stylist",
                    duration_min=60, booking_mode="exclusive", custom_fields=[],
                ),
            ],
            business_hours=BusinessHours(
                mon=["09:00-18:00"], tue=["09:00-18:00"],
                wed=["09:00-18:00"], thu=["09:00-18:00"],
                fri=["09:00-18:00"], sat=["10:00-16:00"],
            ),
            booking_window_days=14,
            min_notice_min=30,
        ),
        guardrails=GuardrailsConfig(
            max_call_seconds=300, max_turns=20,
            monthly_budget_inr=5000.0, scope="booking_only",
            recording_consent=True, retention_days=90,
        ),
        pipeline=PipelineConfig(
            stt_provider="sarvam",
            tts_provider="sarvam",
            llm_provider="openai",
            tts_voices={"en-IN": "anushka", "hi-IN": "anushka"},
        ),
    )


if __name__ == "__main__":
    logger.info("")
    logger.info("=" * 55)
    logger.info("  VOICE BOOKING AGENT — Pipecat Flows + GPT-4o")
    logger.info("  Tunnel: %s", TUNNEL_URL or "(not set)")
    logger.info("  Webhook: https://%s/incoming-call", TUNNEL_URL)
    logger.info("=" * 55)
    logger.info("")

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
