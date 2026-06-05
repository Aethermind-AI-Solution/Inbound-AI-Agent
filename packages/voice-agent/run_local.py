"""
Local voice agent — speak into your mic, talk to the dialogue state machine.

Uses MockDataAdapter (no Supabase needed) and StubNLU (keyword matching).
Requires DEEPGRAM_API_KEY in environment or .env file.

Usage:
  export DEEPGRAM_API_KEY=your_key
  python packages/voice-agent/run_local.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_local")


async def main() -> None:
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_key:
        logger.error("DEEPGRAM_API_KEY not set")
        sys.exit(1)

    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.audio.vad.silero import SileroVADAnalyzer
    except ImportError as e:
        logger.error(
            f"Missing dependency: {e}\n"
            f"Install with: pip install 'pipecat-ai[local,deepgram,silero]' pyaudio"
        )
        sys.exit(1)

    tts_service = None
    try:
        from pipecat.services.google.tts import GoogleTTSService
        tts_service = GoogleTTSService(voice_id="en-IN-Standard-D", language="en-IN")
        logger.info("Using Google Cloud TTS")
    except (ImportError, Exception):
        pass

    if tts_service is None:
        try:
            from pipecat.services.deepgram.tts import DeepgramTTSService
            tts_service = DeepgramTTSService(
                api_key=deepgram_key, voice="aura-asteria-en"
            )
            logger.info("Using Deepgram TTS")
        except (ImportError, Exception):
            pass

    if tts_service is None:
        logger.error("No TTS service available")
        sys.exit(1)

    from packages.voice_agent.data.mock_adapter import MockDataAdapter
    from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore
    from packages.voice_agent.dialogue.manager import DialogueManager
    from packages.voice_agent.pipeline_adapter import PipelineAdapter
    from packages.voice_agent.tests.test_states.conftest import make_tenant_config

    config = make_tenant_config()
    data = MockDataAdapter()
    checkpoint = InMemoryCheckpointStore()

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        from packages.voice_agent.dialogue.nlu.llm_client import AnthropicLLMClient
        llm_client = AnthropicLLMClient(api_key=anthropic_key)
        nlu = ClaudeNLUService(llm_client, config)
        logger.info("Using Claude Haiku NLU")
    else:
        from packages.voice_agent.dialogue.nlu.stub import StubNLUService
        nlu = StubNLUService()
        logger.info("Using StubNLU (set ANTHROPIC_API_KEY for Claude NLU)")

    dm = DialogueManager(
        config=config,
        data_adapter=data,
        nlu=nlu,
        checkpoint_store=checkpoint,
        caller_phone="+919876543210",
        call_id="local-test-001",
    )
    adapter = PipelineAdapter(dm)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    stt = DeepgramSTTService(
        api_key=deepgram_key,
        language="en",
        model="nova-2",
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        adapter,
        tts_service,
        transport.output(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    runner = PipelineRunner()

    logger.info("")
    logger.info("=" * 50)
    logger.info("VOICE AGENT — speak into your microphone")
    logger.info("Walk through a booking flow with the state machine.")
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 50)
    logger.info("")

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
