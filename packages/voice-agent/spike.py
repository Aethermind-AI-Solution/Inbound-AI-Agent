"""
Voice-loop spike — measure STT + TTS latency using local mic/speaker.

No phone line needed. Speak into your mic, the agent echoes back what you said.
Measures P50/P95 turn latency. Press Ctrl+C to stop and see the report.

Prerequisites:
  brew install portaudio  (already done)
  pip install "pipecat-ai[local,deepgram,google]" pyaudio google-cloud-texttospeech

Usage:
  export DEEPGRAM_API_KEY=your_key
  python packages/voice-agent/spike.py

Or with a .env file in the project root:
  python packages/voice-agent/spike.py
"""

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("spike")


@dataclass
class LatencyTracker:
    """Track per-turn latency: transcription-received → TTS-first-byte."""
    turn_latencies: list[float] = field(default_factory=list)
    _pending_start: float | None = None

    def mark_transcription(self) -> None:
        self._pending_start = time.monotonic()

    def mark_tts_started(self) -> None:
        if self._pending_start is not None:
            latency = time.monotonic() - self._pending_start
            self.turn_latencies.append(latency)
            logger.info(f"Turn latency (transcription→TTS): {latency:.3f}s")
            self._pending_start = None

    def report(self) -> None:
        if not self.turn_latencies:
            logger.warning("No latency data collected. Did you speak into the mic?")
            return
        sorted_lat = sorted(self.turn_latencies)
        n = len(sorted_lat)
        p50_idx = n // 2
        p95_idx = min(int(n * 0.95), n - 1)
        avg = sum(sorted_lat) / n
        logger.info("=" * 50)
        logger.info(f"LATENCY REPORT — {n} turns")
        logger.info(f"  Min:  {sorted_lat[0]:.3f}s")
        logger.info(f"  P50:  {sorted_lat[p50_idx]:.3f}s")
        logger.info(f"  P95:  {sorted_lat[p95_idx]:.3f}s")
        logger.info(f"  Max:  {sorted_lat[-1]:.3f}s")
        logger.info(f"  Avg:  {avg:.3f}s")
        logger.info("=" * 50)


tracker = LatencyTracker()


async def run_spike() -> None:
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_key:
        logger.error("DEEPGRAM_API_KEY not set. Get one at deepgram.com (free $200 credit)")
        sys.exit(1)

    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.frames.frames import (
            Frame,
            TranscriptionFrame,
            TTSStartedFrame,
            TextFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    except ImportError as e:
        logger.error(
            f"Missing dependency: {e}\n"
            f"Install with: pip install 'pipecat-ai[local,deepgram,silero]' pyaudio"
        )
        sys.exit(1)

    # Try to import a TTS service — try multiple options
    tts_service = None

    # Option 1: Google Cloud TTS (free tier, easiest)
    try:
        from pipecat.services.google.tts import GoogleTTSService
        tts_service = GoogleTTSService(voice_id="en-IN-Standard-D", language="en-IN")
        logger.info("Using Google Cloud TTS (en-IN)")
    except (ImportError, Exception) as e:
        logger.debug(f"Google TTS not available: {e}")

    # Option 2: Deepgram TTS (uses same key)
    if tts_service is None:
        try:
            from pipecat.services.deepgram.tts import DeepgramTTSService
            tts_service = DeepgramTTSService(api_key=deepgram_key, voice="aura-asteria-en")
            logger.info("Using Deepgram TTS (aura-asteria-en)")
        except (ImportError, Exception) as e:
            logger.debug(f"Deepgram TTS not available: {e}")

    if tts_service is None:
        logger.error(
            "No TTS service available.\n"
            "Install one of:\n"
            "  pip install 'pipecat-ai[deepgram]'    (uses your Deepgram key for TTS too)\n"
            "  pip install 'pipecat-ai[google]' google-cloud-texttospeech\n"
        )
        sys.exit(1)

    class EchoProcessor(FrameProcessor):
        """Echoes transcription back as TTS. Sits BEFORE the TTS service."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._suppress_until: float = 0.0

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame):
                text = frame.text.strip() if frame.text else ""
                if not text:
                    await self.push_frame(frame, direction)
                    return
                now = time.monotonic()
                if now < self._suppress_until or "you said" in text.lower():
                    logger.debug(f"Suppressed echo: {text}")
                    return
                logger.info(f"You said: {text}")
                tracker.mark_transcription()
                echo = f"You said: {text}"
                self._suppress_until = now + 4.0
                await self.push_frame(TextFrame(text=echo), FrameDirection.DOWNSTREAM)
            else:
                await self.push_frame(frame, direction)

    class TTSLatencyCatcher(FrameProcessor):
        """Catches TTSStartedFrame AFTER the TTS service to record latency."""

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TTSStartedFrame):
                tracker.mark_tts_started()
            await self.push_frame(frame, direction)

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

    echo_processor = EchoProcessor()
    latency_catcher = TTSLatencyCatcher()

    pipeline = Pipeline([
        transport.input(),
        stt,
        echo_processor,
        tts_service,
        latency_catcher,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
    )

    runner = PipelineRunner()

    logger.info("")
    logger.info("=" * 50)
    logger.info("VOICE SPIKE — speak into your microphone")
    logger.info("The agent will echo back what you say.")
    logger.info("Press Ctrl+C to stop and see latency report.")
    logger.info("=" * 50)
    logger.info("")

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.report()


if __name__ == "__main__":
    try:
        asyncio.run(run_spike())
    except KeyboardInterrupt:
        tracker.report()
