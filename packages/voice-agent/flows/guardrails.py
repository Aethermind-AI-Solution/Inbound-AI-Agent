from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pipecat.frames.frames import EndFrame, TTSSpeakFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker

logger = logging.getLogger(__name__)


class GuardrailProcessor(FrameProcessor):
    """Deterministic pipeline-level enforcement of max_turns and max_call_seconds.

    Sits between STT and ContextAggregator. Counts transcription frames and
    checks elapsed time. When limits are exceeded, queues a TTS farewell
    followed by EndFrame via the pipeline worker. The LLM cannot bypass this.
    """

    def __init__(self, max_turns: int, max_seconds: float, call_start: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_turns = max_turns
        self._max_seconds = max_seconds
        self._call_start = call_start
        self._turn_count = 0
        self._ended = False
        self._worker: PipelineWorker | None = None

    def set_worker(self, worker: PipelineWorker) -> None:
        self._worker = worker

    @property
    def turn_count(self) -> int:
        return self._turn_count

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)

        if self._ended:
            if isinstance(frame, TranscriptionFrame):
                return
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            self._turn_count += 1
            elapsed = time.monotonic() - self._call_start

            if self._turn_count > self._max_turns:
                logger.warning("Guardrail: max turns (%d) exceeded", self._max_turns)
                await self._end_call(
                    "I want to make sure we get this right. "
                    "Let me have someone call you back to finish up."
                )
                return

            if elapsed > self._max_seconds:
                logger.warning("Guardrail: max duration (%.0fs) exceeded", self._max_seconds)
                await self._end_call(
                    "I'm sorry, we've been on the call for a while. "
                    "Let me have someone call you back."
                )
                return

        await self.push_frame(frame, direction)

    async def _end_call(self, message: str) -> None:
        self._ended = True
        if self._worker:
            await self._worker.queue_frame(TTSSpeakFrame(text=message))
            await self._worker.queue_frame(EndFrame())
        else:
            logger.error("Guardrail: no worker reference — cannot end call gracefully")
