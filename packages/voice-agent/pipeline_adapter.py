from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndTaskFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallEvent,
    EventType,
)

if TYPE_CHECKING:
    from pipecat.frames.frames import Frame

    from packages.voice_agent.dialogue.manager import DialogueManager

logger = logging.getLogger(__name__)

BARGE_IN_COOLDOWN = 1.5
ECHO_GUARD_SECS = 1.0
TRANSCRIPTION_DEBOUNCE = 2.0


class PipelineAdapter(FrameProcessor):
    def __init__(self, dialogue_manager: DialogueManager, **kwargs) -> None:
        super().__init__(**kwargs)
        self._dm = dialogue_manager
        self._started = False
        self._silence_task: asyncio.Task | None = None
        self._pending_silence_timeout: float | None = None
        self._bot_speaking = False
        self._bot_speak_start: float = 0
        self._last_interruption_time: float = 0
        self._last_transcription_time: float = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not self._started:
            self._started = True
            try:
                action = await self._dm.start()
                await self._push_action(action)
            except Exception:
                logger.exception("Error in DialogueManager.start()")
                await self._record_usage()
                await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)
                return

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._bot_speak_start = time.monotonic()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._pending_silence_timeout is not None:
                self._reset_silence_timer(self._pending_silence_timeout)
                self._pending_silence_timeout = None
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            if self._bot_speaking:
                since_bot_started = time.monotonic() - self._bot_speak_start
                if since_bot_started < ECHO_GUARD_SECS:
                    logger.debug(
                        "Ignoring VAD during echo guard (%.1fs after bot started)",
                        since_bot_started,
                    )
                else:
                    logger.info("Barge-in detected — interrupting bot speech")
                    self._last_interruption_time = time.monotonic()
                    await self.broadcast_interruption()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            if self._should_drop_transcription(frame):
                return
            await self._handle_transcription(frame)
            return

        await self.push_frame(frame, direction)

    def _should_drop_transcription(self, frame: TranscriptionFrame) -> bool:
        now = time.monotonic()
        if self._bot_speaking:
            logger.debug("Dropping transcription while bot is speaking: %r", frame.text)
            return True
        since_interruption = now - self._last_interruption_time
        if since_interruption < BARGE_IN_COOLDOWN:
            logger.debug(
                "Dropping stale transcription %.1fs after barge-in: %r",
                since_interruption, frame.text,
            )
            return True
        since_last = now - self._last_transcription_time
        if since_last < TRANSCRIPTION_DEBOUNCE:
            logger.debug(
                "Dropping rapid transcription %.1fs after last: %r",
                since_last, frame.text,
            )
            return True
        return False

    async def _handle_transcription(self, frame: TranscriptionFrame) -> None:
        self._cancel_silence_timer()
        self._last_transcription_time = time.monotonic()
        logger.info("User said: %r", frame.text)
        event = CallEvent(type=EventType.TRANSCRIPTION, text=frame.text)
        try:
            action = await self._dm.handle_event(event)
            await self._push_action(action)
        except Exception:
            logger.exception("Error in DialogueManager.handle_event()")
            await self._record_usage()
            await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)

    async def _record_usage(self) -> None:
        try:
            elapsed = time.monotonic() - self._dm.context.call_start
            await self._dm.record_call_usage(elapsed)
        except Exception:
            logger.exception("Error recording call usage")

    async def _push_action(self, action: Action) -> None:
        if action.type in (ActionType.ASK, ActionType.SPEAK):
            if action.text:
                logger.info("Bot says: %r", action.text)
                await self.push_frame(
                    TTSSpeakFrame(text=action.text), FrameDirection.DOWNSTREAM
                )
            self._pending_silence_timeout = action.timeout_s
        elif action.type == ActionType.END_CALL:
            self._cancel_silence_timer()
            self._pending_silence_timeout = None
            if action.text:
                logger.info("Bot says (closing): %r", action.text)
                await self.push_frame(
                    TTSSpeakFrame(text=action.text), FrameDirection.DOWNSTREAM
                )
            await self._record_usage()
            await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)

    def _reset_silence_timer(self, timeout: float) -> None:
        self._cancel_silence_timer()
        self._silence_task = asyncio.create_task(self._silence_watchdog(timeout))

    async def _silence_watchdog(self, timeout: float) -> None:
        await asyncio.sleep(timeout)
        self._silence_task = None
        event = CallEvent(type=EventType.SILENCE)
        try:
            action = await self._dm.handle_event(event)
            await self._push_action(action)
        except Exception:
            logger.exception("Error handling silence event")
            await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)

    def _cancel_silence_timer(self) -> None:
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
            self._silence_task = None
