from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pipecat.frames.frames import EndTaskFrame, TextFrame, TranscriptionFrame
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


class PipelineAdapter(FrameProcessor):
    def __init__(self, dialogue_manager: DialogueManager, **kwargs) -> None:
        super().__init__(**kwargs)
        self._dm = dialogue_manager
        self._started = False
        self._silence_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not self._started:
            self._started = True
            try:
                action = await self._dm.start()
                await self._push_action(action)
            except Exception:
                logger.exception("Error in DialogueManager.start()")
                await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)
                return

        if isinstance(frame, TranscriptionFrame):
            if not frame.finalized:
                return
            await self._handle_transcription(frame)
            return

        await self.push_frame(frame, direction)

    async def _handle_transcription(self, frame: TranscriptionFrame) -> None:
        self._cancel_silence_timer()
        event = CallEvent(type=EventType.TRANSCRIPTION, text=frame.text)
        try:
            action = await self._dm.handle_event(event)
            await self._push_action(action)
        except Exception:
            logger.exception("Error in DialogueManager.handle_event()")
            await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)

    async def _push_action(self, action: Action) -> None:
        if action.type in (ActionType.ASK, ActionType.SPEAK):
            if action.text:
                await self.push_frame(
                    TextFrame(text=action.text), FrameDirection.DOWNSTREAM
                )
            self._reset_silence_timer(action.timeout_s)
        elif action.type == ActionType.END_CALL:
            self._cancel_silence_timer()
            if action.text:
                await self.push_frame(
                    TextFrame(text=action.text), FrameDirection.DOWNSTREAM
                )
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
