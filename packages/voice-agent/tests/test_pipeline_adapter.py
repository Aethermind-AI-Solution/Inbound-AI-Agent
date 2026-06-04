from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.voice_agent.dialogue.models import Action, ActionType, CallEvent, EventType


@pytest.fixture
def mock_dm():
    dm = AsyncMock()
    dm.start = AsyncMock(
        return_value=Action(type=ActionType.ASK, text="Hello, thanks for calling!")
    )
    dm.handle_event = AsyncMock(
        return_value=Action(type=ActionType.ASK, text="What can I help you with?")
    )
    return dm


@pytest.fixture
def adapter(mock_dm):
    from packages.voice_agent.pipeline_adapter import PipelineAdapter
    return PipelineAdapter(mock_dm)


def make_transcription_frame(text: str, finalized: bool = True):
    from pipecat.frames.frames import TranscriptionFrame
    return TranscriptionFrame(
        text=text,
        user_id="user",
        timestamp="0",
        finalized=finalized,
    )


def make_generic_frame():
    return MagicMock()


class TestStartOnFirstFrame:
    @pytest.mark.asyncio
    async def test_start_called_on_first_frame(self, adapter, mock_dm):
        frame = make_generic_frame()
        await adapter.process_frame(frame, MagicMock())
        mock_dm.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_not_called_on_second_frame(self, adapter, mock_dm):
        frame1 = make_generic_frame()
        frame2 = make_generic_frame()
        await adapter.process_frame(frame1, MagicMock())
        await adapter.process_frame(frame2, MagicMock())
        mock_dm.start.assert_called_once()


class TestInboundTranslation:
    @pytest.mark.asyncio
    async def test_finalized_transcription_calls_handle_event(self, adapter, mock_dm):
        adapter._started = True
        frame = make_transcription_frame("I want to book an appointment", finalized=True)

        await adapter.process_frame(frame, MagicMock())

        mock_dm.handle_event.assert_called_once()
        event = mock_dm.handle_event.call_args[0][0]
        assert event.type == EventType.TRANSCRIPTION
        assert event.text == "I want to book an appointment"

    @pytest.mark.asyncio
    async def test_non_finalized_transcription_ignored(self, adapter, mock_dm):
        adapter._started = True
        frame = make_transcription_frame("I want to", finalized=False)

        await adapter.process_frame(frame, MagicMock())

        mock_dm.handle_event.assert_not_called()


class TestOutboundTranslation:
    @pytest.mark.asyncio
    async def test_ask_action_pushes_text_frame(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.return_value = Action(
            type=ActionType.ASK, text="Which service?"
        )

        frame = make_transcription_frame("hello", finalized=True)
        await adapter.process_frame(frame, MagicMock())

        assert len(pushed_frames) >= 1
        assert pushed_frames[0].text == "Which service?"

    @pytest.mark.asyncio
    async def test_end_call_pushes_text_then_end_task(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.return_value = Action(
            type=ActionType.END_CALL, text="Goodbye!"
        )

        frame = make_transcription_frame("bye", finalized=True)
        await adapter.process_frame(frame, MagicMock())

        assert len(pushed_frames) == 2
        assert pushed_frames[0].text == "Goodbye!"

    @pytest.mark.asyncio
    async def test_end_call_no_text_pushes_only_end_task(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.return_value = Action(type=ActionType.END_CALL)

        frame = make_transcription_frame("bye", finalized=True)
        await adapter.process_frame(frame, MagicMock())

        assert len(pushed_frames) == 1


class TestErrorBoundary:
    @pytest.mark.asyncio
    async def test_dm_exception_pushes_end_task(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.side_effect = RuntimeError("boom")

        frame = make_transcription_frame("hello", finalized=True)
        await adapter.process_frame(frame, MagicMock())

        assert len(pushed_frames) == 1


class TestMultipleTranscriptions:
    @pytest.mark.asyncio
    async def test_each_transcription_gets_own_handle_event(self, adapter, mock_dm):
        adapter._started = True
        adapter.push_frame = AsyncMock()

        for text in ["hello", "I want a haircut", "tomorrow"]:
            frame = make_transcription_frame(text, finalized=True)
            await adapter.process_frame(frame, MagicMock())

        assert mock_dm.handle_event.call_count == 3
