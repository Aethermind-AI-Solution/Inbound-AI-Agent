from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.frames.frames import EndTaskFrame, TTSSpeakFrame, TranscriptionFrame


class TestGuardrailProcessor:
    @pytest.mark.asyncio
    async def test_normal_frame_passes_through(self):
        from packages.voice_agent.flows.guardrails import GuardrailProcessor

        proc = GuardrailProcessor(max_turns=20, max_seconds=300, call_start=time.monotonic())
        pushed = []
        proc.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed.append(f))

        frame = MagicMock()
        await proc.process_frame(frame, MagicMock())
        assert len(pushed) == 1
        assert pushed[0] is frame

    @pytest.mark.asyncio
    async def test_transcription_increments_turn_count(self):
        from packages.voice_agent.flows.guardrails import GuardrailProcessor

        proc = GuardrailProcessor(max_turns=20, max_seconds=300, call_start=time.monotonic())
        proc.push_frame = AsyncMock()

        frame = TranscriptionFrame(text="hello", user_id="u", timestamp="0")
        await proc.process_frame(frame, MagicMock())
        assert proc.turn_count == 1

    @pytest.mark.asyncio
    async def test_exceeds_max_turns_sends_end_task(self):
        from packages.voice_agent.flows.guardrails import GuardrailProcessor

        proc = GuardrailProcessor(max_turns=2, max_seconds=300, call_start=time.monotonic())
        pushed = []
        proc.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed.append(f))

        for i in range(3):
            frame = TranscriptionFrame(text=f"msg {i}", user_id="u", timestamp="0")
            await proc.process_frame(frame, MagicMock())

        tts = [f for f in pushed if isinstance(f, TTSSpeakFrame)]
        end = [f for f in pushed if isinstance(f, EndTaskFrame)]
        assert len(tts) == 1
        assert "call you back" in tts[0].text.lower()
        assert len(end) == 1

    @pytest.mark.asyncio
    async def test_exceeds_max_seconds_sends_end_task(self):
        from packages.voice_agent.flows.guardrails import GuardrailProcessor

        proc = GuardrailProcessor(max_turns=100, max_seconds=0, call_start=time.monotonic() - 1)
        pushed = []
        proc.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed.append(f))

        frame = TranscriptionFrame(text="hello", user_id="u", timestamp="0")
        await proc.process_frame(frame, MagicMock())

        end = [f for f in pushed if isinstance(f, EndTaskFrame)]
        assert len(end) == 1

    @pytest.mark.asyncio
    async def test_after_limit_all_transcriptions_dropped(self):
        from packages.voice_agent.flows.guardrails import GuardrailProcessor

        proc = GuardrailProcessor(max_turns=1, max_seconds=300, call_start=time.monotonic())
        pushed = []
        proc.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed.append(f))

        for i in range(5):
            frame = TranscriptionFrame(text=f"msg {i}", user_id="u", timestamp="0")
            await proc.process_frame(frame, MagicMock())

        end_count = sum(1 for f in pushed if isinstance(f, EndTaskFrame))
        assert end_count == 1

    @pytest.mark.asyncio
    async def test_non_transcription_frames_pass_after_limit(self):
        from packages.voice_agent.flows.guardrails import GuardrailProcessor

        proc = GuardrailProcessor(max_turns=1, max_seconds=300, call_start=time.monotonic())
        pushed = []
        proc.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed.append(f))

        frame1 = TranscriptionFrame(text="msg 0", user_id="u", timestamp="0")
        await proc.process_frame(frame1, MagicMock())
        frame2 = TranscriptionFrame(text="msg 1", user_id="u", timestamp="0")
        await proc.process_frame(frame2, MagicMock())

        other = MagicMock()
        await proc.process_frame(other, MagicMock())
        assert other in pushed
