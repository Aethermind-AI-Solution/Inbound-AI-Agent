# PipelineAdapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Pipecat `FrameProcessor` that bridges DialogueManager into the voice pipeline, plus a mock DataAdapter and a local runner script for end-to-end mic testing.

**Architecture:** Single `PipelineAdapter(FrameProcessor)` translates `TranscriptionFrame` → `CallEvent` and `Action` → `TextFrame`/`EndTaskFrame`. Manages its own silence timer via `asyncio.Task`. A `MockDataAdapter` provides canned responses so no Supabase is needed. `run_local.py` wires everything into a real STT/TTS pipeline with `LocalAudioTransport`.

**Tech Stack:** Pipecat v1.3.0 (FrameProcessor, Pipeline, LocalAudioTransport), asyncio, pytest + pytest-asyncio

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/voice-agent/data/mock_adapter.py` | In-memory `DataAdapter` replacement with canned salon data |
| `packages/voice-agent/tests/test_mock_adapter.py` | Tests for MockDataAdapter |
| `packages/voice-agent/pipeline_adapter.py` | `PipelineAdapter(FrameProcessor)` — frame↔event translation + silence timer |
| `packages/voice-agent/tests/test_pipeline_adapter.py` | Unit tests for PipelineAdapter (mocked DialogueManager) |
| `packages/voice-agent/run_local.py` | Runner script — local mic pipeline with mock deps |

---

### Task 1: MockDataAdapter

Build an in-memory DataAdapter that returns canned salon data. Every state that calls `self.deps.data_adapter.*` uses `asyncio.to_thread(...)`, so the mock must have the same synchronous method signatures as the real `DataAdapter`.

**Files:**
- Create: `packages/voice-agent/data/mock_adapter.py`
- Create: `packages/voice-agent/tests/test_mock_adapter.py`

- [ ] **Step 1: Write failing tests for MockDataAdapter**

Create `packages/voice-agent/tests/test_mock_adapter.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from packages.voice_agent.data.mock_adapter import MockDataAdapter


class TestMockResolveOrCreateCaller:
    def test_returns_caller_info(self):
        adapter = MockDataAdapter()
        caller = adapter.resolve_or_create_caller("t1", "+919876543210")
        assert caller.id == "caller-1"
        assert caller.phone == "+919876543210"
        assert caller.tenant_id == "t1"

    def test_uses_provided_phone(self):
        adapter = MockDataAdapter()
        caller = adapter.resolve_or_create_caller("t1", "+911111111111")
        assert caller.phone == "+911111111111"


class TestMockCheckAvailability:
    def test_returns_resources_with_no_blocked_slots(self):
        adapter = MockDataAdapter()
        now = datetime.now(tz=UTC)
        result = adapter.check_availability("t1", "s1", "stylist", now, now)
        assert len(result) >= 1
        assert result[0]["resource_id"]
        assert result[0]["resource_name"]
        assert result[0]["blocked_slots"] == []


class TestMockHoldSlot:
    def test_returns_hold_result(self):
        adapter = MockDataAdapter()
        now = datetime.now(tz=UTC)
        hold = adapter.hold_slot("t1", "r1", now, "caller-1")
        assert hold.hold_id
        assert hold.resource_id == "r1"
        assert hold.expires_at > hold.start_ts


class TestMockConfirmBooking:
    def test_returns_booking_result(self):
        adapter = MockDataAdapter()
        now = datetime.now(tz=UTC)
        result = adapter.confirm_booking(
            "t1", "hold-1", "s1", "r1", "caller-1", now, now, {}
        )
        assert result.booking_id
        assert result.status == "confirmed"

    def test_idempotent_on_same_hold_id(self):
        adapter = MockDataAdapter()
        now = datetime.now(tz=UTC)
        r1 = adapter.confirm_booking("t1", "hold-1", "s1", "r1", "caller-1", now, now, {})
        r2 = adapter.confirm_booking("t1", "hold-1", "s1", "r1", "caller-1", now, now, {})
        assert r1.booking_id == r2.booking_id


class TestMockLookupBookings:
    def test_returns_list_of_bookings(self):
        adapter = MockDataAdapter()
        bookings = adapter.lookup_bookings("t1", "caller-1")
        assert isinstance(bookings, list)
        assert len(bookings) >= 1
        assert "id" in bookings[0]
        assert bookings[0]["status"] == "confirmed"


class TestMockCancelBooking:
    def test_returns_true(self):
        adapter = MockDataAdapter()
        assert adapter.cancel_booking("t1", "booking-1", "caller-1") is True

    def test_lookup_after_cancel_shows_cancelled(self):
        adapter = MockDataAdapter()
        adapter.cancel_booking("t1", "booking-1", "caller-1")
        bookings = adapter.lookup_bookings("t1", "caller-1")
        cancelled = [b for b in bookings if b["id"] == "booking-1"]
        if cancelled:
            assert cancelled[0]["status"] == "cancelled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/abhisheksingh/Documents/Inbound call assistant" && python -m pytest packages/voice-agent/tests/test_mock_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.data.mock_adapter'`

- [ ] **Step 3: Implement MockDataAdapter**

Create `packages/voice-agent/data/mock_adapter.py`:

```python
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from packages.voice_agent.data.adapter import BookingResult, CallerInfo, HoldResult


class MockDataAdapter:
    def __init__(self) -> None:
        self._bookings: dict[str, dict[str, Any]] = {
            "booking-1": {
                "id": "booking-1",
                "service_id": "s1",
                "resource_id": "r1",
                "caller_id": "caller-1",
                "start_ts": (datetime.now(tz=UTC) + timedelta(days=2)).isoformat(),
                "status": "confirmed",
                "idempotency_key": "hold-existing",
            },
        }
        self._confirmed_holds: dict[str, str] = {}

    def resolve_or_create_caller(self, tenant_id: str, phone: str) -> CallerInfo:
        return CallerInfo(
            id="caller-1",
            phone=phone,
            tenant_id=tenant_id,
            verified_at=None,
        )

    def check_availability(
        self,
        tenant_id: str,
        service_id: str,
        resource_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        return [
            {"resource_id": "r1", "resource_name": "Priya", "blocked_slots": []},
            {"resource_id": "r2", "resource_name": "Rahul", "blocked_slots": []},
        ]

    def hold_slot(
        self,
        tenant_id: str,
        resource_id: str,
        start_ts: datetime,
        caller_id: str,
        hold_ttl_seconds: int = 180,
    ) -> HoldResult:
        hold_id = f"hold-{uuid.uuid4().hex[:8]}"
        return HoldResult(
            hold_id=hold_id,
            resource_id=resource_id,
            start_ts=start_ts,
            expires_at=start_ts + timedelta(seconds=hold_ttl_seconds),
        )

    def confirm_booking(
        self,
        tenant_id: str,
        hold_id: str,
        service_id: str,
        resource_id: str,
        caller_id: str,
        start_ts: datetime,
        end_ts: datetime,
        custom_values: dict[str, Any],
    ) -> BookingResult:
        if hold_id in self._confirmed_holds:
            return BookingResult(
                booking_id=self._confirmed_holds[hold_id],
                idempotency_key=hold_id,
                status="confirmed",
            )
        booking_id = f"bk-{uuid.uuid4().hex[:8]}"
        self._confirmed_holds[hold_id] = booking_id
        self._bookings[booking_id] = {
            "id": booking_id,
            "service_id": service_id,
            "resource_id": resource_id,
            "caller_id": caller_id,
            "start_ts": start_ts.isoformat(),
            "status": "confirmed",
            "idempotency_key": hold_id,
        }
        return BookingResult(
            booking_id=booking_id,
            idempotency_key=hold_id,
            status="confirmed",
        )

    def lookup_bookings(self, tenant_id: str, caller_id: str) -> list[dict[str, Any]]:
        return [
            b for b in self._bookings.values()
            if b["caller_id"] == caller_id and b["status"] == "confirmed"
        ]

    def cancel_booking(self, tenant_id: str, booking_id: str, caller_id: str) -> bool:
        if booking_id in self._bookings:
            self._bookings[booking_id]["status"] = "cancelled"
            return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/abhisheksingh/Documents/Inbound call assistant" && python -m pytest packages/voice-agent/tests/test_mock_adapter.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/data/mock_adapter.py packages/voice-agent/tests/test_mock_adapter.py
git commit -m "feat: add MockDataAdapter with canned salon data for local testing"
```

---

### Task 2: PipelineAdapter — Core Frame Translation

Build the `PipelineAdapter` class with `process_frame` handling finalized transcriptions and action-to-frame translation. No silence timer yet — that's Task 3.

**Files:**
- Create: `packages/voice-agent/pipeline_adapter.py`
- Create: `packages/voice-agent/tests/test_pipeline_adapter.py`

- [ ] **Step 1: Write failing tests for inbound/outbound translation**

Create `packages/voice-agent/tests/test_pipeline_adapter.py`:

```python
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
    frame = MagicMock()
    frame.__class__.__name__ = "TranscriptionFrame"
    frame.text = text
    frame.finalized = finalized
    return frame


def make_text_frame_class():
    class FakeTextFrame:
        def __init__(self, text: str):
            self.text = text
    return FakeTextFrame


def make_end_task_frame_class():
    class FakeEndTaskFrame:
        def __init__(self, reason=None):
            self.reason = reason
    return FakeEndTaskFrame


class TestStartOnFirstFrame:
    @pytest.mark.asyncio
    async def test_start_called_on_first_frame(self, adapter, mock_dm):
        frame = MagicMock()
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame"):
            await adapter.process_frame(frame, MagicMock())
        mock_dm.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_not_called_on_second_frame(self, adapter, mock_dm):
        frame1 = MagicMock()
        frame2 = MagicMock()
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame"):
            await adapter.process_frame(frame1, MagicMock())
            await adapter.process_frame(frame2, MagicMock())
        mock_dm.start.assert_called_once()


class TestInboundTranslation:
    @pytest.mark.asyncio
    async def test_finalized_transcription_calls_handle_event(self, adapter, mock_dm):
        adapter._started = True
        frame = MagicMock()
        frame.text = "I want to book an appointment"
        frame.finalized = True

        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
            await adapter.process_frame(frame, MagicMock())

        mock_dm.handle_event.assert_called_once()
        event = mock_dm.handle_event.call_args[0][0]
        assert event.type == EventType.TRANSCRIPTION
        assert event.text == "I want to book an appointment"

    @pytest.mark.asyncio
    async def test_non_finalized_transcription_ignored(self, adapter, mock_dm):
        adapter._started = True
        frame = MagicMock()
        frame.text = "I want to"
        frame.finalized = False

        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
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

        frame = MagicMock()
        frame.text = "hello"
        frame.finalized = True
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
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

        frame = MagicMock()
        frame.text = "bye"
        frame.finalized = True
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
            await adapter.process_frame(frame, MagicMock())

        assert len(pushed_frames) == 2
        assert pushed_frames[0].text == "Goodbye!"

    @pytest.mark.asyncio
    async def test_end_call_no_text_pushes_only_end_task(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.return_value = Action(type=ActionType.END_CALL)

        frame = MagicMock()
        frame.text = "bye"
        frame.finalized = True
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
            await adapter.process_frame(frame, MagicMock())

        assert len(pushed_frames) == 1


class TestErrorBoundary:
    @pytest.mark.asyncio
    async def test_dm_exception_pushes_end_task(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.side_effect = RuntimeError("boom")

        frame = MagicMock()
        frame.text = "hello"
        frame.finalized = True
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
            await adapter.process_frame(frame, MagicMock())

        assert len(pushed_frames) == 1


class TestMultipleTranscriptions:
    @pytest.mark.asyncio
    async def test_each_transcription_gets_own_handle_event(self, adapter, mock_dm):
        adapter._started = True
        adapter.push_frame = AsyncMock()

        for text in ["hello", "I want a haircut", "tomorrow"]:
            frame = MagicMock()
            frame.text = text
            frame.finalized = True
            with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
                TF.__instancecheck__ = lambda self, x: x is frame
                await adapter.process_frame(frame, MagicMock())

        assert mock_dm.handle_event.call_count == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/abhisheksingh/Documents/Inbound call assistant" && python -m pytest packages/voice-agent/tests/test_pipeline_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.pipeline_adapter'`

- [ ] **Step 3: Implement PipelineAdapter (core translation, no silence timer)**

Create `packages/voice-agent/pipeline_adapter.py`:

```python
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
        self._last_timeout: float = 10.0

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
            self._last_timeout = action.timeout_s
            self._reset_silence_timer(self._last_timeout)
        elif action.type == ActionType.END_CALL:
            self._cancel_silence_timer()
            if action.text:
                await self.push_frame(
                    TextFrame(text=action.text), FrameDirection.DOWNSTREAM
                )
            await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)

    def _reset_silence_timer(self, timeout: float) -> None:
        self._cancel_silence_timer()
        self._silence_task = asyncio.ensure_future(self._silence_watchdog(timeout))

    async def _silence_watchdog(self, timeout: float) -> None:
        await asyncio.sleep(timeout)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/abhisheksingh/Documents/Inbound call assistant" && python -m pytest packages/voice-agent/tests/test_pipeline_adapter.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/pipeline_adapter.py packages/voice-agent/tests/test_pipeline_adapter.py
git commit -m "feat: add PipelineAdapter with frame-to-event and action-to-frame translation"
```

---

### Task 3: PipelineAdapter — Silence Timer Tests

Add tests specifically for the silence timer behavior: fires after timeout, resets on transcription, cancels on end call.

**Files:**
- Modify: `packages/voice-agent/tests/test_pipeline_adapter.py`

- [ ] **Step 1: Add silence timer tests**

Append to `packages/voice-agent/tests/test_pipeline_adapter.py`:

```python
class TestSilenceTimer:
    @pytest.mark.asyncio
    async def test_silence_fires_after_timeout(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.return_value = Action(
            type=ActionType.ASK, text="Are you still there?"
        )

        adapter._reset_silence_timer(0.05)
        await asyncio.sleep(0.1)

        mock_dm.handle_event.assert_called_once()
        event = mock_dm.handle_event.call_args[0][0]
        assert event.type == EventType.SILENCE

    @pytest.mark.asyncio
    async def test_silence_timer_resets_on_transcription(self, adapter, mock_dm):
        adapter._started = True
        adapter.push_frame = AsyncMock()

        adapter._reset_silence_timer(0.1)
        await asyncio.sleep(0.05)

        frame = MagicMock()
        frame.text = "hello"
        frame.finalized = True
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
            await adapter.process_frame(frame, MagicMock())

        await asyncio.sleep(0.1)
        silence_calls = [
            c for c in mock_dm.handle_event.call_args_list
            if c[0][0].type == EventType.SILENCE
        ]
        assert len(silence_calls) == 0

    @pytest.mark.asyncio
    async def test_silence_timer_cancelled_on_end_call(self, adapter, mock_dm):
        adapter._started = True
        adapter.push_frame = AsyncMock()

        adapter._reset_silence_timer(0.1)

        mock_dm.handle_event.return_value = Action(type=ActionType.END_CALL, text="Bye")
        frame = MagicMock()
        frame.text = "bye"
        frame.finalized = True
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
            await adapter.process_frame(frame, MagicMock())

        await asyncio.sleep(0.15)
        silence_calls = [
            c for c in mock_dm.handle_event.call_args_list
            if c[0][0].type == EventType.SILENCE
        ]
        assert len(silence_calls) == 0

    @pytest.mark.asyncio
    async def test_silence_timer_uses_action_timeout(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.return_value = Action(
            type=ActionType.ASK, text="Pick a service", timeout_s=0.05
        )

        frame = MagicMock()
        frame.text = "hello"
        frame.finalized = True
        with patch("packages.voice_agent.pipeline_adapter.TranscriptionFrame") as TF:
            TF.__instancecheck__ = lambda self, x: x is frame
            await adapter.process_frame(frame, MagicMock())

        mock_dm.handle_event.reset_mock()
        mock_dm.handle_event.return_value = Action(
            type=ActionType.ASK, text="Are you still there?"
        )
        await asyncio.sleep(0.1)

        silence_calls = [
            c for c in mock_dm.handle_event.call_args_list
            if c[0][0].type == EventType.SILENCE
        ]
        assert len(silence_calls) == 1
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd "/Users/abhisheksingh/Documents/Inbound call assistant" && python -m pytest packages/voice-agent/tests/test_pipeline_adapter.py -v`
Expected: All 13 tests PASS (9 from Task 2 + 4 new)

- [ ] **Step 3: Commit**

```bash
git add packages/voice-agent/tests/test_pipeline_adapter.py
git commit -m "test: add silence timer tests for PipelineAdapter"
```

---

### Task 4: Runner Script (`run_local.py`)

Wire PipelineAdapter into a real Pipecat pipeline with LocalAudioTransport, Deepgram STT, and a TTS service for local mic testing.

**Files:**
- Create: `packages/voice-agent/run_local.py`

- [ ] **Step 1: Create the runner script**

Create `packages/voice-agent/run_local.py`:

```python
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

    from packages.voice_agent.config.models import TenantConfig
    from packages.voice_agent.data.mock_adapter import MockDataAdapter
    from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore
    from packages.voice_agent.dialogue.manager import DialogueManager
    from packages.voice_agent.dialogue.nlu.stub import StubNLUService
    from packages.voice_agent.pipeline_adapter import PipelineAdapter
    from packages.voice_agent.tests.test_states.conftest import make_tenant_config

    config = make_tenant_config()
    data = MockDataAdapter()
    nlu = StubNLUService()
    checkpoint = InMemoryCheckpointStore()

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
```

- [ ] **Step 2: Verify the script parses correctly**

Run: `cd "/Users/abhisheksingh/Documents/Inbound call assistant" && python -c "import ast; ast.parse(open('packages/voice-agent/run_local.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add packages/voice-agent/run_local.py
git commit -m "feat: add run_local.py runner script for local mic voice testing"
```

---

### Task 5: Run All Tests and Verify

Run the full test suite to make sure nothing is broken.

**Files:**
- No changes — verification only

- [ ] **Step 1: Run entire test suite**

Run: `cd "/Users/abhisheksingh/Documents/Inbound call assistant" && python -m pytest packages/voice-agent/tests/ -v --tb=short`
Expected: All tests PASS (238 existing + ~22 new = ~260 total)

- [ ] **Step 2: If any failures, fix and re-run**

Fix any failures, then re-run the full suite until green.

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve test issues from PipelineAdapter integration"
```

(Skip this step if no fixes were needed.)
