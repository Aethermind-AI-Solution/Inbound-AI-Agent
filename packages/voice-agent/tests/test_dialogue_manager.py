from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from packages.voice_agent.data.adapter import BookingResult, CallerInfo, HoldResult
from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker
from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore
from packages.voice_agent.dialogue.manager import DialogueManager
from packages.voice_agent.dialogue.models import (
    ActionType,
    CallEvent,
    CallState,
    EventType,
)
from packages.voice_agent.dialogue.nlu.stub import StubNLUService
from packages.voice_agent.resolver.date_resolver import (
    ResolutionResult,
    ResolutionStatus,
    ResolvedSlot,
)
from packages.voice_agent.tests.test_states.conftest import make_tenant_config

IST = ZoneInfo("Asia/Kolkata")


def make_event(text: str) -> CallEvent:
    return CallEvent(type=EventType.TRANSCRIPTION, text=text)


@pytest.fixture
def config():
    return make_tenant_config()


@pytest.fixture
def data_adapter():
    adapter = MagicMock()
    adapter.resolve_or_create_caller.return_value = CallerInfo(
        id="c1", phone="+919876543210", tenant_id="t1", verified_at=None, is_new=False
    )
    adapter.check_availability.return_value = [
        {"resource_id": "r1", "resource_name": "Priya", "blocked_slots": []},
    ]
    adapter.hold_slot.return_value = HoldResult(
        hold_id="h1", resource_id="r1",
        start_ts=datetime(2026, 6, 10, 10, 0, tzinfo=IST),
        expires_at=datetime(2026, 6, 10, 10, 3, tzinfo=IST),
    )
    adapter.confirm_booking.return_value = BookingResult(
        booking_id="b1", idempotency_key="h1", status="confirmed"
    )
    adapter.lookup_bookings.return_value = [
        {"id": "b1", "service_id": "s1", "start_ts": "2026-06-10T10:00:00"}
    ]
    adapter.cancel_booking.return_value = True
    return adapter


@pytest.fixture
def resolver():
    r = MagicMock()
    dt = datetime(2026, 6, 10, 10, 0, tzinfo=IST)
    r.resolve.return_value = ResolutionResult(
        status=ResolutionStatus.RESOLVED,
        slot=ResolvedSlot(date=dt.date(), time=dt.time(), datetime_ist=dt),
    )
    return r


@pytest.fixture
def manager(config, data_adapter, resolver):
    nlu = StubNLUService()
    checkpoint = InMemoryCheckpointStore()
    m = DialogueManager(
        config=config,
        data_adapter=data_adapter,
        nlu=nlu,
        checkpoint_store=checkpoint,
        caller_phone="+919876543210",
        call_id="call-001",
    )
    m._make_date_resolver = MagicMock(return_value=resolver)
    return m


class TestNewBookingFlow:
    @pytest.mark.asyncio
    async def test_full_happy_path(self, manager):
        # start() → GREETING → IDENTIFY_CALLER → INTENT (ask)
        action = await manager.start()
        assert action.type == ActionType.ASK
        assert manager.current_state.name == CallState.INTENT

        # "book appointment" → COLLECT_SERVICE (ask)
        action = await manager.handle_event(make_event("I want to book an appointment"))
        assert action.type == ActionType.ASK
        assert manager.current_state.name == CallState.COLLECT_SERVICE

        # "haircut" → COLLECT_DATETIME (ask)
        action = await manager.handle_event(make_event("haircut"))
        assert action.type == ActionType.ASK
        assert manager.current_state.name == CallState.COLLECT_DATETIME

        # "tomorrow at 10am" → OFFER_SLOTS (ask)
        action = await manager.handle_event(make_event("tomorrow at 10am"))
        assert action.type == ActionType.ASK
        assert manager.current_state.name == CallState.OFFER_SLOTS

        # "Priya" → READ_BACK (ask)
        action = await manager.handle_event(make_event("Priya"))
        assert action.type == ActionType.ASK
        assert manager.current_state.name == CallState.READ_BACK

        # "yes" → CONFIRM → CLOSE (end_call)
        action = await manager.handle_event(make_event("yes"))
        assert action.type == ActionType.END_CALL


class TestCancelFlow:
    @pytest.mark.asyncio
    async def test_cancel_single_booking(self, manager):
        action = await manager.start()
        assert manager.current_state.name == CallState.INTENT

        action = await manager.handle_event(make_event("cancel my booking"))
        assert manager.current_state.name == CallState.CONFIRM_CANCEL

        action = await manager.handle_event(make_event("yes"))
        assert action.type == ActionType.END_CALL


class TestFallbackFlow:
    @pytest.mark.asyncio
    async def test_four_unknown_intents_go_to_callback(self, manager):
        await manager.start()
        await manager.handle_event(make_event("blah"))
        await manager.handle_event(make_event("blah"))
        await manager.handle_event(make_event("blah"))
        action = await manager.handle_event(make_event("blah"))
        assert manager.current_state.name == CallState.CALLBACK_CAPTURE

        action = await manager.handle_event(make_event("yes"))
        assert action.type == ActionType.END_CALL


class TestGuardrails:
    @pytest.mark.asyncio
    async def test_max_turns_triggers_callback(self, manager):
        manager.context.tenant_config.guardrails.max_turns = 2
        await manager.start()
        await manager.handle_event(make_event("blah"))  # turn 1
        action = await manager.handle_event(make_event("blah"))  # turn 2
        # turn 3 should hit guardrail
        action = await manager.handle_event(make_event("blah"))
        assert manager.current_state.name == CallState.CALLBACK_CAPTURE


class TestGlobalEvents:
    @pytest.mark.asyncio
    async def test_hangup_ends_call(self, manager):
        await manager.start()
        action = await manager.handle_event(CallEvent(type=EventType.HANGUP))
        assert action.type == ActionType.END_CALL

    @pytest.mark.asyncio
    async def test_two_silences_go_to_callback(self, manager):
        await manager.start()
        await manager.handle_event(CallEvent(type=EventType.SILENCE))
        action = await manager.handle_event(CallEvent(type=EventType.SILENCE))
        assert manager.current_state.name == CallState.CALLBACK_CAPTURE

    @pytest.mark.asyncio
    async def test_error_event_goes_to_callback(self, manager):
        await manager.start()
        action = await manager.handle_event(CallEvent(type=EventType.ERROR))
        assert manager.current_state.name == CallState.CALLBACK_CAPTURE


class TestCheckpointResume:
    @pytest.mark.asyncio
    async def test_checkpoint_written_on_transition(self, manager):
        await manager.start()
        data = await manager.checkpoint_store.load("+919876543210")
        assert data is not None
        assert data["state"] == CallState.INTENT


class TestBudgetGate:
    @pytest.mark.asyncio
    async def test_under_budget_proceeds_normally(self, config, data_adapter, resolver):
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        nlu = StubNLUService()
        checkpoint = InMemoryCheckpointStore()
        m = DialogueManager(
            config=config, data_adapter=data_adapter, nlu=nlu,
            checkpoint_store=checkpoint, caller_phone="+919876543210",
            call_id="call-001", budget_tracker=tracker,
        )
        m._make_date_resolver = MagicMock(return_value=resolver)
        action = await m.start()
        assert action.type == ActionType.ASK
        assert m.current_state.name == CallState.INTENT

    @pytest.mark.asyncio
    async def test_over_budget_goes_to_callback(self, config, data_adapter, resolver):
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        await tracker.record_usage("t1", 150000.0)  # 5000 INR
        nlu = StubNLUService()
        checkpoint = InMemoryCheckpointStore()
        m = DialogueManager(
            config=config, data_adapter=data_adapter, nlu=nlu,
            checkpoint_store=checkpoint, caller_phone="+919876543210",
            call_id="call-001", budget_tracker=tracker,
        )
        m._make_date_resolver = MagicMock(return_value=resolver)
        action = await m.start()
        assert m.current_state.name == CallState.CALLBACK_CAPTURE
        assert m.context.fallback_reason == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_record_call_usage(self, config, data_adapter, resolver):
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        nlu = StubNLUService()
        checkpoint = InMemoryCheckpointStore()
        m = DialogueManager(
            config=config, data_adapter=data_adapter, nlu=nlu,
            checkpoint_store=checkpoint, caller_phone="+919876543210",
            call_id="call-001", budget_tracker=tracker,
        )
        await m.record_call_usage(300.0)  # 5 min = 10 INR = 1000 paisa
        assert await tracker.check_budget("t1", 9.0) is False
        assert await tracker.check_budget("t1", 11.0) is True
