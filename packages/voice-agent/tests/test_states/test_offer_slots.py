from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from packages.voice_agent.data.adapter import CallerInfo, HoldResult, SlotConflictError
from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.offer_slots import OfferSlotsState
from packages.voice_agent.tests.test_states.conftest import make_transcription

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def context_with_service(context):
    context.slots.service_id = "s1"
    context.slots.service_name = "Haircut"
    context.slots.datetime_ist = datetime(2026, 6, 10, 10, 0, tzinfo=IST)
    context.intent = "new_booking"
    context.caller = CallerInfo(id="c1", phone="+919876543210", tenant_id="t1", verified_at=None)
    return context


class TestOfferSlotsState:
    @pytest.mark.asyncio
    async def test_enter_presents_available_slots(self, deps, context_with_service):
        deps.data_adapter.check_availability.return_value = [
            {"resource_id": "r1", "resource_name": "Priya", "blocked_slots": []},
            {"resource_id": "r2", "resource_name": "Rahul", "blocked_slots": []},
        ]
        state = OfferSlotsState(deps)
        action = await state.enter(context_with_service)
        assert action.type == ActionType.ASK
        assert "Priya" in action.text

    @pytest.mark.asyncio
    async def test_enter_no_availability_goes_to_callback(self, deps, context_with_service):
        deps.data_adapter.check_availability.return_value = []
        state = OfferSlotsState(deps)
        action = await state.enter(context_with_service)
        assert action.next_state == CallState.CALLBACK_CAPTURE

    @pytest.mark.asyncio
    async def test_handle_slot_selection_holds_and_transitions(self, deps, context_with_service):
        deps.data_adapter.check_availability.return_value = [
            {"resource_id": "r1", "resource_name": "Priya", "blocked_slots": []},
        ]
        deps.data_adapter.hold_slot.return_value = HoldResult(
            hold_id="h1", resource_id="r1",
            start_ts=datetime(2026, 6, 10, 10, 0, tzinfo=IST),
            expires_at=datetime(2026, 6, 10, 10, 3, tzinfo=IST),
        )
        state = OfferSlotsState(deps)
        await state.enter(context_with_service)
        action = await state.handle(make_transcription("Priya"), context_with_service)
        assert context_with_service.slots.hold_id == "h1"
        assert action.next_state == CallState.READ_BACK

    @pytest.mark.asyncio
    async def test_handle_slot_conflict_reprompts(self, deps, context_with_service):
        deps.data_adapter.check_availability.return_value = [
            {"resource_id": "r1", "resource_name": "Priya", "blocked_slots": []},
            {"resource_id": "r2", "resource_name": "Rahul", "blocked_slots": []},
        ]
        deps.data_adapter.hold_slot.side_effect = SlotConflictError("taken")
        state = OfferSlotsState(deps)
        await state.enter(context_with_service)
        action = await state.handle(make_transcription("Priya"), context_with_service)
        assert action.type == ActionType.ASK
        assert action.next_state is None
