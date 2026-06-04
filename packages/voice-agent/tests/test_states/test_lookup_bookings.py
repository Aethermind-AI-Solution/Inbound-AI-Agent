import pytest

from packages.voice_agent.data.adapter import CallerInfo
from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.lookup_bookings import LookupBookingsState


@pytest.fixture
def context_with_caller(context):
    context.caller = CallerInfo(id="c1", phone="+919876543210", tenant_id="t1", verified_at=None)
    return context


class TestLookupBookingsState:
    @pytest.mark.asyncio
    async def test_no_bookings_goes_to_close(self, deps, context_with_caller):
        context_with_caller.intent = "cancel"
        deps.data_adapter.lookup_bookings.return_value = []
        state = LookupBookingsState(deps)
        action = await state.enter(context_with_caller)
        assert action.next_state == CallState.CLOSE
        assert "don't see" in action.text.lower()

    @pytest.mark.asyncio
    async def test_status_intent_goes_to_read_status(self, deps, context_with_caller):
        context_with_caller.intent = "status"
        deps.data_adapter.lookup_bookings.return_value = [
            {"id": "b1", "service_id": "s1", "start_ts": "2026-06-10T10:00:00"}
        ]
        state = LookupBookingsState(deps)
        action = await state.enter(context_with_caller)
        assert action.next_state == CallState.READ_STATUS

    @pytest.mark.asyncio
    async def test_single_booking_cancel_auto_selects(self, deps, context_with_caller):
        context_with_caller.intent = "cancel"
        deps.data_adapter.lookup_bookings.return_value = [
            {"id": "b1", "service_id": "s1", "start_ts": "2026-06-10T10:00:00"}
        ]
        state = LookupBookingsState(deps)
        action = await state.enter(context_with_caller)
        assert context_with_caller.slots.booking_id == "b1"
        assert action.next_state == CallState.CONFIRM_CANCEL

    @pytest.mark.asyncio
    async def test_multiple_bookings_goes_to_select(self, deps, context_with_caller):
        context_with_caller.intent = "cancel"
        deps.data_adapter.lookup_bookings.return_value = [
            {"id": "b1", "service_id": "s1", "start_ts": "2026-06-10T10:00:00"},
            {"id": "b2", "service_id": "s2", "start_ts": "2026-06-12T14:00:00"},
        ]
        state = LookupBookingsState(deps)
        action = await state.enter(context_with_caller)
        assert action.next_state == CallState.SELECT_BOOKING
