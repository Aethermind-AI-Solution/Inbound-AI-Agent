import pytest

from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.select_booking import SelectBookingState
from packages.voice_agent.tests.test_states.conftest import make_transcription


class TestSelectBookingState:
    @pytest.mark.asyncio
    async def test_enter_lists_bookings(self, deps, context):
        context.intent = "cancel"
        state = SelectBookingState(deps)
        state._bookings = [
            {"id": "b1", "service_id": "s1", "start_ts": "2026-06-10T10:00:00"},
            {"id": "b2", "service_id": "s2", "start_ts": "2026-06-12T14:00:00"},
        ]
        action = await state.enter(context)
        assert action.type == ActionType.ASK

    @pytest.mark.asyncio
    async def test_select_by_number_cancel(self, deps, context):
        context.intent = "cancel"
        state = SelectBookingState(deps)
        state._bookings = [
            {"id": "b1", "service_id": "s1", "start_ts": "2026-06-10T10:00:00"},
            {"id": "b2", "service_id": "s2", "start_ts": "2026-06-12T14:00:00"},
        ]
        await state.enter(context)
        action = await state.handle(make_transcription("1"), context)
        assert context.slots.booking_id == "b1"
        assert action.next_state == CallState.CONFIRM_CANCEL
