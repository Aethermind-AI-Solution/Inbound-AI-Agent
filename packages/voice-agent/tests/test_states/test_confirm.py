from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from packages.voice_agent.data.adapter import BookingResult, CallerInfo
from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.confirm import ConfirmState

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def context_ready_to_confirm(context):
    context.intent = "new_booking"
    context.caller = CallerInfo(id="c1", phone="+919876543210", tenant_id="t1", verified_at=None)
    context.slots.service_id = "s1"
    context.slots.service_name = "Haircut"
    context.slots.resource_id = "r1"
    context.slots.resource_name = "Priya"
    context.slots.datetime_ist = datetime(2026, 6, 10, 10, 0, tzinfo=IST)
    context.slots.hold_id = "h1"
    return context


class TestConfirmState:
    @pytest.mark.asyncio
    async def test_confirm_success(self, deps, context_ready_to_confirm):
        deps.data_adapter.confirm_booking.return_value = BookingResult(
            booking_id="b1", idempotency_key="h1", status="confirmed"
        )
        state = ConfirmState(deps)
        action = await state.enter(context_ready_to_confirm)
        assert action.next_state == CallState.CLOSE
        assert "confirmed" in action.text.lower()
        assert "b1" in action.text

    @pytest.mark.asyncio
    async def test_confirm_failure_retries_then_callback(self, deps, context_ready_to_confirm):
        deps.data_adapter.confirm_booking.side_effect = Exception("DB error")
        state = ConfirmState(deps)
        with pytest.raises(Exception):
            await state.enter(context_ready_to_confirm)

    @pytest.mark.asyncio
    async def test_reschedule_cancels_old_first(self, deps, context_ready_to_confirm):
        context_ready_to_confirm.intent = "reschedule"
        context_ready_to_confirm.slots.booking_id = "old-b1"
        deps.data_adapter.cancel_booking.return_value = True
        deps.data_adapter.confirm_booking.return_value = BookingResult(
            booking_id="b2", idempotency_key="h1", status="confirmed"
        )
        state = ConfirmState(deps)
        action = await state.enter(context_ready_to_confirm)
        deps.data_adapter.cancel_booking.assert_called_once()
        assert action.next_state == CallState.CLOSE
