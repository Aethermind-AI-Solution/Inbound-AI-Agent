import pytest

from packages.voice_agent.data.adapter import CallerInfo
from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.confirm_cancel import ConfirmCancelState
from packages.voice_agent.tests.test_states.conftest import make_transcription


@pytest.fixture
def context_with_booking_to_cancel(context):
    context.intent = "cancel"
    context.caller = CallerInfo(id="c1", phone="+919876543210", tenant_id="t1", verified_at=None)
    context.slots.booking_id = "b1"
    context.slots.service_name = "Haircut"
    return context


class TestConfirmCancelState:
    @pytest.mark.asyncio
    async def test_enter_asks_confirmation(self, deps, context_with_booking_to_cancel):
        state = ConfirmCancelState(deps)
        action = await state.enter(context_with_booking_to_cancel)
        assert action.type == ActionType.ASK
        assert "cancel" in action.text.lower()

    @pytest.mark.asyncio
    async def test_yes_cancels_and_closes(self, deps, context_with_booking_to_cancel):
        deps.data_adapter.cancel_booking.return_value = True
        state = ConfirmCancelState(deps)
        await state.enter(context_with_booking_to_cancel)
        action = await state.handle(make_transcription("yes"), context_with_booking_to_cancel)
        assert action.next_state == CallState.CLOSE
        assert "cancelled" in action.text.lower()

    @pytest.mark.asyncio
    async def test_no_keeps_booking_and_closes(self, deps, context_with_booking_to_cancel):
        state = ConfirmCancelState(deps)
        await state.enter(context_with_booking_to_cancel)
        action = await state.handle(make_transcription("no"), context_with_booking_to_cancel)
        assert action.next_state == CallState.CLOSE
        assert "still active" in action.text.lower()
