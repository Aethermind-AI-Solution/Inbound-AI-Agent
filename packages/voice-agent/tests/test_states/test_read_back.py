from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.read_back import ReadBackState
from packages.voice_agent.tests.test_states.conftest import make_transcription

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def context_with_booking(context):
    context.intent = "new_booking"
    context.slots.service_id = "s1"
    context.slots.service_name = "Haircut"
    context.slots.resource_id = "r1"
    context.slots.resource_name = "Priya"
    context.slots.datetime_ist = datetime(2026, 6, 10, 10, 0, tzinfo=IST)
    context.slots.hold_id = "h1"
    return context


class TestReadBackState:
    @pytest.mark.asyncio
    async def test_enter_reads_back_booking_details(self, deps, context_with_booking):
        state = ReadBackState(deps)
        action = await state.enter(context_with_booking)
        assert action.type == ActionType.ASK
        assert "Haircut" in action.text
        assert "Priya" in action.text
        assert "confirm" in action.text.lower()

    @pytest.mark.asyncio
    async def test_affirmative_transitions_to_confirm(self, deps, context_with_booking):
        state = ReadBackState(deps)
        await state.enter(context_with_booking)
        action = await state.handle(make_transcription("yes"), context_with_booking)
        assert action.next_state == CallState.CONFIRM

    @pytest.mark.asyncio
    async def test_negative_asks_what_to_change(self, deps, context_with_booking):
        state = ReadBackState(deps)
        await state.enter(context_with_booking)
        action = await state.handle(make_transcription("no"), context_with_booking)
        assert action.type == ActionType.ASK
        assert "change" in action.text.lower()

    @pytest.mark.asyncio
    async def test_unclear_twice_goes_to_callback(self, deps, context_with_booking):
        state = ReadBackState(deps)
        await state.enter(context_with_booking)
        await state.handle(make_transcription("hmm"), context_with_booking)
        action = await state.handle(make_transcription("umm"), context_with_booking)
        assert action.next_state == CallState.CALLBACK_CAPTURE
