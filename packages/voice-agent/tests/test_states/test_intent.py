import pytest

from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.intent import IntentState
from packages.voice_agent.tests.test_states.conftest import make_transcription


class TestIntentState:
    @pytest.mark.asyncio
    async def test_enter_asks_how_to_help(self, deps, context):
        state = IntentState(deps)
        action = await state.enter(context)
        assert action.type == ActionType.ASK
        assert "help" in action.text.lower()

    @pytest.mark.asyncio
    async def test_new_booking_intent(self, deps, context):
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("I want to book an appointment"), context)
        assert action.next_state == CallState.COLLECT_SERVICE
        assert context.intent == "new_booking"

    @pytest.mark.asyncio
    async def test_cancel_intent(self, deps, context):
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("cancel my booking"), context)
        assert action.next_state == CallState.LOOKUP_BOOKINGS
        assert context.intent == "cancel"

    @pytest.mark.asyncio
    async def test_reschedule_intent(self, deps, context):
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("reschedule please"), context)
        assert action.next_state == CallState.LOOKUP_BOOKINGS
        assert context.intent == "reschedule"

    @pytest.mark.asyncio
    async def test_status_intent(self, deps, context):
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("check my booking"), context)
        assert action.next_state == CallState.LOOKUP_BOOKINGS
        assert context.intent == "status"

    @pytest.mark.asyncio
    async def test_unknown_intent_reprompts(self, deps, context):
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("the weather is nice"), context)
        assert action.type == ActionType.ASK
        assert action.next_state is None

    @pytest.mark.asyncio
    async def test_unknown_three_times_goes_to_callback(self, deps, context):
        state = IntentState(deps)
        await state.enter(context)
        await state.handle(make_transcription("blah"), context)
        await state.handle(make_transcription("blah"), context)
        action = await state.handle(make_transcription("blah"), context)
        assert action.next_state == CallState.CALLBACK_CAPTURE
