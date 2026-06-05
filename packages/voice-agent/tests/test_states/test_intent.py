import pytest

from packages.voice_agent.config.models import AuthConfig
from packages.voice_agent.data.adapter import CallerInfo
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


class TestIntentAuthGating:
    @pytest.mark.asyncio
    async def test_authorized_intent_proceeds(self, deps, context):
        context.tenant_config.auth.action_policy = {"cancel": "soft"}
        context.caller = CallerInfo(
            id="c1", phone="+919876543210", tenant_id="t1",
            verified_at=None, is_new=False,
        )
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("cancel my booking"), context)
        assert action.next_state == CallState.LOOKUP_BOOKINGS

    @pytest.mark.asyncio
    async def test_unauthorized_intent_offers_alternative(self, deps, context):
        context.tenant_config.auth.action_policy = {"cancel": "soft", "new_booking": "none"}
        context.caller = CallerInfo(
            id="c1", phone="+919876543210", tenant_id="t1",
            verified_at=None, is_new=True,
        )
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("cancel my booking"), context)
        assert action.type == ActionType.ASK
        assert action.next_state is None

    @pytest.mark.asyncio
    async def test_unauthorized_twice_goes_to_callback(self, deps, context):
        context.tenant_config.auth.action_policy = {"cancel": "soft", "status": "soft"}
        context.caller = CallerInfo(
            id="c1", phone="+919876543210", tenant_id="t1",
            verified_at=None, is_new=True,
        )
        state = IntentState(deps)
        await state.enter(context)
        await state.handle(make_transcription("cancel my booking"), context)
        action = await state.handle(make_transcription("check my status"), context)
        assert action.next_state == CallState.CALLBACK_CAPTURE
        assert context.fallback_reason == "auth_required"
